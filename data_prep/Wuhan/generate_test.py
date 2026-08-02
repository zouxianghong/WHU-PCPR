# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import csv
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm

# For Wuhan: Hankou (easting, northing)
def read_xy_from_csv(filename, header=False, delimiter=','):
    points = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader, None)  # 跳过标题行
        for line_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue  # 忽略字段不足的行
            try:
                y = float(row[0].strip())
                x = float(row[1].strip())
            except ValueError:
                # 若某行格式错误，可选择忽略或抛出异常
                print(f"警告: 第 {line_num} 行包含非数值数据，已跳过。")
                continue
            points.append([y, x])
    return points

P = read_xy_from_csv('/home/ericxhzou/Data/WHU-PCGL/PublishData-V2/test_region.csv', header=False, delimiter=',')

do_check = False
ENVS = ['Hankou','Campus']


def construct_query_and_database_sets(base_path, runs, save_dir, file_extension, p, output_name, setting=None, do_check = False):
    database_trees = []
    test_trees = []
    for run in runs:
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, run, "{}.csv".format(setting)), sep=',')
        for index, row in df_locations.iterrows():
            if not do_check:
                df_test = df_test._append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p):
                df_test = df_test._append(row, ignore_index=True)
            df_database = df_database._append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    query_sets = []
    database_sets = []
    for run in runs:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, run, f"{setting}.csv"), sep=',')
        df_locations['timestamp'] = base_path + '/' + run + '/' + setting + '/' + df_locations['timestamp'].astype(str) + file_extension
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        
        for index, row in df_locations.iterrows():
            if not do_check:
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'],
                                          'mx': row['mx'], 'my': row['my'], 'mz': row['mz'], 'scale': row['scale']}
            elif check_in_test_set(row['northing'], row['easting'], p):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'],
                                          'mx': row['mx'], 'my': row['my'], 'mz': row['mz'], 'scale': row['scale']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'],
                                          'mx': row['mx'], 'my': row['my'], 'mz': row['mz'], 'scale': row['scale']}
        database_sets.append(database)
        query_sets.append(test)

    for i in tqdm(range(len(database_sets))):
        tree = database_trees[i]
        for j in range(len(query_sets)):
            if i == j:
                continue
            for key in range(len(query_sets[j].keys())):
                coor = np.array([[query_sets[j][key]["northing"], query_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=25)
                # indices of the positive matches in database i of each query (key) in test set j
                query_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, save_dir, f'{output_name}_eval_database_{setting}.pickle')
    output_to_file(query_sets, save_dir, f'{output_name}_eval_query_{setting}.pickle')


def check_in_test_set(northing, easting, points, x_width=60, y_width=60):
    in_test_set = False
    for point in points:
        if point[0] - x_width < northing < point[0] + x_width and point[1] - y_width < easting < point[1] + y_width:
            in_test_set = True
            break
    return in_test_set

def output_to_file(output, save_dir, filename):
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    parser = argparse.ArgumentParser(description='Generate Inhouse Training Dataset')
    parser.add_argument('--root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--eval_thresh', type = int, default = 30, help = 'Threshold for positive examples')
    parser.add_argument('--file_extension', type = str, default = '.npy', help = 'File extension expected')
    parser.add_argument('--setting', type = str, required = False, default="pointcloud_30m_2m", help = 'Directory to save pre-processed data to')
    parser.add_argument('--save_dir', type = str, required = True, help = 'Folder to save pickle files to')
    args = parser.parse_args()

    # Check dataset root exists, make save dir if doesn't exist
    print('Dataset root: {}'.format(args.root))
    assert os.path.exists(args.root), f"Cannot access dataset root folder: {args.root}"
    base_path = args.root
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # validation: Hankou1 & Hankou2, query in test region
    ENV = 'Hankou'
    runs = ['1', '2']
    construct_query_and_database_sets(os.path.join(base_path, ENV), runs, args.save_dir, args.file_extension, P, f'{ENV}_1_2', setting=args.setting, do_check=True)
    
    # test: Hankou1 & Hankou3
    ENV = 'Hankou'
    runs = ['1', '3']
    construct_query_and_database_sets(os.path.join(base_path, ENV), runs, args.save_dir, args.file_extension, P, f'{ENV}_1_3', setting=args.setting, do_check=False)
    
    # test: Hankou2 & Hankou3
    ENV = 'Hankou'
    runs = ['2', '3']
    construct_query_and_database_sets(os.path.join(base_path, ENV), runs, args.save_dir, args.file_extension, P, f'{ENV}_2_3', setting=args.setting, do_check=False)
    
    # test: Campus1 & Campus2
    ENV = 'Campus'
    runs = ['1', '2']
    construct_query_and_database_sets(os.path.join(base_path, ENV), runs, args.save_dir, args.file_extension, P, f'{ENV}_1_2', setting=args.setting, do_check=False)
    
    # test: Campus1 & Campus3
    ENV = 'Campus'
    runs = ['1', '3']
    construct_query_and_database_sets(os.path.join(base_path, ENV), runs, args.save_dir, args.file_extension, P, f'{ENV}_1_3', setting=args.setting, do_check=False)
    
    # test: Campus2 & Campus3
    ENV = 'Campus'
    runs = ['2', '3']
    construct_query_and_database_sets(os.path.join(base_path, ENV), runs, args.save_dir, args.file_extension, P, f'{ENV}_2_3', setting=args.setting, do_check=False)
