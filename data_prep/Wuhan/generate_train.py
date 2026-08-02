# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm
from datasets.oxford import TrainingTuple
# Import test set boundaries
from data_prep.Wuhan.generate_test import P, check_in_test_set


def construct_query_dict(df_centroids, save_dir, filename, ind_nn_r, ind_r_r):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        mean = np.array(df_centroids.iloc[anchor_ndx][['mx', 'my', 'mz']], dtype=np.float64)
        scale = np.float64(df_centroids.iloc[anchor_ndx]["scale"])
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        # assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos,
                                            mean=mean, scale=scale)

    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':
    ENVS = ['Hankou']

    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--pos_thresh', type = int, default = 15, help = 'Threshold for positive examples')
    parser.add_argument('--neg_thresh', type = int, default = 60, help = 'Threshold for negative examples')
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

    for env in ENVS:
        df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
        
        runs = ['1', '2']
        for run in runs:
            df_locations = pd.read_csv(os.path.join(base_path, env, run, f"{args.setting}.csv"), sep = ',')
            df_locations['timestamp'] = base_path + '/' + env + '/' + run + '/' + args.setting + '/' + df_locations['timestamp'].astype(str) + args.file_extension
            df_locations = df_locations.rename(columns = {'timestamp': 'file'})
        
            for index, row in df_locations.iterrows():
                if check_in_test_set(row['northing'], row['easting'], P):
                    df_test = df_test._append(row, ignore_index=True)
                else:
                    df_train = df_train._append(row, ignore_index=True)
        
        print(f'Train Samples for Environment {env} : {len(df_train)}')
        print("Number of training submaps: " + str(len(df_train['file'])))
        print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
        construct_query_dict(df_train, args.save_dir, f'{env}_1_2_train_queries_{args.setting}.pickle', args.pos_thresh, args.neg_thresh)
