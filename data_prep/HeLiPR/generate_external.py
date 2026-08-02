import os
import pandas as pd
import pickle
import argparse


FILENAME = "pd_northing_easting.csv"
ENVS = ['Bridge', 'KAIST', 'Roundabout', 'Town']  # 'Bridge', 'DCC', 'KAIST', 'Riverside', 'Roundabout', 'Town'
RUNS = ['01', '04']
SENSORS = ['Velodyne', 'Ouster', 'Aeva', 'Avia']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, default='/home/ericxhzou/HardDisk/HeliPR/Processed', help='Dataset root folder')
    parser.add_argument('--save_folder', type = str, default='/home/ericxhzou/HardDisk/HeliPR/Processed', help = 'Folder to save pickle files to')
    parser.add_argument('--file_extension', type = str, default = '.npy', help = 'File extension expected')
    args, opts = parser.parse_known_args()
    
    # Check dataset root exists, make save dir if doesn't exist
    print('Dataset root: {}'.format(args.dataset_root))
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        
    # construct external data dict
    datum = {}
    for ENV in ENVS:
        folders = [f'{ENV}{RUN}' for RUN in RUNS]
        for SENSOR in SENSORS:
            for folder in folders:
                csv_file = os.path.join(base_path, ENV, f'{folder}_{SENSOR}', FILENAME)
                if not os.path.exists(csv_file):
                    continue
                df_locations = pd.read_csv(csv_file, sep = ',')
                df_locations['timestamp'] = base_path + '/' + ENV + '/' + f'{folder}_{SENSOR}' + '/' + SENSOR + '/' + df_locations['timestamp'].astype(str) + args.file_extension
                df_locations = df_locations.rename(columns = {'timestamp': 'file'})
                for index, row in df_locations.iterrows():
                    datum[len(datum)] = row['file']
    
    filename = 'HeLiPR_external.pickle'
    file_path = os.path.join(args.save_folder, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(datum, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done: ", filename, " Data Size: ", len(datum))
    