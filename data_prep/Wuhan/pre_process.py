import os 
import argparse 
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import multiprocessing as mp


def get_pointcloud_tensor(xyz):
    # normalize
    m = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - m
    s = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / s

    return xyz, m, s


def process_pointcloud(ARGS):
    pc_path, source_dir, save_dir = ARGS
    if not os.path.isfile(pc_path):
        return None
    xyz = np.load(pc_path).reshape(-1,3)
    if len(xyz) == 0:
        return None
    if len(xyz) < 4096:
        add_idxs = np.random.choice(len(xyz), size=4096-len(xyz), replace=True)
        xyz = np.vstack([xyz, xyz[add_idxs]])
    xyz, m, s = get_pointcloud_tensor(xyz)
    save_path = pc_path.replace(source_dir, save_dir).split('.')[0]
    np.save(save_path, xyz)
    return m, s


def multiprocessing_preprocessing(run, source_dir, save_dir):
    # Prepare inputs 
    clouds_raw = sorted(glob(os.path.join(run, '*')))
    ARGS = [[c, source_dir, save_dir] for c in clouds_raw]

    # Multiprocessing the pre-processing
    with mp.Pool(32) as p:
        m_s = list(tqdm(p.imap(process_pointcloud, ARGS), total = len(ARGS)))
    return m_s


def global_csv_to_northing_easting(csv_path, source_dir, save_dir, m_s=None):
    df = pd.read_csv(csv_path, sep=',')
    df = df.sort_values('timestamp')  # sort by time stamp
    df.reset_index(drop=True)

    if m_s is not None:
        ms_df = pd.DataFrame(columns=['mx', 'my', 'mz', 'scale'])
        for x in m_s:
            new_row = [x[0][0,0], x[0][0,1], x[0][0,2], x[1]]
            new_row = pd.DataFrame([new_row], columns=ms_df.columns)
            ms_df = pd.concat([ms_df, new_row], ignore_index=True)
        df = pd.concat([df, ms_df], axis=1)

    # # Debug: check file exist
    # num_not_exist = 0
    # parent_dir = csv_path.split('.')[0]
    # print(f'-------{parent_dir}-------')
    # for row in df.itertuples():
    #     cur_file = os.path.join(parent_dir, f'{row.timestamp}.bin')
    #     if not os.path.isfile(cur_file):
    #         print(f"File not exist: {row.timestamp}")
    #         num_not_exist += 1
    # print(f'-------{parent_dir}: {num_not_exist}-------')
    
    save_path = os.path.join(csv_path).replace(source_dir, save_dir)
    df.to_csv(save_path, index=0, float_format='%.6f')


def process_Wuhan(root, save_dir, setting):
    environments = ['Hankou','Campus']
    for env in environments:
        for run in os.listdir(os.path.join(root, env)):
            if not os.path.isdir(os.path.join(root, env, run, setting)):
                continue
            print(os.path.join(save_dir, env, run, setting))
            if not os.path.exists(os.path.join(save_dir, env, run, setting)):
                os.makedirs(os.path.join(save_dir, env, run, setting))
            m_s = multiprocessing_preprocessing(os.path.join(root, env, run, setting), root, save_dir)
            global_csv_to_northing_easting(os.path.join(root, env, run, '{}.csv'.format(setting)), root, save_dir, m_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, help = 'Root for Wuhan Dataset')
    parser.add_argument('--save_dir', type = str, required = True, help = 'Directory to save pre-processed data to')
    parser.add_argument('--setting', type = str, required = False, default="pointcloud_30m_2m", help = 'Directory to save pre-processed data to')
    args = parser.parse_args()

    process_Wuhan(args.root, args.save_dir, args.setting)
