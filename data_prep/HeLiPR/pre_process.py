import os 
import csv 
import argparse 
import numpy as np 
import struct
import pandas as pd 
import open3d as o3d 
from tqdm import tqdm 
from glob import glob 
import multiprocessing as mp

np.random.seed(0)


def read_bin_file(filename):
    points = []

    with open(filename, 'rb') as file:
        while True:
            data = file.read(16)
            if len(data) < 16:
                break
            x, y, z, intensity = struct.unpack('ffff', data[:16])
            points.append([x, y, z])
    
    return np.stack(points, axis=0)


def remove_ground_plane(xyz):
    # Make open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:,:3])

    # Remove ground plane 
    _, inliers = pcd.segment_plane(0.2, 3, 250)
    not_ground_mask = np.ones(len(xyz), bool)
    not_ground_mask[inliers] = 0
    xyz = xyz[not_ground_mask]
    return xyz 

def downsample_point_cloud(xyz, voxel_size=0.05):
    # Make xyz pointcloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:,:3])

    # Downsample point cloud using open3d functions
    pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    return np.asarray(pcd_ds.points)

def pnv_preprocessing(xyz, num_points = 4096, vox_sz = 0.1, dist_thresh = 25):

    # Cut off points past dist_thresh
    dist = np.linalg.norm(xyz[:,:3], axis=1)
    xyz = xyz[dist <= dist_thresh]

    # Slowly increase voxel size until we have less than num_points
    while len(xyz) > num_points:
      xyz = downsample_point_cloud(xyz, vox_sz)
      vox_sz += 0.01
    # Re-sample some points to bring to num_points if under num_points 
    ind = np.arange(xyz.shape[0])
    if num_points - len(ind) > 0:
        # print('xyz shape: ', xyz.shape[0], 'sample: ', num_points - len(ind))
        extra_points_ind = np.random.choice(xyz.shape[0], num_points - len(ind), replace = False)
        ind = np.concatenate([ind, extra_points_ind])
    xyz = xyz[ind,:]
    assert len(xyz) == num_points

    # Regularize xyz to be between -1, 1 in x,y planes 
    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m
    return xyz 

def get_pointcloud_tensor(xyz):

    # Filter our points near the origin 
    r = np.linalg.norm(xyz[:,:3], axis = 1)
    r_filter = np.logical_and(r > 0.1, r < 80)
    xyz = xyz[r_filter]

    # Remove ground plane and pre-process  
    # xyz = remove_ground_plane(xyz)
    xyz = pnv_preprocessing(xyz)

    return xyz

def process_pointcloud(ARGS):
    pc_path, source_dir, save_dir = ARGS 
    xyz = read_bin_file(pc_path)
    if len(xyz) == 0:
        return None  
    xyz = get_pointcloud_tensor(xyz)
    save_path = pc_path.replace(source_dir, save_dir).split('.')[0]
    np.save(save_path, xyz)

def multiprocessing_preprocessing(raw_data_sir, source_dir, save_dir):
    # Prepare inputs 
    clouds_raw = sorted(glob(os.path.join(raw_data_sir, '*')))
    ARGS = [[c, source_dir, save_dir] for c in clouds_raw]
    
    # Multiprocessing the pre-processing 
    with mp.get_context('spawn').Pool(32) as p:
        _ = list(tqdm(p.imap(process_pointcloud, ARGS), total = len(ARGS)))

def global_txt_to_northing_easting(txt_path, sensor, source_dir, save_dir):
    df = pd.DataFrame(columns = ['timestamp', 'northing', 'easting'])
    scan_timestamps = sorted([x.split('.')[0] for x in os.listdir(os.path.join(os.path.dirname(os.path.dirname(txt_path)), 'LiDAR_submap', sensor))])
    with open(txt_path, 'r') as f:
        reader = csv.reader(f, delimiter = ' ')
        for idx, row in tqdm(enumerate(reader)):
            new_row = [scan_timestamps[idx], row[2], row[1]]
            df.loc[idx] = new_row 

    # Save new dataframe 
    df = df[df.timestamp != '1567496784952532897'] # Exclude edge case faulty scan 
    save_path = os.path.join(f'{os.path.dirname(os.path.dirname(txt_path))}_{sensor}', 'pd_northing_easting.csv').replace(source_dir, save_dir)
    df.to_csv(save_path)


def process_HeliPR(root, save_dir):
    environments = ['Bridge', 'KAIST', 'Roundabout', 'Town']  # 'Bridge', 'DCC', 'KAIST', 'Riverside', 'Roundabout', 'Town'
    for env in environments:
        for run in os.listdir(os.path.join(root, env)): # ENV01, ENV02, ENV03
            for sensor in os.listdir(os.path.join(root, env, run, 'LiDAR_submap')):
                cur_dir = os.path.join(root, env, run, 'LiDAR_submap', sensor)
                cur_file = os.path.join(root, env, run, 'LiDAR_submap', f'{sensor}.csv')
                if not os.path.isdir(cur_dir) or not os.path.exists(cur_file):
                    continue
                print(cur_dir)
                save_dir_i = os.path.join(f'{os.path.join(root, env, run)}_{sensor}'.replace(root, save_dir), sensor)
                if not os.path.exists(save_dir_i):
                    os.makedirs(save_dir_i)
                global_txt_to_northing_easting(cur_file, sensor, root, save_dir)
                multiprocessing_preprocessing(cur_dir, source_dir=cur_dir, save_dir=save_dir_i)


def bin_to_txt(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    clouds_raw = sorted(glob(os.path.join(in_dir, '*')))
    for in_file in clouds_raw:
        pc = np.load(in_file)
        filename = os.path.basename(in_file).split('.')[0]
        out_file = os.path.join(out_dir, f'{filename}.txt')
        with open(out_file, 'w') as f:
            for i in range(pc.shape[0]):
                f.write(f'{pc[i, 0]} {pc[i, 1]} {pc[i, 2]} \n')


if __name__ == '__main__':
    # # debug
    # in_dir = '/home/ericxhzou/HardDisk/HeliPR/Processed/KAIST/KAIST04_Avia/Avia'
    # out_dir = '/home/ericxhzou/HardDisk/HeliPR/Processed/KAIST/KAIST04_Avia/Avia_txt'
    # bin_to_txt(in_dir, out_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default='/home/ericxhzou/HardDisk/HeliPR', help = 'Root for HeliPR Dataset')
    parser.add_argument('--save_dir', type = str, default='/home/ericxhzou/HardDisk/HeliPR/Processed', help = 'Directory to save pre-processed data to')
    args, opts = parser.parse_known_args()

    process_HeliPR(args.root, args.save_dir)