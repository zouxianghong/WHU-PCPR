# Author: Jacek Komorowski
# Warsaw University of Technology

import os
import configparser
import time
import numpy as np
import random
import copy
import pickle
import open3d as o3d
import torch
from torchpack.utils.config import configs

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate


class ListDict(object):
    def __init__(self, items=None):
        if items is not None:
            self.items = copy.deepcopy(items)
            self.item_to_position = {item: ndx for ndx, item in enumerate(items)}
        else:
            self.items = []
            self.item_to_position = {}

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        # Add gating as the last step
        if 'vlad' in self.model.lower():
            self.cluster_size = params.getint('cluster_size', 64)   # Size of NetVLAD cluster
            self.gating = params.getboolean('gating', True)         # Use gating after the NetVlad

        #######################################################################
        # Model dependent
        #######################################################################

        if 'MinkFPN' in self.model:
            # Models using MinkowskiEngine
            self.mink_quantization_size = params.getfloat('mink_quantization_size')
            # Size of the local features from backbone network (only for MinkNet based models)
            # For PointNet-based models we always use 1024 intermediary features
            self.feature_size = params.getint('feature_size', 256)
            if 'planes' in params:
                self.planes = [int(e) for e in params['planes'].split(',')]
            else:
                self.planes = [32, 64, 64]

            if 'layers' in params:
                self.layers = [int(e) for e in params['layers'].split(',')]
            else:
                self.layers = [1, 1, 1]

            self.num_top_down = params.getint('num_top_down', 1)
            self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def xyz_from_depth(depth_image, depth_intrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = np.nan
    return xyz


class MinkLocParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """
    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.num_points = params.getint('num_points', 4096)
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 128)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.normalize_embeddings = params.getboolean('normalize_embeddings', True)    # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)

        self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                    'residential_evaluation_database.pickle', 'university_evaluation_database.pickle']

        self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                 'residential_evaluation_query.pickle', 'university_evaluation_query.pickle']

        assert len(self.eval_database_files) == len(self.eval_query_files)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')


def save_pickle(data, pickle_path):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save ', pickle_path)


def load_pickle(pickle_path):
    print('Load ', pickle_path)
    if os.path.exists(pickle_path):
       pass 
    elif os.path.exists(os.path.join(configs.data.dataset_folder, pickle_path)):
        pickle_path = os.path.join(configs.data.dataset_folder, pickle_path)
    else:
        raise FileNotFoundError(f"Error: Pickle path {pickle_path} not found in dataset folder or on absolute path")

    with open(pickle_path, 'rb') as f:
        pickle_opened = pickle.load(f)
    return pickle_opened


def l2_norm_np(x, axis=-1):
    '''
    '''
    if x is None:
        return None
    x = x / np.linalg.norm(x, axis=axis, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=None, neginf=None)
    return x


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def EMA_model(model_k, model_q, alpha=0.99):
    ''' model_k = model_k * α + model_q * (1 - α) '''
    for param_k, param_q in zip(model_k.parameters(), model_q.parameters()):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)


def make_sparse_tensor(lidar_pc, voxel_size=0.05, lidar_feat=None):
    # get rounded coordinates
    lidar_pc = lidar_pc.numpy()
    lidar_pc = np.hstack((lidar_pc, np.zeros((len(lidar_pc),1), dtype=np.float32)))
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc if lidar_feat is None else lidar_feat

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    coords = coords[indices]
    feats = feats[indices]
    points = torch.from_numpy(lidar_pc[indices][:, :3])

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    return inputs, indices


def sparcify_and_collate_list(list_data, voxel_size, list_feat=None):
    outputs, indices = [], []
    if list_feat is None:
        for xyzr in list_data:
            out, idxs = make_sparse_tensor(xyzr, voxel_size)
            outputs.append(out)
            indices.append(idxs)
    else:
        for xyzr, feat in zip(list_data, list_feat):
            out, idxs = make_sparse_tensor(xyzr, voxel_size, feat)
            outputs.append(out)
            indices.append(idxs)
    outputs =  sparse_collate(outputs)
    outputs.C = outputs.C.int()
    return outputs, indices


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_files(dir, ext_filter=None, ignore_sub_dirs=True):
    if not os.path.exists(dir):
        return []
    out_files = []
    for root, directories, files in os.walk(dir):
        if root != dir and ignore_sub_dirs:
            continue
        for filename in files:
            if os.path.splitext(filename)[1] == ext_filter:
                filepath = os.path.join(root, filename)
                out_files.append(filepath)
    return out_files


def split_filepath(filepath):
    dir_name = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    ss = file_name.split('.')
    file_name_no_ext, file_ext = ss[0], ss[1]
    return dir_name, file_name_no_ext, file_ext


def hashM(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


def load_pc(pc_file):
    if '.bin' in pc_file:
        pc = np.fromfile(pc_file, dtype=np.float64)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    elif '.npy' in pc_file:
        pc = np.load(pc_file)[:,:3]
    return pc


def normalize_point_cloud(pc):
    """ Normalize point cloud to [-1.0, 1.0]
    """
    pc.reshape([-1, 3])
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale
    return pc, centroid, scale


def nn_dist(c):
    # c: m x 3, or b x m x 3; Return: m x m, or b x m x m
    if len(c.shape) == 2:
        c1 = torch.unsqueeze(c, dim=1)
        c2 = c[None, ...]
    elif len(c.shape) == 3:
        c1 = torch.unsqueeze(c, dim=2)
        c2 = c[:, None, ...]
    return torch.sum((c1 - c2)**2, dim=-1) ** 0.5


def nn_dist_np(c):
    """ c: n x d """
    c1 = c[:, None, :]  # n x 1 x d
    c2 = c[None, ...]  # 1 x n x d
    return np.sum((c1 - c2)**2, axis=-1) ** 0.5  # n x n


def make_open3d_point_cloud(xyz, color=None, tile=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if tile:
            if len(color) != len(xyz):
                color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def get_matching_indices(source, target, search_voxel_size, K=None):
    src_center = np.mean(source, axis=0, keepdims=True)
    source = source - src_center
    target = target - src_center
    source = make_open3d_point_cloud(source)
    target = make_open3d_point_cloud(target)
    
    # source_copy = copy.deepcopy(source)
    # target_copy = copy.deepcopy(target)
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(
            point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append([i, j])
    return np.asarray(match_inds)  # N x 2


def normalize(x, p=2, dim=-1, eps=1e-12):
    """
    与torch.nn.functional.normalize功能相同的NumPy实现
    
    参数:
        x: 输入数组
        p: 范数类型 (1, 2, inf, -inf, 或其他实数)
        dim: 沿哪个维度进行归一化
        eps: 防止除零的小数值
    
    返回:
        归一化后的数组
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 计算指定维度的p范数
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    
    # 防止除零
    norm = np.maximum(norm, eps)
    
    # 归一化
    normalized = x / norm
    
    return normalized