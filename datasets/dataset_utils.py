# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import random
import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
from torchpack.utils.config import configs

from datasets.oxford import OxfordDataset, TrainTransform
from datasets.samplers_inc import BatchSampler
from misc.utils import sparcify_and_collate_list, get_matching_indices
from vis.draw_result import draw_pc_pps


def make_dataset(pickle_file, aug_mode=None):
    # Create training and validation datasets
    train_transform = TrainTransform(aug_mode)

    print(f'Creating Dataset from pickle file : {pickle_file}')

    dataset = OxfordDataset(configs.data.dataset_folder, pickle_file,
                            transform=train_transform, transform_now=configs.model.name != 'LoGG3DNet')
  
    return dataset


def make_collate_fn(dataset: OxfordDataset):

    def collate_fn(data_list):
        # Constructs a batch object
        clouds, labels = [e[0] for e in data_list], [e[1] for e in data_list]
        means, scales = [e[2] for e in data_list], [e[3] for e in data_list]
        
        clouds = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
        means = torch.stack(means, dim=0)
        scales = torch.stack(scales, dim=0)
        
        if configs.model.quantizer is None:  # Not a MinkowskiEngine based model
            if configs.model.name == 'NDTTransformer':
                raise NotImplementedError('NDTTransformer require preprocessed data!')
            else:
                batch = {'cloud': clouds, 'mean': means, 'scale': scales}
        else:
            if configs.model.name == 'LoGG3DNet':
                import copy
                assert not dataset.transform_now
                origin_clouds = copy.deepcopy(clouds.detach().cpu().numpy())
                if dataset.transform is not None:
                    for x in range(len(clouds)):
                        clouds[x] = dataset.transform(clouds[x])
                inputs, indices = sparcify_and_collate_list(clouds, configs.model.quantization_size)
                # pos pairs
                pos_pairs = []
                for x in range(0, len(clouds), 2):
                    # q_pc1 = origin_clouds[x][indices[x]] * scales[x].detach().cpu().numpy() + means[x].detach().cpu().numpy()
                    # p_pc1 = origin_clouds[x+1][indices[x+1]] * scales[x+1].detach().cpu().numpy() + means[x+1].detach().cpu().numpy()
                    # pos_pairs_x = get_matching_indices(q_pc1, p_pc1, search_voxel_size=0.5)
                    # q_kpt1 = q_pc1[pos_pairs_x[:, 0]]
                    # p_kpt1 = p_pc1[pos_pairs_x[:, 1]]
                    # draw_pc_pps(q_pc1, q_kpt1, p_pc1, p_kpt1, offset_x=0.0, save_filepath='/home/ericxhzou/Code/PCGL-Benchmark/exp/pr/pq1.png')
                    
                    # q_pc2 = clouds[x][indices[x]]
                    # p_pc2 = clouds[x+1][indices[x+1]]
                    # q_pc2 = q_pc2.detach().cpu().numpy()
                    # p_pc2 = p_pc2.detach().cpu().numpy()
                    # pos_pairs_x = get_matching_indices(q_pc2, p_pc2, search_voxel_size=0.015)
                    pos_pairs_x = []
                    # q_kpt2 = q_pc2[pos_pairs_x[:, 0]]
                    # p_kpt2 = p_pc2[pos_pairs_x[:, 1]]
                    # draw_pc_pps(q_pc2, q_kpt2, p_pc2, p_kpt2, offset_x=0.0, save_filepath='/home/ericxhzou/Code/PCGL-Benchmark/exp/pr/pq2.png')
                    
                    pos_pairs.append(pos_pairs_x)
                batch = {'inputs': inputs, 'mean': means, 'scale': scales, 'pos_pairs': pos_pairs}
            else:
                coords = [configs.model.quantizer(e)[0] for e in clouds]
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                batch = {'coords': coords, 'features': feats, 'cloud': clouds, 'mean': means, 'scale': scales}

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, list(dataset.queries[label].positives)) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch, positives_mask, negatives_mask

    return collate_fn


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_dataloader(pickle_file, aug_mode=None):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements

    :return:
    """
    dataset = make_dataset(pickle_file, aug_mode)

    train_sampler = BatchSampler(dataset, batch_size=configs.train.batch_size,
                                 batch_size_limit=configs.train.batch_size_limit,
                                 batch_expansion_rate=configs.train.batch_expansion_rate)

    # Reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(dataset)
    dataloader = DataLoader(dataset, batch_sampler=train_sampler, collate_fn=train_collate_fn,
                            num_workers=configs.train.num_workers, pin_memory=configs.data.pin_memory,
                            worker_init_fn = seed_worker, generator = g)

    return dataloader


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    if len(array) == 0:
        return False
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
