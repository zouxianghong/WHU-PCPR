import os 
import torch 
import numpy as np
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from torchpack.utils.config import configs
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools

from datasets.oxford import TrainTransform
from misc.utils import sparcify_and_collate_list
from baseline.pr.ScanContext.python.make_sc_example import ScanContext
from baseline.pr.ScanContext.python.Distance_SC import distance_sc


class EvalDataset:
    def __init__(self, dataset_dict, aug_mode=None):
        self.set = dataset_dict 
        self.n_points = 4096
        self.root = configs.data.dataset_folder
        self.transform = None
        if aug_mode is not None:
            self.transform = TrainTransform(aug_mode, fixed_theta=configs.eval.rotate_theta)

    def load_pc(self, filename):
        if not os.path.exists(filename):
            filename = os.path.join(self.root, filename)
        if '.bin' in filename:
            pc = np.fromfile(filename, dtype = np.float64)
            # coords are within -1..1 range in each dimension
            assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(filename)
            pc = np.reshape(pc, (pc.shape[0] // 3, 3))
            pc = torch.tensor(pc, dtype=torch.float)
        elif '.npy' in filename:
            try:
                pc = np.load(filename)[:,:3]
                assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(filename)
                pc = torch.tensor(pc, dtype = torch.float)
            except:
                print(filename)
                pc = np.load(filename)[:,:3]
                assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(filename)
                pc = torch.tensor(pc, dtype = torch.float)
        return pc

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        pc = self.load_pc(self.set[idx]['query'])
        
        if configs.model.name == 'ScanContext':
            if 'mx' in self.set[idx]:
                mean = np.array([self.set[idx]['mx'], self.set[idx]['my'], self.set[idx]['mz']])
                scale = np.array([self.set[idx]['scale']])
            else:
                mean = np.array([0.0, 0.0, 0.0])
                scale = np.array([0.0])
            mean = torch.tensor(mean.reshape(1,3), dtype=torch.float)
            scale = torch.tensor(scale.reshape(1,1), dtype=torch.float)
            center = np.array([self.set[idx]['easting'], self.set[idx]['northing'], 0.0])
            center = torch.tensor(center.reshape(1,3), dtype=torch.float)
            pc = pc * scale + mean - center
        
        if self.transform is not None:
            pc = self.transform(pc)
        return pc


def get_eval_dataloader(dataset_dict, aug_mode=None):
    dataset = EvalDataset(dataset_dict, aug_mode)

    def collate_fn(data_list):
        clouds = [e for e in data_list]
        clouds = torch.stack(clouds, dim = 0)

        if configs.model.quantizer is None:  # Not a MinkowskiEngine based model
            if configs.model.name == 'NDTTransformer':
                raise NotImplementedError('NDTTransformer require preprocessed data!')
            else:
                batch = {'cloud': clouds}
        else:
            if configs.model.name == 'LoGG3DNet':
                inputs, indices = sparcify_and_collate_list(clouds, configs.model.quantization_size)
                points = []
                for x in range(len(indices)):
                    points.append(clouds[x][indices[x]])
                batch = {'inputs': inputs, 'points': points}
            else:
                coords = [configs.model.quantizer(e)[0] for e in clouds]
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                batch = {'coords': coords, 'features': feats, 'cloud': clouds}

        return batch

    dataloader = DataLoader(
        dataset,
        batch_size = configs.eval.batch_size,
        shuffle = False, 
        collate_fn = collate_fn,
        num_workers = configs.train.num_workers
    )

    return dataloader 


@torch.no_grad()
def get_latent_vectors(model, dataset_dict, aug_mode=None, parallel=True):
    g_descs, l_kpts, l_descs = [], [], []
    dataloader = get_eval_dataloader(dataset_dict, aug_mode)
    
    if configs.model.name == 'ScanContext':
        sc = ScanContext(configs.model.num_sector, configs.model.num_ring, configs.model.max_length, configs.model.lidar_height)
        if parallel:
            pcs = []
            for i in range(len(dataloader.dataset)):
                pc = dataloader.dataset[i]
                pcs.append(pc)
            with Pool(processes=min(cpu_count(), len(pcs))) as pool:
                scs_list = list(tqdm(
                    pool.imap(sc.gen_sc, pcs),
                    total=len(pcs),
                    desc="Generate ScanContext"
                ))
            bevs = np.vstack([np.expand_dims(sc, axis=0) for sc in scs_list])
            g_descs = bevs.reshape(bevs.shape[0], configs.model.num_ring * configs.model.num_sector)
        else:
            for idx, batch in enumerate(dataloader):
                batch = {x: batch[x].to('cuda') for x in batch}
                for x in range(batch['cloud'].shape[0]):
                    pc = batch['cloud'][x]
                    bev = sc.gen_sc(pc.cpu().numpy())
                    g_desc = bev.reshape(configs.model.num_ring * configs.model.num_sector)
                    g_descs.append(np.expand_dims(g_desc, axis=0))
            g_descs = np.vstack(g_descs)
    else:
        model.eval()
        for idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                batch = {x: batch[x].to('cuda') if x != 'points' else batch[x] for x in batch}
                if 'points' in batch:
                    batch['points'] = [batch['points'][x].to('cuda') for x in range(len(batch['points']))]
            else:
                batch = batch.to('cuda')

            out = model(batch)
            g_desc = out
            if configs.model.name == 'EgoNN' or configs.model.name == 'PPTNet':
                g_desc, l_kpt, l_desc = out['global'], out['keypoints'], out['descriptors']
                for x in range(len(l_kpt)):
                    l_kpts.append(l_kpt[x].cpu().numpy())
                    l_descs.append(l_desc[x].cpu().numpy())
            elif configs.model.name == 'LoGG3DNet':
                g_desc, l_kpt, l_desc = out[0], batch['points'], out[1]
                for x in range(len(l_kpt)):
                    l_kpts.append(l_kpt[x].cpu().numpy())
                    l_descs.append(l_desc[x].cpu().numpy())
            if len(g_desc.shape) == 1:
                g_desc = torch.unsqueeze(g_desc, dim=0)
            if configs.model.normalize_embeddings:
                g_desc = F.normalize(g_desc, dim=-1)
            g_descs += list(g_desc.cpu().numpy())
        g_descs = np.vstack(g_descs)
    return g_descs, l_kpts, l_descs


def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()


def cosine_dist(query, database):
    return np.array(1 - torch.einsum('D,ND->N', torch.tensor(query), torch.tensor(database)))


def scan_context_dist(query, database):
    q = query.reshape(configs.model.num_ring, configs.model.num_sector)
    d = database.reshape(configs.model.num_ring, configs.model.num_sector)
    dist = distance_sc(q, d)
    return dist


def scan_context_kneighbors(query, database, k):
    dist_func = functools.partial(scan_context_dist, query)
    with Pool(processes=min(cpu_count(), len(database))) as pool:
        dists = pool.map(dist_func, database)
    
    dists = np.array(dists)
    indices = np.argsort(dists)[:k]
    return indices