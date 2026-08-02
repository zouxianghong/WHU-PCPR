import os
import random
import numpy as np
from collections import deque

from sklearn.neighbors import KDTree
from torchpack.utils.config import configs
from misc.utils import load_pickle, load_pc, normalize, ensure_dir
from vis.draw_result import draw_pc


class RerankDataset():
    """ Dataset for reranking the top k place recognition results """
    def __init__(self, env, query_filename, database_filename, pr_backbone, k=25):
        super(RerankDataset, self).__init__()
        self.env = env
        self.pr_backbone = pr_backbone
        self.query_sets = load_pickle(query_filename)
        self.database_sets = load_pickle(database_filename)
        assert len(self.query_sets) == len(self.database_sets)
        self.reset_run_id(0, 0)
        self.cache_size = 1000
        self.k = k
    
    def reset_run_id(self, query_id, database_id):
        self.query_run_id = query_id
        self.database_run_id = database_id
        self.g_descs, self.init_top_k, self.new_top_k = [], dict(), dict()
        self.cache_pcs = dict()
        self.cache_pc_idxs = deque()
        self.cache_l_kpts_descs = dict()
        self.cache_l_idxs = deque()
        self.load_top_k()
    
    def load_top_k(self):
        if self.query_run_id == self.database_run_id or self.query_run_id >= len(self.query_sets) or self.database_run_id >= len(self.database_sets):
            return
        # load global desc
        for i in range(self.get_num_sample()):
            sample = self.get_sample(i)
            dir_name = os.path.dirname(sample['query'])
            file_name = os.path.basename(sample['query'])
            g_desc_file = os.path.join(dir_name, 'g_desc', self.pr_backbone, file_name)
            self.g_descs.append(np.load(g_desc_file))
        # find init top k
        n_query = self.get_num_query()
        d_tree = KDTree(self.g_descs[n_query:])
        for i in range(n_query):
            distances, indices = d_tree.query(np.array([self.g_descs[i]]), k=self.k)
            init_top_k = indices[0] + n_query
            sample = self.get_sample(i)
            positives = np.array(sample[self.database_run_id]) + n_query
            top_k_label = [x in positives for x in init_top_k]
            self.init_top_k[i] = tuple([init_top_k, top_k_label])
    
    def get_num_run(self):
        return len(self.query_sets)
    
    def get_num_query(self):
        return len(self.query_sets[self.query_run_id])
    
    def get_num_sample(self):
        return len(self.query_sets[self.query_run_id]) + len(self.database_sets[self.database_run_id])
    
    def get_sample(self, idx):
        n_query = self.get_num_query()
        assert idx >= 0 and idx < self.get_num_sample()
        if idx < n_query:
            return self.query_sets[self.query_run_id][idx]
        else:
            return self.database_sets[self.database_run_id][idx-n_query]
    
    def get_g_desc(self, idx):
        return self.g_descs[idx]
    
    def get_pc(self, idx):
        if idx in self.cache_pc_idxs:
            return self.cache_pcs[idx]
        sample = self.get_sample(idx)
        pc = load_pc(sample['query'])
        self.cache_pc_idxs.append(idx)
        self.cache_pcs[idx] = pc
        if len(self.cache_pc_idxs) > self.cache_size:
            pop_idx = self.cache_pc_idxs.popleft()
            assert pop_idx in self.cache_pcs, f'pop idx: {pop_idx} is not in pc cache'
            del self.cache_pcs[pop_idx]
        return pc
    
    def get_l_kpt_desc(self, idx, num_kpt=128):
        if idx in self.cache_l_idxs:
            return self.cache_l_kpts_descs[idx][0], self.cache_l_kpts_descs[idx][1]
        sample = self.get_sample(idx)
        dir_name = os.path.dirname(sample['query'])
        file_name = os.path.basename(sample['query'])
        l_kpt = np.load(os.path.join(dir_name, 'l_kpt', configs.rerank.pr_backbone, file_name))
        l_desc = np.load(os.path.join(dir_name, 'l_desc', configs.rerank.pr_backbone, file_name))
        if len(l_kpt) > num_kpt:
            indices = random.sample(range(len(l_kpt)), num_kpt)
            l_kpt = l_kpt[indices]
            l_desc = l_desc[indices]
        # scale
        l_kpt = l_kpt * sample['scale']  # + np.array([[sample['mx'], sample['my'], sample['mz']]])
        l_desc = normalize(l_desc)
        self.cache_l_idxs.append(idx)
        self.cache_l_kpts_descs[idx] = tuple([l_kpt, l_desc])
        if len(self.cache_l_idxs) > self.cache_size:
            pop_idx = self.cache_l_idxs.popleft()
            assert pop_idx in self.cache_l_kpts_descs, f'pop idx: {pop_idx} is not in local cache'
            del self.cache_l_kpts_descs[pop_idx]
        return l_kpt, l_desc
    
    def get_init_top_k(self, idx):
        assert idx in self.init_top_k
        return self.init_top_k[idx][0], self.init_top_k[idx][1]
    
    def get_new_top_k(self, idx):
        assert idx in self.new_top_k
        return self.new_top_k[idx][0], self.new_top_k[idx][1]


def analyse_rerank_result(pkl_file, k=5):
    t_dataset = load_pickle(pkl_file)
    save_dir = os.path.dirname(pkl_file)
    save_dir = os.path.join(save_dir, t_dataset.env, f'{t_dataset.query_run_id}_{t_dataset.database_run_id}')
    n_query = t_dataset.get_num_query()
    last_i = 0
    for i in range(n_query):
        init_top_k, init_gt_labels = t_dataset.get_init_top_k(i)
        new_top_k, new_gt_labels = t_dataset.get_new_top_k(i)
        imp = np.sum(new_gt_labels[:k]) - np.sum(init_gt_labels[:k])
        # save = not init_gt_labels[0] and new_gt_labels[0] and imp > 1  # success cases
        save = not init_gt_labels[0] and not new_gt_labels[0] and np.sum(new_gt_labels[:k]) == 0
        if save and i - last_i > 5:
            last_i = i
            ensure_dir(os.path.join(save_dir, str(i)))
            # query
            q_pc = t_dataset.get_pc(i)
            draw_pc(q_pc, os.path.join(save_dir, str(i), f'query_{i}.svg'))
            # init top k
            for j in range(k):
                d_pc = t_dataset.get_pc(init_top_k[j])
                state = 'good' if init_gt_labels[j] else 'bad'
                draw_pc(d_pc, os.path.join(save_dir, str(i), f'init_{j}_{state}.svg'))
                d_pc = t_dataset.get_pc(new_top_k[j])
                state = 'good' if new_gt_labels[j] else 'bad'
                draw_pc(d_pc, os.path.join(save_dir, str(i), f'new_{j}_{state}.svg'))


if __name__ == '__main__':
    pkl_files = ['/home/ericxhzou/Code/PCGL-Benchmark/exp/rerank/SGV/LoGG3DNet/Hankou_1_2_0_1_top25_rerank.pickle',
                 '/home/ericxhzou/Code/PCGL-Benchmark/exp/rerank/SGV/LoGG3DNet/Hankou_1_2_1_0_top25_rerank.pickle',
                 '/home/ericxhzou/Code/PCGL-Benchmark/exp/rerank/SGV/LoGG3DNet/Campus_1_2_0_1_top25_rerank.pickle',
                 '/home/ericxhzou/Code/PCGL-Benchmark/exp/rerank/SGV/LoGG3DNet/Campus_1_2_1_0_top25_rerank.pickle']
    for pkl_file in pkl_files:
        analyse_rerank_result(pkl_file)
