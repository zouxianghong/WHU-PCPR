import argparse
import copy
import time
import os
import numpy as np
import random
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from torchpack.utils.config import configs
from misc.utils import ensure_dir, save_pickle
from datasets.rerank_dataset import RerankDataset

import baseline.rerank.AlphaQE.alpha_qe as AQE
import baseline.rerank.SpectralGV.sgv as SGV
import baseline.rerank.RankPointRetrieval.rpr as RPR


def gather_rerank_result(a_idxs, rerank_pn_idxs, rerank_gt_labels):
    top_k_dict = dict()
    for i in tqdm(range(len(a_idxs)), desc='Gather Rerank Result'):
        a_idx = a_idxs[i]
        top_k_dict[a_idx] = tuple([list(rerank_pn_idxs[i]), list(rerank_gt_labels[i])])
    return top_k_dict


def save_rerank_result(out_dir, t_dataset: RerankDataset):
    if len(t_dataset.new_top_k) == 0:
        return
    ensure_dir(out_dir)
    pkl_file = os.path.join(out_dir, f"{t_dataset.env}_{t_dataset.query_run_id}_{t_dataset.database_run_id}_top{t_dataset.k}_rerank.pickle")
    save_pickle(t_dataset, pkl_file)


def run_one_query(q_idx: int, t_dataset: RerankDataset, rerank_type):
    # idx
    init_top_k, top_k_label = t_dataset.get_init_top_k(q_idx)
    # anchor
    a_data = {}
    if rerank_type == 'SGV' or rerank_type == 'RANSAC':
        a_data['l_kpt'], a_data['l_desc'] = t_dataset.get_l_kpt_desc(q_idx)
    elif rerank_type == 'RPR':
        a_data['pc'] = t_dataset.get_pc(q_idx)
        a_data['g_desc'] = t_dataset.get_g_desc(q_idx)
        a_data['l_kpt'], a_data['l_desc'] = t_dataset.get_l_kpt_desc(q_idx)
    elif rerank_type == 'aQE' or rerank_type == 'avgQE':
        a_data['g_desc'] = t_dataset.get_g_desc(q_idx)
    # top k
    k_datas = []
    for k_idx in init_top_k:
        k_data = {}
        if rerank_type == 'SGV' or rerank_type == 'RANSAC':
            k_data['l_kpt'], k_data['l_desc'] = t_dataset.get_l_kpt_desc(k_idx)
        elif rerank_type == 'RPR':
            k_data['pc'] = t_dataset.get_pc(k_idx)
            k_data['g_desc'] = t_dataset.get_g_desc(k_idx)
            k_data['l_kpt'], k_data['l_desc'] = t_dataset.get_l_kpt_desc(k_idx)
        elif rerank_type == 'aQE' or rerank_type == 'avgQE':
            k_data['g_desc'] = t_dataset.get_g_desc(k_idx)
        k_datas.append(k_data)
    # registration
    if rerank_type == 'SGV':
        if configs.rerank.pr_backbone == 'logg3d_net':
            query = {'keypoints': torch.from_numpy(a_data['l_kpt']),  # N x 3
                     'features': torch.from_numpy(a_data['l_desc'])  # N x d
                    }
            reg_score = np.zeros(len(k_datas))
            for i in range(len(k_datas)):
                ref = {'keypoints': torch.from_numpy(k_datas[i]['l_kpt']),
                       'features': torch.from_numpy(k_datas[i]['l_desc'])
                      }
                reg_score[i] = SGV.sgv_fn_logg3dnet(query, ref, d_thresh=0.4, use_cpu=configs.rerank.use_cpu)
        else:
            query = {'keypoints': torch.from_numpy(a_data['l_kpt']),  # N x 3
                     'features': torch.from_numpy(a_data['l_desc'])  # N x d
                    }
            num_kpt = 100000
            for i in range(len(k_datas)):
                num_kpt = np.minimum(k_datas[i]['l_kpt'].shape[0], num_kpt)
            ref_kpts, ref_feats = [], []
            for i in range(len(k_datas)):
                indices = random.sample(range(len(k_datas[i]['l_kpt'])), num_kpt)
                ref_kpts.append(torch.from_numpy(k_datas[i]['l_kpt'][indices]))
                ref_feats.append(torch.from_numpy(k_datas[i]['l_desc'][indices]))
            ref_kpts = torch.stack(ref_kpts, dim=0)
            ref_feats = torch.stack(ref_feats, dim=0)
            reg_score = SGV.sgv_fn(query, ref_feats, ref_kpts, d_thresh=0.4, use_cpu=configs.rerank.use_cpu)
    elif rerank_type == 'RPR':
        reg_score = RPR.evaluate_matchs(a_data, k_datas, distance_threshold=configs.rerank.fgr_dist_thresh)
        for i in range(len(reg_score)):
            score1 = 1.0 / np.linalg.norm(a_data['g_desc'] - k_datas[i]['g_desc'])
            score2 = reg_score[i]
            reg_score[i] = score1 * configs.rerank.lamda1 + score2 * configs.rerank.lamda2
    elif rerank_type == 'RANSAC':
        reg_score = []
        a_kpts, a_feats = copy.deepcopy(a_data['l_kpt']), copy.deepcopy(a_data['l_desc'])
        for i in range(len(k_datas)):
            k_kpts, k_feats = copy.deepcopy(k_datas[i]['l_kpt']), copy.deepcopy(k_datas[i]['l_desc'])
            ransac_result = RPR.get_ransac_result(a_feats, k_feats, a_kpts, k_kpts, ransac_dist_th=configs.rerank.ransac_dist_thresh, vis=configs.rerank.vis)
            reg_score.append(ransac_result.fitness)
    elif rerank_type == 'aQE' or rerank_type == 'avgQE':
        new_a_g = [a_data['g_desc']]
        for i in range(len(k_datas)):
            k_data = k_datas[i]
            if rerank_type == 'avgQE':
                new_a_g.append(k_data['g_desc'])
            else:
                new_a_g.append(k_data['g_desc'] * np.linalg.norm(a_data['g_desc'] - k_data['g_desc']) ** configs.rerank.alpha)
        new_a_g = np.vstack(new_a_g)
        new_a_g = np.average(new_a_g, axis=0)
        AQE.make_new_query(a_data, k_datas, rerank_type)
        reg_score = []
        for i in range(len(k_datas)):
            k_data = k_datas[i]
            score = 1.0 - np.linalg.norm(new_a_g - k_data['g_desc'])
            reg_score.append(score)
    return q_idx, np.array(init_top_k), np.array(top_k_label), np.array(reg_score)


def run_one_data(t_dataset: RerankDataset):
    q_idxs, topk_idxs, topk_labels, reg_scores = [], [], [], []
    t1 = time.time()
    for q_idx in tqdm(range(t_dataset.get_num_query()), desc=f'Rerank {t_dataset.query_run_id} -> {t_dataset.database_run_id}'):
        if q_idx > 10 and configs.debug:
            continue
        q_idx, pn_idx, gt_label, reg_score = run_one_query(q_idx, t_dataset, rerank_type=configs.rerank.name)
        q_idxs.append(q_idx)
        topk_idxs.append(pn_idx)
        topk_labels.append(gt_label)
        reg_scores.append(reg_score)
    t2 = time.time()
    t3 = time.time()
    # rerank by reg_scores
    topk_idxs = np.stack(np.array(topk_idxs), axis=0)
    topk_labels = np.stack(np.array(topk_labels), axis=0)
    reg_scores = np.stack(np.array(reg_scores), axis=0)
    sorted_indices = np.argsort(-reg_scores, axis=1)
    rerank_topk_labels = np.take_along_axis(topk_labels, sorted_indices, axis=-1)
    rerank_topk_idxs = np.take_along_axis(topk_idxs, sorted_indices, axis=-1)
    t4 = time.time()
    mean_time = (t2 - t1 + t4 - t3) * 1000 / reg_scores.shape[0]
    # recall and precision
    rerank_topk_labels_cumsum = np.cumsum(rerank_topk_labels, axis=1)
    recall = rerank_topk_labels_cumsum > 0
    recall = np.sum(recall, axis=0) / float(rerank_topk_labels_cumsum.shape[0]) * 100
    precision = np.sum(rerank_topk_labels_cumsum, axis=0) / float(
        rerank_topk_labels_cumsum.shape[0]) * 100 / np.arange(1, rerank_topk_labels_cumsum.shape[1] + 1, 1)
    top_k_dict = gather_rerank_result(q_idxs, rerank_topk_idxs, rerank_topk_labels)
    return recall, precision, mean_time, top_k_dict


def run(env, database_file, query_file):
    # query / ref index
    t_dataset = RerankDataset(env, query_file, database_file, configs.rerank.pr_backbone, configs.rerank.k)
    recall_list, precision_list, mean_time = [], [], []
    num_run = t_dataset.get_num_run()
    for d_run_id in range(num_run):
        for q_run_id in range(num_run):
            if d_run_id == q_run_id:
                continue
            t_dataset.reset_run_id(q_run_id, d_run_id)
            recall, precision, mean_time_i, top_k_dict_i = run_one_data(t_dataset)
            t_dataset.new_top_k = top_k_dict_i
            recall_list.append(recall)
            precision_list.append(precision)
            mean_time.append(mean_time_i)
            # save rerank result
            save_rerank_result(configs.save_dir, t_dataset)
            # print(f'Evaluate {q_run_id} -> {d_run_id}')
            # with np.printoptions(precision=2, suppress=True):
            #     print("Recall @N: {}".format(recall))
            #     print("Precision @N: {}".format(precision))
    # recall / precision / run time
    recall_list = np.stack(recall_list, axis=0)
    precision_list = np.stack(precision_list, axis=0)
    avg_recall = np.mean(recall_list, axis=0)
    avg_precision = np.mean(precision_list, axis=0)
    mean_time = np.mean(mean_time)
    return avg_recall, avg_precision, mean_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True)
    parser.add_argument('--pr_backbone', type = str, required = False)
    parser.add_argument('--debug', type = int, required = False)
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.load(configs.protocol, recursive = True)
    if args.pr_backbone is not None:
        configs.rerank.pr_backbone = args.pr_backbone
    if args.debug is not None:
        configs.debug = args.debug
    configs.save_dir = os.path.join(configs.exp_dir, configs.rerank.name, configs.rerank.pr_backbone)
    
    for k in configs.eval.environments:
        submap_type = configs.eval.environments[k]['submap_type']
        database_files, query_files = [], []
        for d, q in zip(configs.eval.environments[k]['database_files'], configs.eval.environments[k]['query_files']):
            tmp = d.split('.')
            database_files.append(f'{tmp[0]}_{submap_type}.{tmp[1]}')
            tmp = q.split('.')
            query_files.append(f'{tmp[0]}_{submap_type}.{tmp[1]}')
        configs.eval.environments[k]['database_files'] = database_files
        configs.eval.environments[k]['query_files'] = query_files
    print(configs)
    
    # Rank-PointRetrieval initialization
    if configs.rerank.name == 'RPR':
        RPR.init(
            reg_type=configs.rerank.reg_type,
            keypoints=configs.rerank.num_keypt,
            with_noise=configs.rerank.with_noise,
            sigma=configs.rerank.sigma
        )

    # run one by one
    res_df_cols = ['env'] + [f'R@{x}' for x in range(1, 26)] + [f'P@{x}' for x in range(1, 26)] + ['avg time (ms)']
    res_df = pd.DataFrame(columns=res_df_cols)
    for env in configs.eval.environments:
        if not res_df.empty and configs.debug:
            continue
        print(f'-------------------- Rerank: {env} --------------------')
        database_files = configs.eval.environments[env]['database_files']
        query_files = configs.eval.environments[env]['query_files']
        avg_recall, avg_precision, avg_time = [], [], []
        for d, q in zip(database_files, query_files):
            if not os.path.isfile(d):
                continue
            if q != None:
                recall_x, precision_x, time_x = run(env, d, q)
                avg_recall.append(recall_x)
                avg_precision.append(precision_x)
                avg_time.append(time_x)
        avg_recall = np.stack(avg_recall, axis=0)
        avg_precision = np.stack(avg_precision, axis=0)
        avg_recall = np.mean(avg_recall, axis=0)
        avg_precision = np.mean(avg_precision, axis=0)
        avg_time = np.mean(avg_time)
        
        new_row = [env] + list(avg_recall) + list(avg_precision) + [avg_time]
        new_row = pd.DataFrame([new_row], columns=res_df.columns)
        res_df = pd.concat([res_df, new_row], ignore_index=True)
        with pd.option_context('display.precision', 2):
            print(new_row)
    save_path = os.path.join(configs.save_dir, f'{configs.rerank.name}_{configs.rerank.pr_backbone}_rerank.csv')
    res_df.to_csv(save_path, index=False, float_format='%.2f')
