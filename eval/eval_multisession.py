import pickle 
import numpy as np 
from tqdm import tqdm 
from eval.eval_utils import get_latent_vectors, scan_context_kneighbors
from sklearn.neighbors import KDTree

import time
import os
import argparse
import torch
from misc.model_factory import model_factory
from misc.utils import ensure_dir
from torchpack.utils.config import configs
import matplotlib
from matplotlib import pyplot as plt

def eval_multisession(model, database_sets, query_sets):
    recall = np.zeros(25)
    precision = np.zeros(25)
    count = 0
    one_percent_recall = []
    n_query = 0

    database_g_descs = []
    database_sets = pickle.load(open(database_sets, 'rb'))
    for run in tqdm(database_sets, disable=False, desc = 'Getting database embeddings'):
        g_descs, l_kpts, l_descs = get_latent_vectors(model, run)
        database_g_descs.append(g_descs)
        # save desc
        if configs.eval.save_desc:
            assert len(run.keys()) == g_descs.shape[0]
            if len(l_kpts) > 0:
                assert g_descs.shape[0] == len(l_kpts) and len(l_kpts) == len(l_descs)
            for k in range(len(run.keys())):
                dir_name = os.path.dirname(run[k]['query'])
                file_name = os.path.basename(run[k]['query'])
                ensure_dir(os.path.join(dir_name, 'g_desc', configs.model.name))
                np.save(os.path.join(dir_name, 'g_desc', configs.model.name, file_name), g_descs[k])
                if len(l_kpts) > 0:
                    ensure_dir(os.path.join(dir_name, 'l_kpt', configs.model.name))
                    ensure_dir(os.path.join(dir_name, 'l_desc', configs.model.name))
                    np.save(os.path.join(dir_name, 'l_kpt', configs.model.name, file_name), l_kpts[k])
                    np.save(os.path.join(dir_name, 'l_desc', configs.model.name, file_name), l_descs[k])
    
    query_g_descs = []
    query_sets = pickle.load(open(query_sets, 'rb'))
    t0 = time.time()
    for run in tqdm(query_sets, disable=False, desc = 'Getting query embeddings'):
        g_descs, _, _ = get_latent_vectors(model, run, aug_mode=configs.eval.aug_mode, parallel=False)
        query_g_descs.append(g_descs)

    for i in tqdm(range(len(query_sets)), desc = 'Getting Recall'):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pair_recall, pair_precision, pair_similarity, pair_opr = get_recall(
                i, j, database_g_descs, query_g_descs, query_sets, database_sets
            )
            recall += np.array(pair_recall)
            precision += np.array(pair_precision)
            count += 1 
            one_percent_recall.append(pair_opr)
            n_query += len(query_g_descs[j])
    t1 = time.time()
    
    avg_recall = recall / count
    avg_precision = precision / count
    avg_one_percent_recall = np.mean(one_percent_recall)
    avg_time = (t1 - t0) * 1000 / n_query  # ms
    stats = {'Recall@1%': avg_one_percent_recall, 'Recall@N': avg_recall, 'Precision@N': avg_precision, 'time': avg_time}
    return stats


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    if configs.model.name != 'ScanContext':
        database_nbrs = KDTree(database_output)

    num_neighbors = 25
    labels = []

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0

    for i in range(len(queries_output)): #size
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        if configs.model.name == 'ScanContext':
            indices = scan_context_kneighbors(queries_output[i], database_output, k=num_neighbors)
        else:
            _, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
            indices = indices[0]

        labels_i = [1 if indices[j] in true_neighbors else 0 for j in range(len(indices))]
        labels.append(np.array(labels_i))
        for j in range(len(indices)):
            if indices[j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[j]])
                    top1_similarity_score.append(similarity)
                break
            else:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[j]])

        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    labels = np.stack(np.array(labels), axis=0)
    labels_cumsum = np.cumsum(labels, axis=1)
    
    recall = labels_cumsum > 0
    recall = np.sum(recall, axis=0) / float(labels_cumsum.shape[0]) * 100
    precision = np.sum(labels_cumsum, axis=0) / float(labels_cumsum.shape[0]) * 100 / np.arange(1, labels_cumsum.shape[1] + 1, 1)
    
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    
    return recall, precision, top1_similarity_score, one_percent_recall

# TODO Write code to evaluate an individual run

def get_res_on_oxford(model, cmp_dir, save_file):
    res = {}
    
    database_files = configs.eval.environments['Oxford']['database_files']
    query_files = configs.eval.environments['Oxford']['query_files']
    database_sets = pickle.load(open(database_files[0], 'rb'))
    query_sets = pickle.load(open(query_files[0], 'rb'))
    
    database_embeddings, query_embeddings = [], []
    for run in tqdm(database_sets, disable=False, desc = 'Getting database embeddings'):
        database_embeddings.append(get_latent_vectors(model, run))

    for run in tqdm(query_sets, disable=False, desc = 'Getting query embeddings'):
        query_embeddings.append(get_latent_vectors(model, run))

    for i in tqdm(range(len(query_sets)), desc = 'Getting Recall'):
        for j in range(len(query_sets)):
            if i == j:
                continue
            database_output = database_embeddings[i]
            queries_output = query_embeddings[j]
            
            database_nbrs = KDTree(database_output)
            
            for k in range(len(queries_output)): #size 
                # i is query element ndx
                query_details = query_sets[j][k]    # {'query': path, 'northing': , 'easting': }
                true_neighbors = query_details[i]
                if len(true_neighbors) == 0:
                    continue
                _, indices = database_nbrs.query(np.array([queries_output[k]]), k=1)
                top1_idx = indices[0][0]  # top 1
                top1_state = 1 if top1_idx in true_neighbors else 0
                top1_file = database_sets[i][top1_idx]['query']
                res_jik = {'query': query_sets[j][k]['query'],
                           'pos': [query_sets[j][k]['easting'], query_sets[j][k]['northing']],
                           'top1_state': top1_state,
                           'top1_file': top1_file}
                res['{}_{}_{}'.format(j,i,k)] = res_jik
    # save file
    if not os.path.exists(cmp_dir):
        os.makedirs(cmp_dir)
    save_path = os.path.join(cmp_dir, save_file)
    with open(save_path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return res


def compare_res_on_oxford(res_oxford, res_FT_seq1, res_Our_seq1, res_FT_seq2, res_Our_seq2, cmp_dir):
    res_oxford = pickle.load(open(res_oxford, 'rb'))
    res_FT_seq1 = pickle.load(open(res_FT_seq1, 'rb'))
    res_Our_seq1 = pickle.load(open(res_Our_seq1, 'rb'))
    res_FT_seq2 = pickle.load(open(res_FT_seq2, 'rb'))
    res_Our_seq2 = pickle.load(open(res_Our_seq2, 'rb'))

    for key in res_oxford:
        if res_oxford[key]['top1_state'] == 0:
            continue
        if res_Our_seq1[key]['top1_state'] == 0 and res_Our_seq2[key]['top1_state'] == 0:
            # make dir
            key_dir = os.path.join(cmp_dir, key)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir)
            # draw point cloud in matplot
            def draw_pc(pc_file, save_filepath=None, title_info='', pt_size=3, show_fig=False):
                if not show_fig:
                    matplotlib.use('Agg')
                pc_file = os.path.join(configs.data.dataset_folder, pc_file)
                pc = np.fromfile(pc_file, dtype = np.float64)
                pc = np.reshape(pc, (pc.shape[0] // 3, 3))
                x = pc[:, 0]
                y = pc[:, 1]
                z = pc[:, 2]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, s=pt_size, c=z,  # height data for color
                        cmap='rainbow')
                ax.set_title(title_info, fontsize=30)
                ax.axis()
                # set init view
                ax.view_init(elev=65.0, azim=-45.0)
                if save_filepath:
                    fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close('all')
            # draw
            draw_pc(res_oxford[key]['query'], os.path.join(key_dir, 'query.svg'))
            draw_pc(res_oxford[key]['top1_file'], os.path.join(key_dir, 'top1_oxford.svg'))
            # draw_pc(res_FT_seq1[key]['top1_file'], os.path.join(key_dir, 'top1_FT_seq1.svg'))
            draw_pc(res_Our_seq1[key]['top1_file'], os.path.join(key_dir, 'top1_Our_seq1.svg'))
            # draw_pc(res_FT_seq2[key]['top1_file'], os.path.join(key_dir, 'top1_FT_seq2.svg'))
            draw_pc(res_Our_seq2[key]['top1_file'], os.path.join(key_dir, 'top1_Our_seq2.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True)
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    print(configs)

    # Compare eval results
    cmp_dir = '/home/ericxhzou/Code/PCGL-Benchmark/exp/Ours_submodular/Compare'
    model = model_factory(ckpt = torch.load('exp/xxx.pth'))
    get_res_on_oxford(model, cmp_dir, 'res_train_on_oxford_FT.pickle')
    