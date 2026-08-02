import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchpack.utils.config import configs 
import numpy as np
import pandas as pd
from datetime import datetime

import argparse 
import torch
from misc.model_factory import model_factory
from misc.quantization import make_quantizer
from misc.utils import ensure_dir
from eval.eval_multisession import eval_multisession
from eval.eval_singlesession import eval_singlesession


def evaluate(model, save_path=None):
    # Wrapper of other eval functions for incremental learning
    res_df_cols = ['env'] + [f'R@{x}' for x in range(1, 26)] + ['R@1%'] + [f'P@{x}' for x in range(1, 26)] + ['P@1%'] + ['time (ms)']
    res_df = pd.DataFrame(columns=res_df_cols)
    for env in configs.eval.environments.keys():
        if not res_df.empty and configs.debug:
            continue
        print(f'> Evaluate on {env}')
        database_files = configs.eval.environments[env]['database_files']
        query_files = configs.eval.environments[env]['query_files']
        env_recall_N, env_precision_N, env_recall_one_percent, env_time = [], [], [], []
        for d, q in zip(database_files, query_files):
            if not os.path.isfile(d):
                continue
            if q != None:
                env_res = eval_multisession(model, d, q)
            else:
                world_thresh = configs.eval.world_thresh[env]
                false_pos_thresh = configs.eval.false_pos_thresh[env]
                time_thresh = configs.eval.time_thresh[env]
                # FIXME: evaluation for single session
                env_res = eval_singlesession(model, d, world_thresh, false_pos_thresh, time_thresh)
            env_recall_N.append(env_res['Recall@N'])
            env_precision_N.append(env_res['Precision@N'])
            env_recall_one_percent.append(env_res['Recall@1%'])
            env_time.append(env_res['time'])
        if len(env_recall_N) == 0:
            continue
        env_recall_N = np.mean(env_recall_N, axis=0)
        env_precision_N = np.mean(env_precision_N, axis=0)
        env_recall_one_percent = np.mean(env_recall_one_percent)
        env_time = np.mean(env_time)
        new_row = [env] + list(env_recall_N) + [env_recall_one_percent] + list(env_precision_N) + [None] + [env_time]
        new_row = pd.DataFrame([new_row], columns=res_df.columns)
        res_df = pd.concat([res_df, new_row], ignore_index=True)
        with pd.option_context('display.precision', 2):
            print(new_row)
    if save_path is not None:
        res_df.to_csv(save_path, index=False, float_format='%.2f')
        print(f'Save evaluation result: {save_path}')
    return res_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True)
    parser.add_argument('--train_aug', type = int, required = False)
    parser.add_argument('--eval_aug', type = int, required = False)
    parser.add_argument('--save_desc', type = int, required = False)
    parser.add_argument('--rot_theta', type = float, required = False)
    parser.add_argument('--debug', type = int, required = False)

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.load(configs.protocol, recursive = True)

    if args.train_aug is not None:
        configs.train.aug_mode = args.train_aug
    if args.eval_aug is not None:
        configs.eval.aug_mode = args.eval_aug
    if args.save_desc is not None:
        configs.eval.save_desc = args.save_desc
    if args.rot_theta is not None:
        configs.eval.rotate_theta = args.rot_theta
    if args.debug is not None:
        configs.debug = args.debug
    
    configs.model.quantizer = None
    if 'quantization_type' in configs.model:
        quantization_type = configs.model.quantization_type
        quantization_size = configs.model.quantization_size
        configs.model.quantizer = make_quantizer(quantization_type, quantization_size)
    
    tmp = configs.train.environment.split('.')
    configs.train.environment = f'{tmp[0]}_{configs.train.submap_type}.{tmp[1]}'
    
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
    
    if configs.model.name == 'ScanContext':
        name = f'{configs.model.name}'
        model = None
    else:
        ckpt = os.path.join(configs.save_dir, configs.model.name, 'models',
                            f'{configs.model.name}_train-aug{configs.train.aug_mode}_on_{configs.train.env_name}.pth')
        model = model_factory(ckpt = torch.load(ckpt))
        name = f'{configs.model.name}_train-aug{configs.train.aug_mode}_eval-aug{configs.eval.aug_mode}_rotate{configs.eval.rotate_theta}_on_{configs.train.env_name}'
    save_dir = os.path.join(configs.save_dir, configs.model.name, 'eval')
    ensure_dir(save_dir)
    res = evaluate(model, save_path=os.path.join(save_dir, f'{name}.csv'))
    print(res)
