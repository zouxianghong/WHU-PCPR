import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch 
import argparse 
import random 
import numpy as np
from datetime import datetime

from torchpack.utils.config import configs
from misc.utils import ensure_dir
from misc.quantization import make_quantizer

from eval.evaluate_pr import evaluate
from trainer_pr import Trainer

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # Repeatability
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Get args and configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True, help = 'Path to configuration YAML file')
    parser.add_argument('--train_aug', type = int, required = False)
    parser.add_argument('--debug', type = int, required = False)
    
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.load(configs.protocol, recursive = True)
    
    if args.train_aug is not None:
        configs.train.aug_mode = args.train_aug
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
    
    if 'scheduler_milestones' in configs.train.optimizer:
        if isinstance(configs.train.optimizer.scheduler_milestones, str):
            configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    model_dir = os.path.join(configs.save_dir, configs.model.name, 'models')
    eval_dir = os.path.join(configs.save_dir, configs.model.name, 'eval')
    ensure_dir(model_dir)
    ensure_dir(eval_dir)
    logger = SummaryWriter(os.path.join(configs.save_dir, configs.model.name, 'tf_logs'))
    
    # Train model
    trainer = Trainer(logger, configs.train.environment)
    trained_model = trainer.train()
    
    # Save model
    name = f'{configs.model.name}_train-aug{configs.train.aug_mode}_on_{configs.train.env_name}'
    torch.save(trained_model.state_dict(), os.path.join(model_dir, f'{name}.pth'))
    
    # Evaluate
    eval_res = evaluate(trained_model, save_path=os.path.join(eval_dir, f'{name}.csv'))
