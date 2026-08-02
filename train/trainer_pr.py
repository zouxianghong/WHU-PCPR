import torch
from tqdm import tqdm

from torchpack.utils.config import configs

from misc.utils import AverageMeter, get_matching_indices
from misc.model_factory import model_factory
from misc.loss_factory import make_g_loss, make_l_loss
from datasets.dataset_utils import get_train_dataloader


class Trainer:
    def __init__(self, logger, train_environment):
        # Initialise inputs
        self.logger = logger
        self.epochs = configs.train.optimizer.epochs

        # Set up meters and stat trackers
        self.loss_pr_meter = AverageMeter()
        self.num_triplets_meter = AverageMeter()
        self.non_zero_triplets_meter = AverageMeter()
        self.embedding_norm_meter = AverageMeter()

        # Make dataloader
        self.dataloader = get_train_dataloader(pickle_file=train_environment, aug_mode=configs.train.aug_mode)

        # Build model
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model = model_factory(ckpt = None, device = 'cuda')

        # Make loss functions
        self.g_loss_fn = make_g_loss()
        self.l_loss_fn = make_l_loss() if 'l_loss' in configs.train else None
   
    def make_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

    def make_scheduler(self):
        if 'scheduler' not in configs.train.optimizer:
            self.scheduler = None
        else:
            if configs.train.optimizer.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs.train.optimizer.epochs+1,
                                                                            eta_min=configs.train.optimizer.min_lr)
            elif configs.train.optimizer.scheduler == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=configs.train.optimizer.scheduler_step_size, gamma=configs.train.optimizer.scheduler_gamma)
            elif configs.train.optimizer.scheduler == 'MultiStepLR':
                if not isinstance(configs.train.optimizer.scheduler_milestones, list):
                    configs.train.optimizer.scheduler_milestones = [configs.train.optimizer.scheduler_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, configs.train.optimizer.scheduler_milestones, gamma=0.1)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(configs.train.optimizer.scheduler))
    
    def before_epoch(self):
        # Reset meters
        self.loss_pr_meter.reset()
        self.num_triplets_meter.reset()
        self.non_zero_triplets_meter.reset()
        self.embedding_norm_meter.reset()

    def training_step(self, batch, positives_mask, negatives_mask):
        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        if n_positives == 0 or n_negatives == 0:
            # Skip a batch without positives or negatives
            print('WARNING: Skipping batch without positive or negative examples')
            return None

        # Get embeddings and Loss (embeddings stop gradients)
        self.optimizer.zero_grad()
        if isinstance(batch, dict):
            batch = {x: batch[x].to('cuda') if x != 'pos_pairs' else batch[x] for x in batch}
        else:
            batch = batch.to('cuda')
        out = self.model(batch)  # B x 256, B x *

        # local loss
        g_descs, loss_local = out, 0.0
        if configs.model.name == 'EgoNN' or configs.model.name == 'PPTNet':
            g_descs, B = out['global'], out['global'].shape[0]
            
            if self.l_loss_fn is not None:
                # transfrom to global coordinate
                for x in range(B):
                    batch['cloud'][x] = batch['cloud'][x] * batch['scale'][x] + batch['mean'][x]
                    out['keypoints'][x] = out['keypoints'][x] * batch['scale'][x] + batch['mean'][x]

                q_cloud = [batch['cloud'][x] for x in range(0, B, 2)]
                q_kpt = [out['keypoints'][x] for x in range(0, B, 2)]
                q_sigma = [out['sigma'][x] for x in range(0, B, 2)]
                q_desc = [out['descriptors'][x] for x in range(0, B, 2)]
                
                p_cloud = [batch['cloud'][x] for x in range(1, B, 2)]
                p_kpt = [out['keypoints'][x] for x in range(1, B, 2)]
                p_sigma = [out['sigma'][x] for x in range(1, B, 2)]
                p_desc = [out['descriptors'][x] for x in range(1, B, 2)]
                
                # local loss refer to 'baseline/pr/EgoNN/training/trainer.py' line 186
                loss_local, _ = self.l_loss_fn(q_cloud, q_kpt, q_sigma, q_desc, p_cloud, p_kpt, p_sigma, p_desc)
        elif configs.model.name == 'LoGG3DNet':
            g_descs, B = out[0], out[0].shape[0]
            q_desc = [out[1][x] for x in range(0, B, 2)]
            p_desc = [out[1][x] for x in range(1, B, 2)]
            
            # find positive pairs and compute local loss, refer to 'baseline/pr/LoGG3DNet/training/train.py' line 102
            n_l_loss = 0
            for x in range(len(q_desc)):
                pos_pairs = batch['pos_pairs'][x]
                if len(pos_pairs) > 0:
                    loss_local += self.l_loss_fn(q_desc[x], p_desc[x], pos_pairs)
                    n_l_loss += 1
            loss_local = loss_local / n_l_loss if n_l_loss > 0 else 0.0
        # place recognition loss
        loss_place_rec, num_triplets, non_zero_triplets, embedding_norm = self.g_loss_fn(g_descs, positives_mask, negatives_mask)
        loss_total = loss_place_rec + loss_local
        
        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_pr_meter.update(loss_place_rec.item())
        self.num_triplets_meter.update(num_triplets)
        self.non_zero_triplets_meter.update(non_zero_triplets)
        self.embedding_norm_meter.update(embedding_norm)

        return None

    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Dynamic Batch Expansion
        if configs.train.batch_expansion_th is not None:
            ratio_non_zeros = self.non_zero_triplets_meter.avg / self.num_triplets_meter.avg
            if ratio_non_zeros < configs.train.batch_expansion_th:
                self.dataloader.batch_sampler.expand_batch()
        
        # Tensorboard plotting
        self.logger.add_scalar(f'Place_Rec_Loss', self.loss_pr_meter.avg, epoch)
        self.logger.add_scalar(f'Non_Zero_Triplets', self.non_zero_triplets_meter.avg, epoch)
        self.logger.add_scalar(f'Embedding_Norm', self.embedding_norm_meter.avg, epoch)

    def train(self):
        # Make optimizer
        self.make_optimizer()
        # Scheduler
        self.make_scheduler()
        # Prepare before training
        self.model.prepare_before_train()
        # Training
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch()
            for idx, (batch, positives_mask, negatives_mask) in enumerate(self.dataloader):
                self.training_step(batch, positives_mask, negatives_mask)
                if configs.debug and idx > 2:
                    break
            self.after_epoch(epoch)
        
        return self.model
