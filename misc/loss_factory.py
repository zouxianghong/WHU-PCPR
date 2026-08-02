from torchpack.utils.config import configs 
from misc.global_loss import *
from misc.local_loss import *


def make_g_loss():  # global loss for place recognition
    g_loss_name = configs.train.g_loss.name
    if g_loss_name == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(configs.train.g_loss.margin, configs.model.normalize_embeddings)
    elif g_loss_name == 'BatchHardContrastiveLoss':
        loss_fn = BatchHardContrastiveLossWithMasks(configs.train.g_loss.pos_margin,
                                                    configs.train.g_loss.neg_margin,
                                                    configs.model.normalize_embeddings)
    elif g_loss_name == 'QuadrupletLossWithMasks':
        loss_fn = QuadrupletLossWithMasks()
    else:
        print('Unknown loss: {}'.format(g_loss_name))
        raise NotImplementedError
    return loss_fn                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            


def make_l_loss():  # local loss for rerank / pose estimation
    l_loss_name = configs.train.l_loss.name
    if l_loss_name == 'KeypointCorrLoss':  # refer to EgoNN
        loss_fn = KeypointCorrLoss(gamma_c=configs.train.l_loss.gamma_c,
                                   gamma_chamfer=configs.train.l_loss.gamma_chamfer,
                                   gamma_p2p=configs.train.l_loss.gamma_p2p, beta=configs.train.l_loss.beta)
    elif l_loss_name == 'PointContrastiveLoss':  # refer to LoGG3DNet
        loss_fn = PointContrastiveLoss(point_pos_margin=configs.train.l_loss.point_pos_margin,
                                       point_neg_margin=configs.train.l_loss.point_neg_margin,
                                       point_neg_weight=configs.train.l_loss.point_neg_weight,
                                       num_pos=configs.train.l_loss.num_pos,
                                       num_hn_samples=configs.train.l_loss.num_hn_samples)
    else:
        print('Unknown loss: {}'.format(l_loss_name))
        raise NotImplementedError
    return loss_fn
