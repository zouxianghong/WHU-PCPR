# Author: Jacek Komorowski
# Warsaw University of Technology
import torch
import MinkowskiEngine as ME

from torchpack.utils.config import configs

import baseline.pr.PointNetVlad.models.PointNetVlad as PointNetVLAD
import baseline.pr.PPTNet.models.pptnet as PPTNet
import baseline.pr.PatchAugNet.place_recognition.patch_aug_net.models.patch_aug_net as PatchAugNet
import baseline.pr.MinkLoc3D.models.model_factory as MinkLoc3D
import baseline.pr.EgoNN.models.model_factory as EgoNN
import baseline.pr.LoGG3DNet.models.pipeline_factory as LoGG3DNet
import baseline.pr.NDTTransformer.models.NDTNetVlad as NDTTransformer

from misc.utils import freeze_model


class PR_Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # backbone
        if configs.model.name == 'PointNetVLAD':
            self.backbone = PointNetVLAD.PointNetVlad(
                num_points = configs.data.num_points,
                global_feat = True,
                feature_transform = True,
                max_pool = False,
                output_dim = configs.model.output_dim)
        elif configs.model.name == 'PPTNet':
            self.backbone = PPTNet.Network(configs.model)
        elif configs.model.name == 'PatchAugNet':
            self.backbone = PatchAugNet.Network(configs.model, use_l2_norm=True)
        elif configs.model.name == 'MinkLoc3D':
            self.backbone = MinkLoc3D.model_factory(configs.model)
        elif configs.model.name == 'EgoNN':
            self.backbone = EgoNN.create_egonn_model(configs.model)
        elif configs.model.name == 'LoGG3DNet':
            self.backbone = LoGG3DNet.get_pipeline(configs.model.name)
        elif configs.model.name == 'NDTTransformer':
            self.backbone = NDTTransformer.NDTNetVlad(num_points=configs.model.num_points,
                                                      output_dim=configs.model.output_dim,
                                                      emb_dims=configs.model.emb_dims,
                                                      layer_number=configs.model.layer_number)
        else:
            raise NotImplementedError('Model not implemented: {}'.format(configs.model.name))
    
    def prepare_before_train(self):
        return
    
    def forward(self, batch):
        x = self.backbone(batch)  # B x 256
        return x


def model_factory(ckpt = None, device = 'cuda'):
    model = PR_Model()
    if ckpt != None:
        model.load_state_dict(ckpt)
    model = model.to(device)
    return model


def copy_frozen_model(model):
    model_frozen = model_factory(model.state_dict())
    freeze_model(model_frozen)
    return model_frozen
