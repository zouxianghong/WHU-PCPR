# Author: Jacek Komorowski
# Warsaw University of Technology

import baseline.pr.MinkLoc3D.models.minkloc as minkloc


def model_factory(params):
    in_channels = 1

    if 'MinkFPN' in params.model:
        model = minkloc.MinkLoc(params.model, in_channels=in_channels,
                                feature_size=params.feature_size,
                                output_dim=params.output_dim, planes=params.planes,
                                layers=params.layers, num_top_down=params.num_top_down,
                                conv0_kernel_size=params.conv0_kernel_size)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model))

    return model
