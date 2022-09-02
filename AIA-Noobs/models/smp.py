import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.decoders.unetplusplus import UnetPlusPlus
import torch.nn as nn
import torch
from . import smp_customized as smp_cus

def freeze_encoder(model,layers=-3):
    for i,_ in enumerate(model.encoder.children()):
        pass
    L=i+1
    
    if layers>=0:
        endpoint=layers
    else:
        endpoint=L+layers
    
    cur_layer=0
    for child in model.encoder.children():
        if cur_layer<=endpoint:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        cur_layer+=1

def unfreeze_encoder(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

#ref: https://github.com/qubvel/segmentation_models.pytorch#architectures
class SMPModel(nn.Module):
    def __init__(self, config, num_cls, aux=False):
        super().__init__()
        encoder_name = config['encoder_name']
        seg_name = config['seg_name']
        encoder_w = 'imagenet' if config['seg_encoder_weights'] else None
        if aux and seg_name != 'UnetPlusPlus':
            raise ValueError('Not implemented decoder')
        if seg_name == 'DeepLabV3Plus':
            model = smp.DeepLabV3Plus(encoder_name=encoder_name,
                                      encoder_depth=5,
                                      encoder_weights=encoder_w,
                                      encoder_output_stride=16,
                                      decoder_channels=256,
                                      decoder_atrous_rates=(12, 24, 36),
                                      in_channels=3,
                                      classes=num_cls, activation=None, upsampling=4, aux_params=None)
        elif seg_name == 'UnetPlusPlus':
            decoder_channels = (256, 128, 64, 32, 16)
            encoder_depth = 5
            decoder_use_batchnorm = True
            decoder_attention_type = None
            if not aux:
                model = smp.UnetPlusPlus(encoder_name=encoder_name,
                                         encoder_depth=encoder_depth,
                                         encoder_weights=encoder_w,
                                         decoder_use_batchnorm=decoder_use_batchnorm,
                                         decoder_channels=decoder_channels,
                                         decoder_attention_type=decoder_attention_type,
                                         in_channels=3,
                                         classes=num_cls, activation=None, aux_params=None)
            else:
                model = UnetPluxPluxAux(encoder_name=encoder_name,
                                       encoder_depth=encoder_depth,
                                       encoder_weights=encoder_w,
                                       decoder_use_batchnorm=decoder_use_batchnorm,
                                       decoder_channels=decoder_channels,
                                       decoder_attention_type=decoder_attention_type,
                                       in_channels=3,
                                       classes=num_cls, activation=None, aux_params=None)
                decoder = UnetPlusPlusDecoderAux(
                    encoder_channels=model.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=encoder_depth,
                    use_batchnorm=decoder_use_batchnorm,
                    center=True if encoder_name.startswith("vgg") else False,
                    attention_type=decoder_attention_type,
                )
                model.decoder = decoder
        elif seg_name== 'UnetPlusPlusCGM':
            model = smp_cus.UnetPlusPlusCGM(encoder_name=encoder_name,
                                     encoder_depth=5,
                                     encoder_weights='imagenet',
                                     decoder_use_batchnorm=True,
                                     decoder_channels=(256, 128, 64, 32, 16),
                                     decoder_attention_type=None,
                                     in_channels=3,
                                     classes=num_cls, activation=None, aux_params=None)
        elif seg_name== 'UnetPlusPlusCosSim':
            model = smp_cus.UnetPlusPlusCosSim(encoder_name=encoder_name,
                                     encoder_depth=5,
                                     encoder_weights='imagenet',
                                     decoder_use_batchnorm=True,
                                     decoder_channels=(256, 128, 64, 32, 16),
                                     decoder_attention_type=None,
                                     in_channels=3,
                                     classes=num_cls, activation=None, aux_params=None)
        elif seg_name == 'PAN':
            model = smp.PAN(encoder_name=encoder_name,
                            encoder_weights=encoder_w,
                            in_channels=3,
                            classes=num_cls)
        elif seg_name == 'MAnet':
            model = smp.MAnet(encoder_name=encoder_name,
                              encoder_depth=5,
                              encoder_weights=encoder_w,
                              in_channels=3,
                              classes=num_cls)
        elif seg_name == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=encoder_name,
                                  encoder_depth=5,
                                  encoder_weights=encoder_w,
                                  in_channels=3,
                                  classes=num_cls)
        elif seg_name == 'PSPNet':
            model = smp.PSPNet(encoder_name=encoder_name,
                               encoder_weights=encoder_w,
                               in_channels=3,
                               classes=num_cls)
        else:
            AssertionError('not implemented segmentation architecture')
            model = None
        if config['freeze_encoder']:
            freeze_encoder(model)
        self._model = model

    def __call__(self, x):
        # TOFIX: bs should not be 1 !!!
        # https://github.com/qubvel/segmentation_models.pytorch/issues/342
        return self._model(x)

class UnetPlusPlusDecoderAux(UnetPlusPlusDecoder):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks,
        use_batchnorm,
        attention_type,
        center,
    ):
        super().__init__(encoder_channels,
                         decoder_channels,
                         n_blocks,
                         use_batchnorm,
                         attention_type,
                         center)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                    print(f"x_{depth_idx}_{depth_idx}")
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
                    print(f"x_{depth_idx}_{dense_l_i}")
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"]) # 0-4
        print(dense_x.keys())
        for k in dense_x.keys():
            print(k, dense_x[k].shape)
        # return dense_x[f"x_{0}_{self.depth}"]
        return [dense_x[f"x_{0}_{self.depth}"],  # 0-4
                dense_x[f"x_{1}_{self.depth-1}"],  # 1-3
                dense_x[f"x_{2}_{self.depth-2}"],  # 2-2
                ]


class UnetPluxPluxAux(UnetPlusPlus):
    def __init__(self,
                 encoder_name,
                 encoder_depth,
                 encoder_weights,
                 decoder_use_batchnorm,
                 decoder_channels,
                 decoder_attention_type,
                 in_channels,
                 classes,
                 activation,
                 aux_params):
        super().__init__(
                 encoder_name,
                 encoder_depth,
                 encoder_weights,
                 decoder_use_batchnorm,
                 decoder_channels,
                 decoder_attention_type,
                 in_channels,
                 classes,
                 activation,
                 aux_params)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        return decoder_output
        masks = [self.segmentation_head(o) for o in decoder_output]

        if self.classification_head is not None:
            raise ValueError('Not implemented')
            # labels = self.classification_head(features[-1])
            # return masks, labels

        return masks