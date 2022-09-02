from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from . import head
import torch.nn.functional as F


def upsample_2x(x):
    return F.interpolate(x,scale_factor=2,mode="bilinear")

class UnetPlusPlusCGMDecoder(UnetPlusPlusDecoder):
    # output depth      ^         ^            ^
    #                  |          |            |
    #         f0 -> out(0,0) -> out(0,1) -> *out(0,2) ->
    #         ^     ^^          ^^^         ^^^^
    #         f1 -> latent(1,1)-> out(1,2) 
    #         ^     ^^            ^^^
    #         f2 -> latent(2,2) 
    #         ^     ^^
    # input-> f3
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
                else:
                    dense_l_i = depth_idx + layer_idx                    
                    cat_features = [dense_x[f"x_{didx}_{dense_l_i}"] for didx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
                    
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        # CGM output
        output=[dense_x[f"x_{0}_{self.depth}"]]
        for d in range(self.depth-1,-1,-1):
            output.append(upsample_2x(output[-1])*dense_x[f"x_{0}_{d}"])
        return output


class UnetPlusPlusCGM(SegmentationModel):
    """
    Reference:
        https://arxiv.org/abs/1807.10165
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder_depth=encoder_depth
        
        self.decoder = UnetPlusPlusCGMDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        
        self.pooling_heads=[torch.nn.AvgPool3d(
            kernel_size=[chs//decoder_channels[-1],1,1]
        ) for chs in decoder_channels]
        
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3)

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        # print([o.shape for o in decoder_output])
        pooled_output = [h(raw_mask) for h,raw_mask in zip(self.pooling_heads,decoder_output)]
        # print([o.shape for o in pooled_output])
        masks=[*map(self.segmentation_head,pooled_output)]
        
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
class UnetPlusPlusCosSim(SegmentationModel):
    """
    Reference:
        https://arxiv.org/abs/1807.10165
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        
        
        self.segmentation_head = head.MetricLayer(
            in_channels=decoder_channels[-1],
            out_channels=classes
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "unetpluspluscossim-{}".format(encoder_name)
        self.initialize()