from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from .layers import FrozenBatchNorm2d
from . import resnet_iseg


class resnet_iseg_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(resnet_iseg_feature_extractor, self).__init__()
        bn_layer = nn.BatchNorm2d
        if freeze_bn:
            bn_layer = FrozenBatchNorm2d
        self.backbone = resnet_iseg.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True], pretrained_weights=pretrained_weights,
            norm_layer=bn_layer)
        return_layers = {'layer4': 'out', 'layer1': 'low'}
        if aux:
            return_layers['layer3'] = 'aux'
        #self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x, interactive_fmaps=None):
        out = self.backbone(x, interactive_fmaps)

        return out


