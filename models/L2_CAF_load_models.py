import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torchvision import transforms as T

def get_activation(feature_maps):  ## forward hook to catch the feature maps at a certain layer
    def hook(model, input, output):
        feature_maps.append( output.detach())
    return hook

def load_architecture(arch_name):
    feature_maps = []
    if arch_name in ['VGG']:
        model = models.vgg16(pretrained=True)

        if torch.cuda.is_available():
            model = model.cuda()
        model.features.register_forward_hook(get_activation(feature_maps))

        ## Replicate the layers after the last conv layer
        post_conv_subnet = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            nn.Flatten(),
            model.classifier,
            )
    else:
        raise NotImplementedError('Invalid arch_name {}'.format(arch_name))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    post_conv_subnet.eval()

    return model, feature_maps, post_conv_subnet

if __name__ == '__main__':
    load_architecture('VGG')