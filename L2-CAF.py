import torch
import torch.nn as nn
import torch.nn.functional as F
from models.L2_CAF_load_models import load_architecture
from ImageSet import *
from torchvision import transforms as T
import torch.optim as optim
import numpy as np
import utils
#L2-CAF核心部分其实就是一个可学习的掩膜

def polynomial_lr_decay(global_step,
             init_learning_rate,
             max_iter,
             min_learning_rate=1e-5):
    power = 1
    lr = (init_learning_rate - min_learning_rate) * ((1 - global_step / max_iter) ** power) + min_learning_rate
    return lr

def normalize_filter(_atten_var,filter_type='l2norm'):
    if filter_type == 'l2norm':
        frame_mask = np.reshape(np.abs(_atten_var), (_atten_var.shape[0], _atten_var.shape[1]))
        frame_mask = frame_mask / np.linalg.norm(frame_mask)
    else:
        raise NotImplementedError('Invalid filter type {}'.format(filter_type))

    return frame_mask

def load_image(data_dir,img_name):
    test_img = Image.open('{}/{}'.format(data_dir, img_name))
    resize_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    normalize_transform = T.Compose([

        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    raw_img = resize_transform(test_img)
    normalized_img = normalize_transform(raw_img)
    normalized_img = normalized_img.unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        normalized_img = normalized_img.cuda()

    return raw_img.permute(1, 2, 0), normalized_img


class L2_CAF(nn.Module):
    def __init__(self,spatial_dim):
        super().__init__()
        self.filter = nn.Parameter(torch.ones((spatial_dim,spatial_dim),dtype=torch.float32),requires_grad=True)

    def forward(self, A):
        return A * F.normalize(self.filter,dim=[0,1])


if __name__ == '__main__':

    ##load images
    rgb_img , pytorch_img = load_image('/Users/lianxing/Desktop/目前学习.nosync/wsol_related_code/data/horse','0213.jpg')


    model, last_layer_feature_maps, post_conv_subnet = load_architecture('VGG')


    NT = model(pytorch_img)
    A = last_layer_feature_maps[-1]
    l2_caf = L2_CAF(A.shape[-1])

    max_iter = 500
    initial_lr = 0.1
    l2_loss = nn.MSELoss()
    optimizer = optim.SGD(l2_caf.parameters(),lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: polynomial_lr_decay(step,
                                                                                      init_learning_rate=initial_lr,
                                                                                      max_iter=max_iter) / initial_lr)

    iteration = 0
    MAX_INT = np.iinfo(np.int16).max
    prev_loss = torch.tensor(MAX_INT)
    min_error = 1e-5
    while iteration<max_iter:
        FT = post_conv_subnet(l2_caf(A))
        loss = l2_loss(FT, NT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if iteration % 50 == 0:
            if torch.abs(loss.item() - prev_loss) < min_error:
                break
            prev_loss = loss

        iteration += 1

    print('Done after {} iterations'.format(iteration))
    frame_mask =  l2_caf.filter.detach().cpu().numpy()
    utils.apply_heatmap(rgb_img, frame_mask, alpha=0.6,
                                # save=output_dir + img_name + '_cls_oblivious_{}.png'.format(arch_name),
                                save='cls_oblivious.png',
                                axis='off', cmap='bwr')
