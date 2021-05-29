#CutMix的核心代码，主要是将图片进行裁剪后mix

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def get_img_tensor(dir):
    img = Image.open(dir)
    img = img.convert('RGB')
    img_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                      std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
    ])
    img_tensor = img_transforms(img)
    return img_tensor

def CutMix(target_a,target_b,beta=1):
    lam = np.random.beta(beta,beta) #根据原文中的描述，组合率lam通过beta分布获取，并且固定beta为1则分布为uniform
    bbx1, bby1, bbx2, bby2 = rand_bbox(target_a.size(),lam)
    target_b[:,:,bbx1:bbx2,bby1:bby2] = target_a[:,:,bbx1:bbx2,bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (target_a.size()[-1] * target_a.size()[-2])) #由于在之前组合率下可能超出图片范围，所以在调整之后需要重新计算组合率
    return target_b,lam

def rand_bbox(size,lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat) #计算裁剪宽度
    cut_h = np.int(h * cut_rat) #计算裁剪高度

    cx = np.random.randint(w) #计算裁剪区域的中心坐标x
    cy = np.random.randint(h) #计算裁剪区域的中心坐标y

    #计算bbox的左上角和右下角坐标
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bby1, bbx2, bby2
if __name__ == '__main__':

    rand_bbox((1,3,224,224),0.2)
    target = get_img_tensor('1.png').unsqueeze(0)
    mix_img = get_img_tensor('2.png').unsqueeze(0)
    mixed_img,_ = CutMix(target,mix_img)
    img = mixed_img.squeeze(0).numpy()
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    plt.imshow(img)
    plt.show()  # 显示图片
