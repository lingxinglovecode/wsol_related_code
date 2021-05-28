import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
import numpy as np
#SPG用一种高层提取特征生成像素级别的标签指导底层进行学习


#对于scomaps的标签生成的过程
class Scoremap_loss:
    def __init__(self):
        self.scoremap_loss =  nn.BCEWithLogitsLoss()

    def get_loss(self,scoremaps):
        mask = torch.zeros_like(scoremaps).fill_(255)
        mask_label = self.generate_mask(mask, scoremaps)

        positions = mask_label.view(-1,1) < 255.0 #只计算背景和前景部分loss，值为255的部分忽略
        loss = self.scoremap_loss(scoremaps.view(-1,1)[positions],mask_label.view(-1,1)[positions])
        return loss

    def generate_mask(self,mask,scoremaps,th_high=0.7,th_low=0.05):
        '''
        生成掩膜，根据原文描述，前景部分应该为1，背景部分为0，其余部分为255
        :param mask: 输入的初始化掩膜，此时为全255 [batch,H,W]
        :param scoremaps: 最后层得到的CAM图 [B,H,W]
        :param th_high:  前景阈值
        :param th_low: 背景阈值
        :return: 生成的前后景标签
        '''
        # mask label for segmentation
        mask = self.mark_obj(mask, scoremaps, 1.0, th_high)
        mask = self.mark_bg(mask, scoremaps, th_low)

        return mask

    def mark_obj(self, label_img, heatmap, label, threshold=0.5):
        '''
        生成前景部分的标签
        :param label_img: 输入的初始化标签掩膜[B,H,W]
        :param heatmap: CAM图 [B,H,W]
        :param label:  对前景部分的标签
        :param threshold:  分割前景的阈值
        :return:  前景部分标记为label后的类别掩膜 [B,H,W] With object location labeled 1 and others 255
        '''
        #得到前景区域
        if isinstance(label, (float, int)):
            np_label = label
        else:
            np_label = label.cpu().data.numpy().tolist()
        #对batch中每一个图片进行处理
        for i in range(heatmap.size()[0]):
            mask_pos = heatmap[i] > threshold
            if torch.sum(mask_pos.float()).data.cpu().numpy() < 30: #如果满足条件的部分较少那就减少阈值
                threshold = torch.max(heatmap[i]) * 0.7
                mask_pos = heatmap[i] > threshold
            label_i = label_img[i]
            if isinstance(label, (float, int)):
                use_label = np_label
            else:
                use_label = np_label[i]
            # label_i.masked_fill_(mask_pos.data, use_label)
            label_i[mask_pos.data] = use_label
            label_img[i] = label_i

        return label_img

    def mark_bg(self, label_img, heatmap, threshold):
        '''
        生成背景部分的标签
        :param mask: 类别掩膜 [B,H,W]
        :param scoremaps: CAM [B,H,W]
        :param th_low: 背景阈值，小于该值则标记为背景0
        :return: 标记完背景后的mask [B,H,W]
        '''
        mask_pos = heatmap < threshold
        # label_img.masked_fill_(mask_pos.data, 0.0)
        label_img[mask_pos.data] = 0.0
        return label_img



if __name__ == '__main__':
    scoremaps = torch.randn((32,224,224)) #插值放大后的scoremaps
    S_loss = Scoremap_loss()
    loss = S_loss.get_loss(scoremaps)
    a = 2



