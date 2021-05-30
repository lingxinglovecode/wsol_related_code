import torch
import random
import numpy as np

#DNet中的DA结构，用来计算不同channel之间余弦距离


#原文中的结构
def calculate_cosineloss(self, maps):
    '''
    计算余弦相似度
    :param maps: 在原文中[batch_size,k*3,h,w] 其中k自己选择代表一个类的cam对应的k个不同feature，3在原文中代表三个支路
    :return: 随机选择一半的feature计算余弦相似度之和
    '''
    batch_size = maps.size(0)
    num_maps = maps.size(1)
    channel_num = int(self.num_maps * 3 / 2) #选取一半的feature，由于原文中三个支路，所以这里*3/2
    eps = 1e-8
    random_seed = random.sample(range(num_maps), channel_num) #随机选取一半的feature
    maps = maps[:, random_seed, :, :].view(batch_size, channel_num, -1)

    X1 = maps.unsqueeze(1) # [batch_size,1,channel_num,H*W]
    X2 = maps.unsqueeze(2) # [batch_size,channel_num,1,H*W]
    dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
    #dot12 [batch_size,channel_num,channel_num]
    dist = dot12 / (torch.sqrt(dot11 * dot22 + eps)) #计算出来距离 [batch_size,channel_num,channel_num]
    tri_tensor = (
        (torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1] * channel_num))).expand(batch_size,
                                                                                                         channel_num,
                                                                                                         channel_num)).cuda()  # 去除掉对角线元素的上三角矩阵，全都为1，用来计算不同feature map之间距离
    dist_num = abs((tri_tensor * dist).sum(1).sum(1)).sum() / (
                batch_size * channel_num * (channel_num - 1) / 2)  # 除以的系数是batch * Cn2（从n个里面选2个）

    return dist_num, random_seed

def calculate_fm_cosineloss(feature_maps):
    '''
    计算feature map不同channel之间的余弦相似度差异
    :param feature_maps: [32,channel,h,w]
    :return: 余弦距离
    '''
    batch_size = feature_maps.size(0)
    channel_num = feature_maps.size(1)
    feature_maps = feature_maps.view(batch_size, channel_num, -1)
    X1 = feature_maps.unsqueeze(1)  # [batch_size,1,channel_num]
    X2 = feature_maps.unsqueeze(2)  # [batch_size,channel_num,1]
    dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
    # dot12 [batch_size,channel_num,channel_num]
    eps = 1e-8
    dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))  # 计算出来距离 [batch_size,channel_num,channel_num]
    tri_tensor = (
        (torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1] * channel_num))).expand(batch_size,
                                                                                                         channel_num,
                                                                                                         channel_num))  # 去除掉对角线元素的上三角矩阵，全都为1，用来计算不同feature map之间距离
    dist_num = abs((tri_tensor * dist).sum(1).sum(1)).sum() / (
            batch_size * channel_num * (channel_num - 1) / 2)  # 除以的系数是batch * Cn2（从n个里面选2个）
    return dist_num

if __name__ == '__main__':
    feature_maps = torch.randn((32,200,14,14))
    dist = calculate_fm_cosineloss(feature_maps)
    print(dist)

