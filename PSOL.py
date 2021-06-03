from sklearn.decomposition import PCA
import torch
#PSOL主要由两部分组成：1.生成伪目标框的算法 2.有监督训练的目标定位



#DDT Deep Descriptor Transforming
#学习同类别图片的共同特征实现无标记数据中的目标定位
class DDT:
    def __init__(self,feature_maps):
        self.feature_maps = feature_maps #从CNN中提取的特征 [class_num,C,H,W] 其中class_num为同类图片数量

    def calculate_descriptors(self):
        descriptor_num = self.feature_maps.size(1)
        descriptors = self.feature_maps.view(-1,descriptor_num)
        descriptors_mean = sum(descriptors)/len(descriptors)
        pca = PCA(n_components=1)
        pca.fit(descriptors)
        trans_vec = pca.components_[0] #[channel]
        trans_vec = torch.FloatTensor(trans_vec)
        return trans_vec,descriptors_mean

    def locate(self,img_feature_map,trans_vec,descriptors_mean):
        '''
        对同类图片中的物体进行定位
        :param img_feature_map:同类图片经过CNN后提取的特征[num,C,H,W]
        :param trans_vec:最大特征向量 [C]
        :param descriptors_mean:特征均值向量 [C]
        :return:
        '''
        num,channel_num,H,W = img_feature_map.size()
        img_feature_map = img_feature_map.view(num,-1,channel_num)
        img_feature_map = img_feature_map - descriptors_mean
        indicator_matrix = torch.matmul(img_feature_map,trans_vec)
        indicator_matrix = indicator_matrix.view(num,H,W)









