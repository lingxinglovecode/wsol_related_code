from sklearn.decomposition import PCA
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ImageSet import ImageDataSet
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from utils import normalize_tensor
from utils import normalize_scoremap
import cv2
import numpy as np
#PSOL主要由两部分组成：1.生成伪目标框的算法 2.有监督训练的目标定位



#DDT Deep Descriptor Transforming
#学习同类别图片的共同特征实现无标记数据中的目标定位
class DDT:
    def __init__(self,feature_maps):
        self.feature_maps = feature_maps #从CNN中提取的特征 [class_num*H*W,C] 其中class_num为同类图片数量

    def calculate_descriptors(self):
        descriptors = self.feature_maps
        descriptors_mean = torch.sum(descriptors,dim=0)/descriptors.size(0)
        pca = PCA(n_components=1)
        pca.fit(descriptors.detach().cpu().numpy())
        trans_vec = pca.components_[0] #[channel]
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
        img_feature_map = img_feature_map.view(num,channel_num,-1)
        descriptors_mean = descriptors_mean.view(1,-1,1)
        img_feature_map = img_feature_map - descriptors_mean
        trans_vec = torch.FloatTensor(trans_vec).cuda()
        trans_vec = trans_vec.repeat(5, 1, 1)
        indicator_matrix = torch.matmul(trans_vec,img_feature_map)
        indicator_matrix = indicator_matrix.squeeze(1)
        indicator_matrix = indicator_matrix.view(num,H,W)
        heatmaps = indicator_matrix.detach().cpu().numpy()

        for i in range(num):
            heatmap = 1 - heatmaps[i]
            heatmap = cv2.resize(heatmap, (224,224),interpolation=cv2.INTER_CUBIC)
            heatmap = normalize_scoremap(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap),cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap / np.max(heatmap)

            plt.imshow(np.uint8(255 * cam))
            plt.show()

class Feature_extractor:
    def __init__(self):
        state_dict = torch.load("D://personal_file//code_test//pretrained_models//vgg19-dcbb9e9d.pth")
        model = models.vgg19(pretrained=False)
        model.load_state_dict(state_dict)
        self.pretrained_model = model.features.cuda()

    def extract(self,img_dir):

        #img_process
        dataset = ImageDataSet(img_dir)
        features = torch.ones((1,512)).cuda()
        for index in range(len(dataset)):
            images = dataset[index]
            feature = self.pretrained_model(images.unsqueeze(0))
            feature = feature.view(-1,512)
            features = torch.cat((features,feature),dim=0)
        features = features[1:]
        return features

    def extract_test(self,img_dir):
        dataset = ImageDataSet(img_dir)
        dataloader = DataLoader(dataset,batch_size=5)
        for batch_idx,imgs in enumerate(dataloader):
            features = self.pretrained_model(imgs)
        return features



if __name__ == '__main__':
    data_train_path = 'D://personal_file//code_test//wsol_related//data//horse'
    data_test_path = 'D://personal_file//code_test//wsol_related//data//horse'
    feature_ex = Feature_extractor()
    feature_maps = feature_ex.extract(data_train_path)
    ddt = DDT(feature_maps)
    trans_vec,descriptors_mean = ddt.calculate_descriptors()

    features_testset = feature_ex.extract_test(data_test_path)
    ddt.locate(features_testset,trans_vec,descriptors_mean)







