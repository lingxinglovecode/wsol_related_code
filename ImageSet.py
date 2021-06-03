
from torch.utils.data import Dataset,DataLoader
import os
import glob
from PIL import Image
from torchvision import transforms
_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
class ImageDataSet(Dataset):
    def __init__(self,path):
        self.files = sorted(glob.glob("%s/*.*"%path))
        self.transforms= transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ])

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img.cuda()
    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    path = "/Users/lianxing/Desktop/目前学习.nosync/wsol_related_code/data"
    img_set = ImageDataSet(path)
    img_load= DataLoader(img_set,batch_size=1)
    for batch_idx,images in enumerate(img_load):
        print(images)


