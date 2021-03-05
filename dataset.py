import torch.utils.data as data
from glob import glob
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class AVA_Dataset(data.Dataset):
    def __init__(self):
        self.sort = lambda x: int(x.split("/")[-1].split("_")[0])
        self.imgs = glob(r"**/datasets/AVA_goodimages/*.jpg", recursive=True)
        self.imgs.sort(key=self.sort)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).convert("RGB")
        data = self.transforms(pil_img)
        return data

    def __len__(self):
        return(len(self.imgs))

class Train_Dataset(data.Dataset):
    def __init__(self):
        self.imgs = glob(r"**/SALICON_images/train/*.jpg", recursive=True)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).convert("RGB")
        data = self.transforms(pil_img)
        return data

    def __len__(self):
        return(len(self.imgs))

class Test_Dataset(data.Dataset):
    def __init__(self):
        self.imgs = glob(r"**/SALICON_images/test/*.jpg", recursive=True)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).convert("RGB")
        data = self.transforms(pil_img)
        return data

    def __len__(self):
        return(len(self.imgs))


if __name__ == "__main__":
    dataset = AVA_Dataset()
    dataset_H = Test_Dataset()
    print(dataset[0].size(), dataset[1].size(), dataset.__len__())


    