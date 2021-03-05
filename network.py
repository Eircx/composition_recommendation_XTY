import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.image_discriminator = torchvision.models.vgg16(pretrained=False)
        self.fc = nn.Linear(1000, 2)
        
    def forward(self, x):
        x = self.image_discriminator(x)
        x = self.fc(x)
        return x

class ImageCropper(nn.Module):
    def __init__(self):
        super(ImageCropper, self).__init__()
        self.coordinate_generator = torchvision.models.vgg16(pretrained=False)
        self.fc = nn.Linear(1000, 4)
    
    def stn(self, x, coordinates):
        theta = torch.zeros(x.size(0), 2, 3).to(x.device)
        theta[:, 0, 0] = coordinates[:, 0]
        theta[:, 0, 2] = 2 - 4 * coordinates[:, 2]
        theta[:, 1, 1] = coordinates[:, 1]
        theta[:, 1, 2] = 2 - 4 * coordinates[:, 3]
        grid = F.affine_grid(theta, (x.size(0), x.size(1), 224, 224))
        cropped_images = F.grid_sample(x, grid)
        return cropped_images
        

    def forward(self, x):
        raw_coordinates = self.fc(self.coordinate_generator(x))
        coordinates = F.sigmoid(raw_coordinates)
        cropped_images = self.stn(x, coordinates)
        return cropped_images


if __name__ == "__main__":
    net1 = ImageCropper()
    net2 = ImageDiscriminator()
    print(net1)
    print(net2)
