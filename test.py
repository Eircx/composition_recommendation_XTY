import torchvision
import torch.nn as nn
import numpy as np
from glob import glob
from PIL import Image

img_list = glob(r"**/test/*.jpg", recursive=True)
print(len(img_list))

theta_np = np.array([[0.5, 0, 0.75], [0, 0.75, 0]]).reshape(1, 2, 3)
print(theta_np)


import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from os import path

current_dir = path.dirname(__file__)
theta = torch.Tensor([[2,0,2],[0,2,2]]).unsqueeze(dim=0)
img = cv2.imread(img_list[0])
plt.subplot(2,1,1)
plt.imshow(img)
print(img.max())
print(img.min())
plt.axis('off')
img = torch.Tensor(img.transpose(2,0,1)).unsqueeze(0)
grid =  F.affine_grid(theta,size=img.shape)
output = F.grid_sample(img,grid)
print(output.shape)
print(output.max())
print(output.min())
output = output[0].numpy().transpose(1,2,0)
print(img.shape, output.shape)
plt.subplot(2,1,2)
plt.imshow(output/255)
plt.axis('off')
plt.savefig(path.join(current_dir, "fig.jpg"))
