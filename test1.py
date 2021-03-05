import torchvision
import torch
model = torchvision.models.resnet34()

input_img = torch.randn(5,3,57,32)
output_img = model(input_img)
print(output_img.size())

input_img = torch.randn(5,3,507,32)
output_img = model(input_img)
print(output_img.size())