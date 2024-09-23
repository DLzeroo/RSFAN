import torch

model_path = './resnext101_ibn_a-6ace051d.pth'

model_dict = torch.load(model_path)

for i in model_dict.keys():
    print(i)
