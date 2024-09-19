import torch

model_path = './resnext101_ibn_a-6ace051d.pth'
# model_path = './resnet50-19c8e357.pth'
model_dict = torch.load(model_path)

# print(model_dict.keys())
# for key in model_dict.keys():
#     print(key)



for i in model_dict.keys():
    print(i)
# for k, v in model_dict['state_dict'].items():
#     new_key = k.replace('module.', '')  # 去掉前缀
#     new_key = new_key.replace('fc.', '')  # 去掉后缀
#     if new_key in model_dict:
#         model_dict[new_key].copy_(v)