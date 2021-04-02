# import yaml
#
# with open('data.yaml') as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)
#
# print(data['train'])
# print(data['val'])
# print(data['nc'])
# print(data['names'])


import torch

predictions = torch.randn((2,3, 13, 13, 9))

best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
print(best_class.shape)






