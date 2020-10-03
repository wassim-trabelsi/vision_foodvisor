from torchvision import models
import torch.nn as nn

vgg_model = models.vgg16(pretrained=True)
vgg_model.classifier._modules['6'] = nn.Linear(in_features = 4096, out_features = 2, bias = True)
vgg_model.to('cuda')
print('OK using pretrained vgg16')