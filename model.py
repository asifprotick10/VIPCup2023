# model.py
import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self, name='resnet101', num_classes=2): 
        super(ResNet, self).__init__()
        if name == 'resnet101':  
            self.encoder = torchvision.models.resnet101(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))
