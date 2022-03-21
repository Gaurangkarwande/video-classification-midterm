import torch
from torch import nn
from models.ConvLayer import ConvLayer
from models.CLassifier import Classifier



class FrameNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.cnn = ConvLayer(config)
        self.classifier = Classifier(config)
    
    def forward(self, x):
        x = self.cnn(x)
        return self.classifier(x)

class MultiResFrameNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        config['linear_in'] *= 2
        self.cnn1 = ConvLayer(config, is_multires=True)
        self.cnn2 = ConvLayer(config, is_multires=True)
        self.classifier = Classifier(config)
    
    def forward(self, x1, x2):
        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x = torch.concat((x1, x2), dim=1)
        return self.classifier(x)
        

