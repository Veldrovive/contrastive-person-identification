"""
From https://arxiv.org/pdf/1707.03321.pdf as referenced by https://ieeexplore.ieee.org/document/8918693
"""

import torch
from torch import nn

from thesis.structs.chambon_structs import ChambonConfig

class ChambonNet(nn.Module):
    """
    An implementation of the model used in the paper "A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series"
    """
    def __init__(self, config: ChambonConfig):
        super(ChambonNet, self).__init__()

        self.min_t = config.k * config.m
        
        self.conv1 = nn.Conv2d(1, config.C, (config.C, 1))

        p2 = (config.k - 1) / 2
        assert int(p2) == p2, "k must be odd"
        self.conv2 = nn.Conv2d(1, 8, (1, config.k), padding=(0, int(p2)))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((1, config.m))

        p3 = (config.k - 1) / 2
        assert int(p3) == p3, "k must be odd"
        self.conv3 = nn.Conv2d(8, 8, (1, config.k), padding=(0, int(p3)))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((1, config.m))

    def forward(self, x):
        # Reshape to (batch_size, C, T, 1)
        x = x.unsqueeze(-1)

        # Conv2d expects (channels [1], C, T)
        x = x.permute(0, 3, 1, 2)
        
        # The first convolution operates on all the channels to combine all information into a set of representations with no temporal information
        # The argument is that this produces a set of virtual channels which can serve to perform spatial filtering. The discussion section of the paper goes more into depth.
        x = self.conv1(x)  # (batch_size, C, 1, T)
        # Transpose to get the channels last and prep for the next convolution which is now over the temporal dimension
        x = x.permute(0, 2, 1, 3)
        
        # The second and third convolutions are purely over the temporal dimension of one virtual channel at a time
        # Since the virtual channels are a linear combination of the original channels, the nice properties of using separated channels holds while
        # also allowing mixed information from all the channels to be used in the temporal filtering
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Conv block 2
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        # Flatten
        x = x.view(x.shape[0], -1)
        
        return x

if __name__ == "__main__":
    config = ChambonConfig(C=64, T=360, k=63, m=16)
    model = ChambonNet(config)
    test_input = torch.randn(32, config.C, config.T)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
