"""
From https://arxiv.org/pdf/1707.03321.pdf as referenced by https://ieeexplore.ieee.org/document/8918693
"""

import torch
from torch import nn

from thesis.structs.chambon_structs import ChambonExtendableConfig

class ChambonBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pooling):
        super(ChambonBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((1, pooling))

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ChambonNetExtendable(nn.Module):
    """
    An implementation of the model used in the paper "A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series"
    """
    def __init__(self, config: ChambonExtendableConfig):
        super(ChambonNetExtendable, self).__init__()
        
        self.conv1 = nn.Conv2d(1, config.C, (config.C, 1))

        assert config.k % 2 == 1, "k must be odd"
        p = (config.k - 1) / 2

        layers = []
        current_num_channels = 1
        for i in range(config.num_blocks):
            new_num_channels = 8
            layers.append(ChambonBlock(current_num_channels, new_num_channels, kernel_size=(1, config.k), stride=(1, 1), padding=(0, int(p)), pooling=config.m))
            current_num_channels = new_num_channels

        # Move the layers to a module list so that they are registered as submodules
        self.layers = nn.ModuleList(layers)

        output_size = config.T
        for layer in self.layers:
            output_size = output_size // config.m
        output_size *= current_num_channels * config.C

        self.fc = nn.Linear(output_size, config.D)

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
        
        # The rest of the convolutions are purely over the temporal dimension of one virtual channel at a time
        # Since the virtual channels are a linear combination of the original channels, the nice properties of using separated channels holds while
        # also allowing mixed information from all the channels to be used in the temporal filtering
        for layer in self.layers:
            x = layer(x)
        
        # Flatten
        x = x.view(x.shape[0], -1)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    config = ChambonExtendableConfig(C=19, T=3840, k=63, m=2, num_blocks=10, D=512)
    model = ChambonNetExtendable(config)
    test_input = torch.randn(32, config.C, config.T)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
