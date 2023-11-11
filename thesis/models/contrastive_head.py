"""
Simple 1 layer contrastive head that projects the logits to a lower dimensional space where the contrastive loss is applied
"""

import torch
from torch import nn

from thesis.structs.contrastive_head_structs import HeadStyle, ContrastiveHeadConfig

class ContrastiveHead(nn.Module):
    """
    A simple model to project the logits to a lower dimensional space where the contrastive loss is applied

    Can follow a selection of implementations from the literature:
    1. Linear projection - Single layer linear projection
    2. MLP - n-layer perceptron with ReLU activations
    3. SimCLR - n-layer perceptron with ReLU activations and a batchnorm layer
    4. MoCo - Same as SimCLR
    """
    def __init__(self, config: ContrastiveHeadConfig):
        super(ContrastiveHead, self).__init__()
        self.logit_dimension = config.logit_dimension
        self.c_loss_dimension = config.c_loss_dimension
        self.head_style = config.head_style
        self.num_layers = len(config.layer_sizes)
        self.normalize = config.normalize
        layer_sizes = [self.logit_dimension] + config.layer_sizes
        if self.head_style == HeadStyle.LINEAR:
            assert self.num_layers == 1, "Linear head only supports 1 layer"
            self.projection = nn.Linear(self.logit_dimension, self.c_loss_dimension)
        elif self.head_style == HeadStyle.MLP:
            # We create a sequential model with the number of layers specified
            # The last layer does not have any activation
            layers = []
            for i in range(self.num_layers - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.projection = nn.Sequential(*layers)
        elif self.head_style == HeadStyle.SIMCLR or self.head_style == HeadStyle.MOCO:
            # Similar to MLP but with batchnorm layers
            layers = []
            for i in range(self.num_layers - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.projection = nn.Sequential(*layers)
        else:
            raise ValueError("Head style not supported")

    def forward(self, x, normalize=None):
        # Input shape: (batch_size, logit_dimension)
        # Output shape: (batch_size, c_loss_dimension)
        features = self.projection(x)
        if (normalize is None and self.normalize) or normalize:
            features = nn.functional.normalize(features, dim=1)
        return features

def create_contrastive_module(embedding_module: nn.Module, head_config: ContrastiveHeadConfig):
    """
    Combines the embedding module and a contrastive head and returns the resulting model
    """
    contrastive_head = ContrastiveHead(head_config)
    return nn.Sequential(embedding_module, contrastive_head)


if __name__ == "__main__":
    test_embedding_model = nn.Linear(512, 1024)

    config = ContrastiveHeadConfig(logit_dimension=1024, c_loss_dimension=128, head_style=HeadStyle.MLP, layer_sizes=[512, 128])
    linear_contrastive_module = create_contrastive_module(test_embedding_model, config)
    mlp_contrastive_module = create_contrastive_module(test_embedding_model, config)
    simclr_contrastive_module = create_contrastive_module(test_embedding_model, config)
    moco_contrastive_module = create_contrastive_module(test_embedding_model, config)