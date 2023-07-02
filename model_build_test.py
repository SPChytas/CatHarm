import torch
import torch.nn as nn
from utils.models import Consecutive3DConvLayerBlock, PatchEmbedding, PositionEmbedding, \
    MultiheadSelfAttention, ViTEncoder, Encoder, TransformerEncoder, InitialResidualNet, \
    DecodeConsecutiveConvNets, AutoEncoder
import matplotlib.pyplot as plt
import datetime
from torchinfo import summary
import sys


if __name__ == '__main__':


    x = torch.randn((32, 1, 157, 189, 156))
    print("Input shape:", x.shape, "([batch_size, channels, depth, height, width])")

    model = AutoEncoder()

    output = model(x)
    print("Output shape:", output.shape, "([batch_size, channels, depth, height, width])")
