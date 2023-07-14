import torch
import torch.nn as nn
from utils.models import Consecutive3DConvLayerBlock, PatchEmbedding, PositionEmbedding, \
    MultiheadSelfAttention, ViTEncoder, Encoder, TransformerEncoder, InitialResidualNet, \
    DecodeConsecutiveConvNets, AutoEncoder, AutoEncoderWithoutShortcuts
import matplotlib.pyplot as plt
import datetime
from torchinfo import summary
import sys


if __name__ == '__main__':
    print("ConvVit v 1.0.3 (Without Shortcuts)", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Test Beginning: ")
    start = datetime.datetime.now()
    batch_size = 2
    x = torch.randn((batch_size, 1, 157, 189, 156))
    print("Input shape:", x.shape, "([batch_size, channels, depth, height, width])")
    model = AutoEncoderWithoutShortcuts()
    output, _ = model(x)
    end = datetime.datetime.now()
    print("Output shape:", output.shape, "([batch_size, channels, depth, height, width])")
    print("Total Time Elapsed:", end - start)
    print("Test Finished.")