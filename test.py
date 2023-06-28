from utils.models import InitialResidualNet, DecodeConsecutiveConvNets, AutoEncoder
import torch

auto_encoder = AutoEncoder()
input = torch.randn((32, 1, 157, 189, 156))

output = auto_encoder(input)

print(output.shape)



