import torch
import torch.nn as nn
from utils.models import Consecutive3DConvLayerBlock, PatchEmbedding, PositionEmbedding, \
    MultiheadSelfAttention, ViTEncoder, Encoder
import matplotlib.pyplot as plt
import datetime
from torchinfo import summary
import sys


if __name__ == '__main__':
    model = Encoder(conv_in_features = 1, conv_out_features = 128, kernel_size = 3, padding = True,
                    batch_norm = True, img_size = 256, in_channels = 8, patch_size = 16,
                    num_transformer_layer = 2, embedding_dim = 2048, mlp_size = 4096, num_heads
                    = 16, attention_dropout = .0, mlp_dropout = .1, embedding_dropout = .1,
                    final_latent_space_dim = 2048)

    # sys.stdout = open("model.txt", "w")
    # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # summary(model, input_size = (1, 1, 157, 189, 156),
    #         col_names = ["input_size", "output_size", "num_params", "trainable"],
    #         col_width = 20,
    #         row_settings = ["var_names"])
    # sys.stdout.close()

    x = torch.randn((32, 1, 157, 189, 156))
    x = model(x)
    print(x.shape)











