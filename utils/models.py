
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# from unet_parts import *

class MLP_encdec(nn.Module):
	def __init__(self, in_depth, hidden_depths, out_depth, batchnorm=True, dropout=0.):
		super(MLP_encdec, self).__init__()

		self.batchnorm = batchnorm
		self.dropout = dropout
		self.depths = [in_depth, *hidden_depths, out_depth]

		self.linear_layers = nn.ModuleList([])
		self.norm = nn.ModuleList([])
		self.act = nn.ModuleList([])

		for i in range(len(self.depths) - 1):
			self.linear_layers.append(nn.Linear(self.depths[i], self.depths[i + 1], bias=not batchnorm))
			if i != len(self.depths) - 2:
				if batchnorm:
					self.norm.append(nn.BatchNorm1d(self.depths[i + 1], eps=1e-8))
				self.act.append(nn.GELU())

	def forward(self, x):
		for i in range(len(self.depths) - 1):

			if self.dropout > 0.:
				x = F.dropout(x, self.dropout, self.training)
			
			x = self.linear_layers[i](x)
			
			if i != len(self.depths) - 2:
				if self.batchnorm:
					x = self.norm[i](x)
				x = self.act[i](x)

		return x


class MLP(nn.Module):
	def __init__(self, in_depth, hidden_depths, out_depth, batchnorm=True, dropout=0., classification=True):
		super(MLP, self).__init__()
		self.batchnorm = batchnorm
		self.dropout = dropout
		self.depths = [in_depth, *hidden_depths, out_depth]

		self.linear_layers = nn.ModuleList([])
		self.norm = nn.ModuleList([])
		self.act = nn.ModuleList([])

		self.classification = classification


		if (self.classification):
			if (out_depth == 1):
				self.final_activation = nn.Sigmoid()
			else:
				self.final_activation = nn.Softmax(1)
		else:
			self.final_activation = nn.Identity()

		for i in range(len(self.depths) - 1):
			self.linear_layers.append(nn.Linear(self.depths[i], self.depths[i + 1], bias=not batchnorm))
			if i != len(self.depths) - 2:
				if batchnorm:
					self.norm.append(nn.BatchNorm1d(self.depths[i + 1], eps=1e-8))
				self.act.append(nn.GELU())

	def forward(self, x):
		for i in range(len(self.depths) - 1):
			if self.dropout > 0.:
				x = F.dropout(x, self.dropout, self.training)
			x = self.linear_layers[i](x)
			if i != len(self.depths) - 2:
				if self.batchnorm:
					x = self.norm[i](x)
				x = self.act[i](x)
			#if i == len(self.depths) - 3:
			#	latent = x
		#return x, latent

		x = self.final_activation(x)

		return x
		
	def l1_norm(self):
		return sum([torch.norm(l.weight, 1) for l in self.linear_layers])
	
	def l2_norm(self):
		return sum([torch.norm(l.weight, 2) for l in self.linear_layers])



class Encoder(nn.Module):
	def __init__(self,
				 conv_in_features: int = 1,
				 conv_out_features: int = 128,
				 kernel_size: int = 3,
				 padding: bool = True,
				 batch_norm: bool = True,
				 img_size: int = 256,
				 in_channels: int = 8,
				 patch_size: int = 16,
				 num_transformer_layer: int = 6,
				 embedding_dim: int = 2048,
				 mlp_size: int = 4096,
				 num_heads: int = 16,
				 attention_dropout: float = .0,
				 mlp_dropout: float = .1,
				 embedding_dropout: float = .1,
				 final_latent_space_dim: int = 2048):
		super().__init__()
		self.num_patches = (img_size * img_size) // (patch_size ** 2)
		self.conv_block = Consecutive3DConvLayerBlock(in_channel = conv_in_features,
													  out_channel = conv_out_features,
													  kernel_size = kernel_size,
													  padding = padding,
													  batch_norm = batch_norm)
		self.vit_block = ViTEncoder(img_size = img_size,
									in_channels = in_channels,
									patch_size = patch_size,
									num_transformer_layer = num_transformer_layer,
									embedding_dim = embedding_dim,
									mlp_size = mlp_size,
									num_heads = num_heads,
									attention_dropout = attention_dropout,
									mlp_dropout = mlp_dropout,
									embedding_dropout = embedding_dropout,
									final_latent_space_dim = final_latent_space_dim)
	def forward(self, x):
		return self.vit_block(self.conv_block(x))


class AutoEncoder(nn.Module):
	"""
	Input: [batch_size, channels, 157, 189, 156] (recommended)
	Output: [batch_size, channels, 157, 189, 156] (recommended)

	User Guide:
	* conv_in_channels: the channels of the inputs
	conv_out_channels: the channels after consecutive 3D CNNs
					   64, then the sequence length / number of patches will be 1024
					   256, then the sequence length / number of patches will be 4096
	* kernel_size: the kernel size of 3D CNNs and Transpose 3D CNNs
	* padding: whether we need to pad the dimensions
	* batch_norm: whether we need to normalization
	* patch_size: the patch size of the vit block
	num_transformer_layer: the number of transformer layers in the vit block
	* embedding_dim: the embedding dimensions of the vit block
	mlp_size: the size of the multi-layer perception block
	num_heads: the attention heads in the multi-head self-attention(MSA) layer
	attention_dropout: the percentage of drop-out in the multi-head self-attention(MSA) layer
	mlp_dropout: the percentage of drop-out in the multi-layer perception layer
	embedding_dropout: the percentage of drop-out after position embedding (before vit encoder)
	final_latent_space_dim: the final latent space dimension that the user want
							e.g. [1, final_latent_space_dim]

	Note: We recommend using the default value in the arguments starting with *.
		  Some unknown errors will occur if the arguments starting with * are changed.
	"""
	def __init__(self,
				 conv_in_channels: int = 1,
				 conv_out_channels: int = 64,
				 kernel_size: int = 3,
				 padding: bool = True,
				 batch_norm: bool = True,
				 patch_size: int = 16,
				 num_transformer_layer: int = 2,
				 embedding_dim: int = 256,
				 mlp_size: int = 2048,
				 num_heads: int = 8,
				 attention_dropout: float = .0,
				 mlp_dropout: float = .1,
				 embedding_dropout: float = .1,
				 final_latent_space_dim: int = 2048):
		super().__init__()

		assert conv_out_channels == 64 or conv_out_channels == 256, "Unsupportable Channels"

		self.conv_block = Consecutive3DConvLayerBlock(in_channel = conv_in_channels,
													  out_channel = conv_out_channels,
													  kernel_size = kernel_size,
													  padding = padding,
													  batch_norm = batch_norm)
		self.vit_block = ViTEncoder(in_channels = conv_in_channels,
								    out_channels = conv_out_channels,
									patch_size = patch_size,
									num_transformer_layer = num_transformer_layer,
									embedding_dim = embedding_dim,
									mlp_size = mlp_size,
									num_heads = num_heads,
									attention_dropout = attention_dropout,
									mlp_dropout = mlp_dropout,
									embedding_dropout = embedding_dropout,
									final_latent_space_dim = final_latent_space_dim)
		self.initial_residual = InitialResidualNet(final_latent_space_dim = final_latent_space_dim,
												   patch_size = patch_size,
												   embedding_dim = embedding_dim,
												   in_channels = conv_in_channels,
												   out_channels = conv_out_channels)

		self.consecutive_transpose_convnets = DecodeConsecutiveConvNets(in_channel =
																		conv_in_channels,
																		out_channel =
																		conv_out_channels,
																		kernel_size = kernel_size,
																		padding = True)

	def forward(self, x):
		x, cache = self.conv_block(x)
		x, patch_embedded = self.vit_block(x)
		x = self.initial_residual(x, patch_embedded)
		x = self.consecutive_transpose_convnets(x, cache)

		return x






class ViTEncoder(nn.Module):
	def __init__(self,
				 img_size: int = 256,
				 in_channels: int = 8,
				 patch_size: int = 16,
				 num_transformer_layer: int = 6,
				 embedding_dim: int = 2048,
				 mlp_size: int = 4096,
				 num_heads: int = 16,
				 attention_dropout: float = .0,
				 mlp_dropout: float = .1,
				 embedding_dropout: float = .1,
				 final_latent_space_dim: int = 2048):
		super().__init__()

		assert img_size % patch_size == 0, f"Input Size: {img_size} must be divisible by Patch " \
										   f"Size: {patch_size}"

		self.num_patches = (img_size * img_size) // (patch_size ** 2)

		self.position_embedding = PositionEmbedding(num_patches = self.num_patches,
													embedding_dimension = embedding_dim)

		self.embedding_dropout = nn.Dropout(p = embedding_dropout)

		self.patch_embedding = PatchEmbedding(in_channels = in_channels,
											  patch_size = patch_size,
											  embedding_dim = embedding_dim)

		self.transformer_encoder = nn.Sequential(*[TransformerEncoder(embedding_dim =
																	  embedding_dim,
																	  num_heads = num_heads,
																	  mlp_size = mlp_size,
																	  mlp_dropout = mlp_dropout,
																	  attention_dropout =
																	  attention_dropout) for _
												 in range(num_transformer_layer)])

		self.latent_space = nn.Sequential(nn.Flatten(),
										  nn.LayerNorm(normalized_shape = embedding_dim *
																		  self.num_patches),
										  nn.Linear(in_features = embedding_dim * self.num_patches,
													out_features = final_latent_space_dim),
										  nn.Dropout(mlp_dropout))

	def forward(self, x):
		x = self.patch_embedding(x)
		x = self.position_embedding(x)
		x = self.embedding_dropout(x)
		x = self.transformer_encoder(x)
		x = self.latent_space(x)
		return x

	def _get_name(self):
		return "Vision Transformer Encoder"

class Consecutive3DConvLayerBlock(nn.Module):
	def __init__(self,
				 in_channel: int = 1,
				 out_channel: int = 128,
				 kernel_size: int = 3,
				 padding: bool = True,
				 batch_norm: bool = True) -> None:
		super().__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.padding = padding
		self.batch_norm = batch_norm


		self.conv1 = nn.Conv3d(in_channels=self.in_channel,
							   out_channels = 32,
							   kernel_size = (self.kernel_size, self.kernel_size,
											  self.kernel_size),
							  stride = self.kernel_size)

		self.conv2 = nn.Conv3d(32, 64, (3, 3, 3), padding = 1, stride = 2)
		self.conv3 = nn.Conv3d(64, self.out_channel, (3, 3, 3), padding = 1, stride = 2)

	def forward(self, x):
		self.batch_size, self.channels = x.shape[0], x.shape[1]
		if self.padding:
			x = _pad_3D_image_patches_with_channel(x, [x.shape[0], x.shape[1], 158, 189, 156])
		x = self.conv1(x)
		if self.padding:
			x = _pad_3D_image_patches_with_channel(x, [x.shape[0], x.shape[1], 64, 64, 64])
		return self.conv3(self.conv2(x)).reshape(x.shape[0], -1, 256, 256)

	def _get_name(self):
		return "3D Conv Layers"

class PatchEmbedding(nn.Module):
	def __init__(self,
				 in_channels: int = 8,
				 patch_size: int = 16,
				 embedding_dim: int = 2048):
		super().__init__()
		self.patcher = nn.Conv2d(in_channels = in_channels,
								 out_channels = embedding_dim,
								 kernel_size = patch_size,
								 stride = patch_size,
								 padding = 0)

		self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)

		self.patch_size = patch_size

	def forward(self, x):
		assert x.shape[-1] % self.patch_size == 0, f"[ERROR] Input Resolution Must be Divisible " \
												   f"by Patch Size. \nImage Shape: " \
												   f"{x.shape[-1]}\nPatch Size: {self.patch_size}"

		x = self.patcher(x)
		x = self.flatten(x)
		return x.permute(0, 2, 1)


class PositionEmbedding(nn.Module):
	def __init__(self,
				 num_patches: int = 256,
				 embedding_dimension: int = 2048):
		super().__init__()
		self.position_matrix = nn.Parameter(torch.randn(1, num_patches,
													   embedding_dimension),
									   requires_grad = True)

	def forward(self, x):
		return x + self.position_matrix

class MultiheadSelfAttention(nn.Module):
	def __init__(self,
				 embedding_dim: int = 2048,
				 num_heads: int = 16,
				 attention_dropout: float = .0):
		super().__init__()
		self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)
		self.multihead_attention = nn.MultiheadAttention(embed_dim = embedding_dim,
														 num_heads = num_heads,
														 dropout = attention_dropout,
														 batch_first = True)

	def forward(self, x):
		x = self.layer_norm(x)
		attention_output, _ = self.multihead_attention(query = x,
													   key = x,
													   value = x,
													   need_weights = False)
		return attention_output

class MultiLayerPerception(nn.Module):
	def __init__(self,
				 embedding_dim: int = 2048,
				 mlp_size: int = 4096,
				 mlp_dropout: float = 0.1):
		super().__init__()

		self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)
		self.mlp = nn.Sequential(
				nn.Linear(in_features = embedding_dim, out_features = mlp_size),
				nn.GELU(),
				nn.Dropout(p = mlp_dropout),
				nn.Linear(in_features = mlp_size, out_features = embedding_dim),
				nn.Dropout(p = mlp_dropout)
				)

	def forward(self, x):
		return self.mlp(self.layer_norm(x))

class TransformerEncoder(nn.Module):
	def __init__(self,
				 embedding_dim: int = 2048,
				 num_heads: int = 16,
				 mlp_size: int = 4096,
				 mlp_dropout: float = .1,
				 attention_dropout: float = .0):
		super().__init__()
		self.msa_block = MultiheadSelfAttention(embedding_dim = embedding_dim,
												num_heads =  num_heads,
												attention_dropout = attention_dropout)
		self.mlp_block = MultiLayerPerception(embedding_dim = embedding_dim,
											  mlp_size = mlp_size,
											  mlp_dropout = mlp_dropout)

	def forward(self, x):
		# Create residual connection for MSA block
		x = self.msa_block(x) + x
		# Create residual connection for MLP block
		x = self.mlp_block(x) + x
		return x

def _pad_3D_image_patches_with_channel(img, desired_size):
	# Make sure your image tensor has 5 dimensions: (batch_size, channel, dim1, dim2, dim3)
	# If channel is missing add a dimension for it
	if len(img.shape) == 4:
		img = img.unsqueeze(1)  # Add the channel dimension

	# The difference between the desired size and the original size
	diff = [desired_size[i] - img.size(i) for i in
			range(-3, 0)]  # -3, -2, -1 correspond to dim1, dim2, dim3
	diff = diff[::-1]
	# Prepend with zeros for batch and channel dimensions
	padding = ()
	for i in diff:
		if i % 2 == 0:
			padding += (i // 2, i // 2)
		else:
			padding += (i // 2, i // 2 + 1)

	padding += (0, 0, 0, 0)

	print("Padding:", padding)
	padded = F.pad(img, padding)

	return padded





