
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


















class Encoder(nn.Module):
	
	def __init__(self, encoded_space_dim):
		super().__init__()
		
		### Convolutional section
		self.encoder_cnn = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), 
										 nn.ReLU(True), 
										 nn.Conv2d(16, 32, 3, stride=2, padding=1), 
										 nn.BatchNorm2d(32), 
										 nn.ReLU(True), 
										 nn.Conv2d(32, 64, 3, stride=2, padding=0), 
										 nn.ReLU(True))
		
		### Flatten layer
		self.flatten = nn.Flatten(start_dim=1)

		### Linear section
		self.encoder_lin = nn.Sequential(nn.Linear(3 * 3 * 64, 128), 
										 nn.ReLU(True), 
										 nn.Linear(128, encoded_space_dim))
		
	def forward(self, x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		x = self.encoder_lin(x)
		return x

class Decoder(nn.Module):
	
	def __init__(self, encoded_space_dim):
		super().__init__()
		self.decoder_lin = nn.Sequential(nn.Linear(encoded_space_dim, 128),
										 nn.ReLU(True), 
										 nn.Linear(128, 3 * 3 * 64), 
										 nn.ReLU(True))

		self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 3, 3))

		self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0), 
										  nn.BatchNorm2d(32), 
										  nn.ReLU(True), 
										  nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
										  nn.BatchNorm2d(16), 
										  nn.ReLU(True), 
										  nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1))
		
	def forward(self, x):
		x = self.decoder_lin(x)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		x = torch.sigmoid(x)
		return x


class Encoder32(nn.Module):

	def __init__(self, num_input_channels=3, base_channel_size=32, latent_dim=256):
		

		super().__init__()
		c_hid = base_channel_size
		self.net = nn.Sequential(nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
								 nn.GELU(),
								 nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
								 nn.GELU(),
								 nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
								 nn.GELU(),
								 nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
								 nn.GELU(),
								 nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
								 nn.GELU(),
								 nn.Flatten(), # Image grid to single feature vector
								 nn.Linear(2*16*c_hid, latent_dim))

	def forward(self, x):
		return self.net(x)

class Decoder32(nn.Module):

	def __init__(self, num_input_channels=3, base_channel_size=32, latent_dim =256):
		

		super().__init__()

		c_hid = base_channel_size

		self.linear = nn.Sequential(nn.Linear(latent_dim, 2*16*c_hid),
									nn.GELU())

		self.net = nn.Sequential(nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
								 nn.GELU(),
								 nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
								 nn.GELU(),
								 nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
								 nn.GELU(),
								 nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
								 nn.GELU(),
								 nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
								 nn.Tanh()) # The input images is scaled between -1 and 1, hence the output has to be bounded as well
		

	def forward(self, x):
		x = self.linear(x)
		x = x.reshape(x.shape[0], -1, 4, 4)
		x = self.net(x)
		return x







class Encoder128(nn.Module):

	def __init__(self, num_input_channels=3, base_channel_size=128, latent_dim=256):
		

		super().__init__()
		c_hid = base_channel_size
		self.net = nn.Sequential(nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 128x128 => 64x64
								 nn.BatchNorm2d(c_hid),
								 nn.GELU(),
								 nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 64x64 => 32x32
								 nn.BatchNorm2d(2*c_hid),
								 nn.GELU(),
								 nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
								 nn.BatchNorm2d(4*c_hid),
								 nn.GELU(),
								 nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
								 nn.BatchNorm2d(4*c_hid),
								 nn.GELU(),
								 nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
								 nn.BatchNorm2d(4*c_hid),
								 nn.Flatten(), # Image grid to single feature vector
								 nn.Linear(4*16*c_hid, latent_dim))

	def forward(self, x):
		return self.net(x)

class Decoder128(nn.Module):

	def __init__(self, num_input_channels=3, base_channel_size=128, latent_dim =256):
		

		super().__init__()

		c_hid = base_channel_size

		self.linear = nn.Sequential(nn.Linear(latent_dim, 4*16*c_hid),
									nn.GELU())

		self.net = nn.Sequential(nn.ConvTranspose2d(4*c_hid, 4*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
								 nn.GELU(),
								 nn.ConvTranspose2d(4*c_hid, 4*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
								 nn.GELU(),
								 nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
								 nn.GELU(),
								 nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 32x32 => 64x64
								 nn.GELU(),
								 nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 64x64 => 128x128
								 nn.Tanh()) # The input images is scaled between -1 and 1, hence the output has to be bounded as well
		

	def forward(self, x):
		x = self.linear(x)
		x = x.reshape(x.shape[0], -1, 4, 4)
		x = self.net(x)
		return x





class VAEncoder(nn.Module):
	
	def __init__(self, encoded_space_dim, sigma=1):
		super().__init__()

		self.encoded_space_dim = encoded_space_dim
		self.sigma = sigma

		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		
		### Convolutional section
		self.encoder_cnn = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), 
										 nn.ReLU(True), 
										 nn.Conv2d(16, 32, 3, stride=2, padding=1), 
										 nn.BatchNorm2d(32), 
										 nn.ReLU(True), 
										 nn.Conv2d(32, 64, 3, stride=2, padding=0), 
										 nn.ReLU(True))
		
		### Flatten layer
		self.flatten = nn.Flatten(start_dim=1)

		### Linear section
		self.encoder_lin = nn.Sequential(nn.Linear(3 * 3 * 64, 128), 
										 nn.ReLU(True), 
										 nn.Linear(128, encoded_space_dim))
		
	def forward(self, x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		x = self.encoder_lin(x)

		normal_gen = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.encoded_space_dim), torch.eye(self.encoded_space_dim))
		samples = x + self.sigma*normal_gen.sample((x.shape[0],)).to(self.device)


		return x, samples







class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=False):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits









class BinaryProductNN(nn.Module):

	def __init__(self, encoded_space_dim):
		super().__init__()

		self.encoded_space_dim = encoded_space_dim


		self.network = nn.Sequential(nn.Linear(2*self.encoded_space_dim, self.encoded_space_dim))
									 # nn.ReLU(True),
									 # nn.Linear(self.encoded_space_dim, self.encoded_space_dim),
									 # nn.ReLU(True),
									 # nn.Linear(self.encoded_space_dim, self.encoded_space_dim))


	def forward(self, x):
		x = self.network(x)

		return x













class ResNet(nn.Module):
	def __init__(self, in_depth, n_blocks, interm_depths, bottleneck=True, n_out_linear=None, dropout=0.):
		super(ResNet, self).__init__()
		self.name = 'Resnet'
		
		assert(len(n_blocks) == len(interm_depths))
		
		self.init_conv = nn.Conv3d(in_depth, interm_depths[0], kernel_size=7, stride=2, padding=3, bias=True)
		self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
		self.stages = nn.ModuleList([self._build_stage(n_blocks[i], interm_depths[max(0, i - 1)],
													   out_depth=interm_depths[i], stride=1 if i == 0 else 2,
													   bottleneck=bottleneck) for i in range(len(n_blocks))])

		self.pool_linear = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.flatten = nn.Flatten()

		if n_out_linear is not None:
			self.output_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], n_out_linear, dropout=dropout)
		else:
			self.output_head = None
		
		for m in self.modules():
			if type(m) in (nn.Conv3d, nn.Linear):
				nn.init.kaiming_normal_(m.weight,
										mode='fan_out',
										nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0.)
			if type(m) is nn.BatchNorm3d:
				nn.init.constant_(m.weight, 1.)
				nn.init.constant_(m.bias, 0.)

	def encode(self, x):
		_, _, latent, _ = self.forward(x, None)
		return latent
	
	def forward(self, x):
		x = self.init_conv(x)
		x = self.init_pool(x)
		for s in self.stages:
			x = s(x)
		x = self.pool_linear(x)
		latent = self.flatten(x)
		if self.output_head is not None:
			x = self.output_head(latent)

		return None, x, latent, None

	def _build_stage(self, n_blocks, in_depth, hidden_depth=None, out_depth=None, stride=2, dilation=1,
					 batchnorm=True, zero_output=True, bottleneck=True):
		if out_depth is None:
			out_depth = in_depth * stride
		blocks = [ResNetBlock(in_depth if i == 0 else out_depth, hidden_depth, out_depth, 
								stride=stride if i == 0 else 1,
								dilation=dilation, 
								batchnorm=batchnorm,  
								zero_output=zero_output, 
								bottleneck=bottleneck) for i in range(n_blocks)]
		return nn.Sequential(*blocks)

	def _build_linear_head(self, in_depth, hidden_depths, out_depth,
						   batchnorm=True, dropout=0.):
		layers = []
		depths = [in_depth, *hidden_depths, out_depth]

		layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
		layers.append(nn.Flatten())
		for i in range(len(depths) - 1):
			if dropout > 0.:
				layers.append(nn.Dropout(p=dropout))
			layers.append(nn.Linear(depths[i], depths[i + 1], bias=not batchnorm))
			if i != len(depths) - 2:
				if batchnorm:
					layers.append(nn.BatchNorm1d(depths[i + 1], eps=1e-8))
				layers.append(nn.GELU())

		return nn.Sequential(*layers)


class MLP(nn.Module):
	def __init__(self, in_depth, hidden_depths, out_depth, batchnorm=True, dropout=0., binary=True):
		super(MLP, self).__init__()
		self.batchnorm = batchnorm
		self.dropout = dropout
		self.depths = [in_depth, *hidden_depths, out_depth]

		self.linear_layers = nn.ModuleList([])
		self.norm = nn.ModuleList([])
		self.act = nn.ModuleList([])

		self.binary = binary

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

		if (self.binary):
			x = torch.sigmoid(x)
		else:
			x = x
			# x = torch.softmax(x, dim=1)

		return x
		
	def l1_norm(self):
		return sum([torch.norm(l.weight, 1) for l in self.linear_layers])
	
	def l2_norm(self):
		return sum([torch.norm(l.weight, 2) for l in self.linear_layers])


class ResNetBlock(nn.Module):
	def __init__(self, in_depth, hidden_depth=None, out_depth=None, stride=1, dilation=1,
				 batchnorm=True, zero_output=True, bottleneck=True):
		super(ResNetBlock, self).__init__()
		if out_depth is None:
			out_depth = in_depth * stride
		if stride > 1:
			self.shortcut_layer = nn.Conv3d(in_depth, out_depth, kernel_size=3, stride=stride,
											padding=1, dilation=dilation, bias=True)
		else:
			self.shortcut_layer = None

		layers = []
		if bottleneck:
			if hidden_depth is None:
				hidden_depth = in_depth // 4
			k_sizes = [3, 1, 3]
			depths = [in_depth, hidden_depth, hidden_depth, out_depth]
			paddings = [1, 0, 1]
			strides = [1, 1, stride]
			dilations = [dilation, 1, dilation]
		else:
			if hidden_depth is None:
				hidden_depth = in_depth
			k_sizes = [3, 3]
			depths = [in_depth, hidden_depth, out_depth]
			paddings = [1, 1]
			strides = [1, stride]
			dilations = [dilation, dilation]
		
		for i in range(len(k_sizes)):
			if batchnorm:
				layers.append(nn.BatchNorm3d(depths[i], eps=1e-8))
			layers.append(nn.GELU())
			layers.append(nn.Conv3d(depths[i], depths[i + 1], k_sizes[i], padding=paddings[i],
									stride=strides[i], dilation=dilations[i], bias=False))
		
		self.layers = nn.Sequential(*layers)
		# for i, l in enumerate(self.layers):
		#	 self.add_module("layer_{:d}".format(i), l)
		
	def forward(self, x):
		Fx = self.layers(x)
		if self.shortcut_layer is not None:
			x = self.shortcut_layer(x)
		return x + Fx





class ResNet_encdec(nn.Module):
	def __init__(self, in_depth=1, n_blocks=1, interm_depths=1, bottleneck=True, n_out_linear=None, dropout=0., latent_dim=1024):
		super(ResNet_encdec, self).__init__()
		self.name = 'ResNet_encdec'
		self.latent_dim = latent_dim

		self.adniConvInit()
		self.adniDeconvInit()
		self.flatten = nn.Flatten()
		
		if n_out_linear is not None:
			self.output_head = MLP(self.latent_dim, [64], n_out_linear, dropout=dropout)
		else:
			self.output_head = None

	def encode(self, inp):
		hid = self.convA1(inp)
		# hid = self.batch1(hid)
		hid = self.swishA1(hid)
		#print('after first conv', hid.shape)

		self.size1 = hid.size()
		hid, self.indices1 = self.maxpoolA1(hid)
		hid = self.convA2(hid)
		# hid = self.batch2(hid)
		hid = self.swishA2(hid)
		#print('after second conv', hid.shape)
		
		self.size2 = hid.size()
		hid, self.indices2 = self.maxpoolA2(hid)
		hid = self.convA3(hid)
		# hid = self.batch3(hid)
		hid = self.swishA3(hid)
		#print('after third conv', hid.shape)

		self.size3 = hid.size()
		hid, self.indices3 = self.maxpoolA3(hid)
		hid = self.convA4(hid)
		# hid = self.batch4(hid)
		hid = self.swishA4(hid)
		#print('after fourth conv', hid.shape)
		  
		self.size4 = hid.size()
		hid, self.indices4 = self.maxpoolA4(hid)
		hid = self.convA5(hid)
		# hid = self.batch5(hid)
		hid = self.swishA5(hid)
		#print('after fifth conv', hid.shape)

		self.size5 = hid.size()
		hid, self.indices5 = self.maxpoolA5(hid)
		hid = self.convA6(hid)
		# hid = self.batch6(hid)
		hid = self.swishA6(hid)
		#print('after sixth conv', hid.shape)

		self.size6 = hid.size()
		hid, self.indices6 = self.maxpoolA6(hid)
		hid = self.convA7(hid)
		# hid = self.batch7(hid)
		hid = self.swishA7(hid)
		#print('after seventh conv', hid.shape)
		hid = self.flatten(hid)
		return hid

	def decode(self, hid):
		hid = hid.view(-1, 512, 1, 2, 1)
		
		#print('hidden shape', hid.shape)
		
		out = self.deconvB7(hid)
		out = self.swishB7(out)
		#print('before first unpool', out.shape)

		out = self.maxunpoolB6(out, self.indices6, self.size6)
		out = self.deconvB6(out)
		out = self.swishB6(out)
		
		#print('before second unpool', out.shape)
		out = self.maxunpoolB5(out, self.indices5, self.size5)
		out = self.deconvB5(out)
		out = self.swishB5(out)
		
		#print('before third unpool', out.shape)
		out = self.maxunpoolB4(out, self.indices4, self.size4)
		out = self.deconvB4(out)
		out = self.swishB4(out)
		
		#print('before fourth unpool', out.shape)
		out = self.maxunpoolB3(out, self.indices3, self.size3)
		out = self.deconvB3(out)
		out = self.swishB3(out)
		
		#print('before fifth unpool', out.shape)
		out = self.maxunpoolB2(out, self.indices2, self.size2)
		out = self.deconvB2(out)
		out = self.swishB2(out)
		
		#print('before sixth unpool', out.shape)
		out = self.maxunpoolB1(out, self.indices1, self.size1)
		out = self.deconvB1(out)
		out = self.swishB1(out)

		#print('out shape', out.shape)
		return out

	def adniConvInit(self):
		#Convolution 1
		self.convA1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
		nn.init.xavier_uniform_(self.convA1.weight) #Xaviers Initialisation
		# self.batch1 = nn.BatchNorm3d(8, eps=1e-8)
		self.swishA1 = nn.GELU()

		#Max Pool 1
		self.maxpoolA1 = nn.MaxPool3d(kernel_size=2, return_indices=True)

		#Convolution 2
		self.convA2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=2, bias=False)
		nn.init.xavier_uniform_(self.convA2.weight)
		# self.batch2 = nn.BatchNorm3d(16, eps=1e-8)
		self.swishA2 = nn.GELU()

		#Max Pool 2
		self.maxpoolA2 = nn.MaxPool3d(kernel_size=2, return_indices=True)

		#Convolution 3
		self.convA3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
		nn.init.xavier_uniform_(self.convA3.weight)
		# self.batch3 = nn.BatchNorm3d(32, eps=1e-8)
		self.swishA3 = nn.GELU()

		#Max Pool 3
		self.maxpoolA3 = nn.MaxPool3d(kernel_size=2, return_indices=True)

		#Convolution 4
		self.convA4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=2, bias=False)
		nn.init.xavier_uniform_(self.convA4.weight)
		# self.batch4 = nn.BatchNorm3d(64, eps=1e-8)
		self.swishA4 = nn.GELU()

		#Max Pool 4
		self.maxpoolA4 = nn.MaxPool3d(kernel_size=2, return_indices=True)

		#Convolution 5
		self.convA5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
		nn.init.xavier_uniform_(self.convA5.weight)
		# self.batch5 = nn.BatchNorm3d(128, eps=1e-8)
		self.swishA5 = nn.GELU()

		#Max Pool 5
		self.maxpoolA5 = nn.MaxPool3d(kernel_size=2, return_indices=True)

		#Convolution 6
		self.convA6 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
		nn.init.xavier_uniform_(self.convA6.weight)
		# self.batch6 = nn.BatchNorm3d(256, eps=1e-8)
		self.swishA6 = nn.GELU()

		#Max Pool 6
		self.maxpoolA6 = nn.MaxPool3d(kernel_size=2, return_indices=True)

		#Convolution 7
		self.convA7 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
		nn.init.xavier_uniform_(self.convA7.weight)
		# self.batch7 = nn.BatchNorm3d(512, eps=1e-8)
		self.swishA7 = nn.GELU()

	def adniDeconvInit(self):
		#De Convolution 7
		self.deconvB7 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
		nn.init.xavier_uniform_(self.deconvB7.weight)
		self.swishB7 = nn.GELU()

		#Max UnPool 6
		self.maxunpoolB6 = nn.MaxUnpool3d(kernel_size=2)

		#De Convolution 6
		self.deconvB6 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
		nn.init.xavier_uniform_(self.deconvB6.weight)
		self.swishB6 = nn.GELU()

		#Max UnPool 5
		self.maxunpoolB5 = nn.MaxUnpool3d(kernel_size=2)

		#De Convolution 5
		self.deconvB5 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
		nn.init.xavier_uniform_(self.deconvB5.weight)
		self.swishB5 = nn.GELU()

		#Max UnPool 4
		self.maxunpoolB4 = nn.MaxUnpool3d(kernel_size=2)

		#De Convolution 4
		self.deconvB4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, padding=2)
		nn.init.xavier_uniform_(self.deconvB4.weight)
		self.swishB4 = nn.GELU()

		#Max UnPool 3
		self.maxunpoolB3 = nn.MaxUnpool3d(kernel_size=2)

		#De Convolution 3
		self.deconvB3 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
		nn.init.xavier_uniform_(self.deconvB3.weight)
		self.swishB3 = nn.GELU()

		#Max UnPool 2
		self.maxunpoolB2 = nn.MaxUnpool3d(kernel_size=2)

		#De Convolution 2
		self.deconvB2 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, padding=2)
		nn.init.xavier_uniform_(self.deconvB2.weight)
		self.swishB2 = nn.GELU()

		#Max UnPool 1
		self.maxunpoolB1 = nn.MaxUnpool3d(kernel_size=2)

		#DeConvolution 1
		self.deconvB1 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
		nn.init.xavier_uniform_(self.deconvB1.weight)
		self.swishB1 = nn.GELU()

	
	def forward(self, x):
		
		latent = self.encode(x)
		
		if self.output_head is not None:
			pred = self.output_head(latent)
		else:
			pred = None
			
		recons = self.decode(latent)
			
		return recons, pred, latent 







class TauResNet(nn.Module):
	def __init__(self, in_depth, n_blocks, interm_depths, bottleneck=True, n_out_linear=None, dropout=0., const=0.01):
		super(TauResNet, self).__init__()
		print('Using TauResNet!!!')
		
		self.name = 'TauResNet'
		self.const = const
		assert(len(n_blocks) == len(interm_depths))
		self.init_conv = nn.Conv3d(in_depth, interm_depths[0], kernel_size=7, stride=2, padding=3, bias=True)
		self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
		self.stages = nn.ModuleList([self._build_stage(n_blocks[i], interm_depths[max(0, i - 1)],
													   out_depth=interm_depths[i], stride=1 if i == 0 else 2,
													   bottleneck=bottleneck) for i in range(len(n_blocks))])

		self.pool_linear = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.flatten = nn.Flatten()

		#Tau head Layers
		self.tau_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], interm_depths[-1], dropout=dropout)
		self.tanh = get_activation_layer('tanh')()
		
		# y-predictor
		if n_out_linear is not None:
			self.output_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], n_out_linear, dropout=dropout)
		else:
			self.output_head = None

			
		
		for m in self.modules():
			if type(m) in (nn.Conv3d, nn.Linear):
				nn.init.kaiming_normal_(m.weight,
										mode='fan_out',
										nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0.)
			if type(m) is nn.BatchNorm3d:
				nn.init.constant_(m.weight, 1.)
				nn.init.constant_(m.bias, 0.)

	def encode(self, x):
		x = self.init_conv(x)
		x = self.init_pool(x)
		for s in self.stages:
			x = s(x)
		x = self.pool_linear(x)
		x = self.flatten(x)
		return x
		
	def encodeTauNet(self, x, gval):
		x = self.encode(x)
		li_latent = self.project_sphere(x)
		gi_latent = self.TauNet(li_latent, gval)
		return gi_latent

	def encodeProject(self, x):
		x = self.encode(x)
		li_latent = self.project_sphere(x)
		return li_latent

	def TauNet(self, li_latent, gval):
		fct = self.tau_head(li_latent)
		gi_latent = gval + self.const*self.tanh(fct)
		return gi_latent

	def project_sphere(self, mu):
		mu_centered = (mu - torch.mean(mu, dim=1, keepdim=True))
		mu_norm = torch.norm(mu_centered, dim=1, keepdim=True)
		return mu_centered / mu_norm
	
	def forward(self, x, gval):
		x = self.encode(x)
		
		li_latent = self.project_sphere(x)
		gi_latent = self.TauNet(li_latent, gval)
		
		if self.output_head is not None:
			pred = self.output_head(li_latent)

		return None, pred, gi_latent, None

	def _build_stage(self, n_blocks, in_depth, hidden_depth=None, out_depth=None, stride=2, dilation=1,
					 batchnorm=True, activation='swish', zero_output=True, bottleneck=True):
		if out_depth is None:
			out_depth = in_depth * stride
		blocks = [ResNetBlock(in_depth if i == 0 else out_depth, hidden_depth, out_depth, 
								stride=stride if i == 0 else 1,
								dilation=dilation, 
								batchnorm=batchnorm, 
								activation=activation, 
								zero_output=zero_output, 
								bottleneck=bottleneck) for i in range(n_blocks)]
		return nn.Sequential(*blocks)

	def _build_linear_head(self, in_depth, hidden_depths, out_depth, activation='swish',
						   batchnorm=True, dropout=0.):
		layers = []
		depths = [in_depth, *hidden_depths, out_depth]

		layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
		layers.append(nn.Flatten())
		for i in range(len(depths) - 1):
			if dropout > 0.:
				layers.append(nn.Dropout(p=dropout))
			layers.append(nn.Linear(depths[i], depths[i + 1], bias=not batchnorm))
			if i != len(depths) - 2:
				if batchnorm:
					layers.append(nn.BatchNorm1d(depths[i + 1], eps=1e-8))
				layers.append(get_activation_layer(activation)())

		return nn.Sequential(*layers)


class CelebA_encoder(nn.Module):


	def __init__(self, encoded_space_dim):
		super().__init__()
		
		### Convolutional section
		self.encoder_cnn = nn.Sequential(nn.Conv2d(3, 32, 4, stride=2, padding=0), 
										 nn.BatchNorm2d(32),
										 nn.GELU(), 
										 nn.Conv2d(32, 64, 4, stride=2, padding=0), 
										 nn.BatchNorm2d(64), 
										 nn.GELU(), 
										 nn.Conv2d(64, 128, 4, stride=2, padding=0), 
										 nn.BatchNorm2d(128), 
										 nn.GELU(),
										 nn.Conv2d(128, 256, 4, stride=2, padding=0), 
										 nn.BatchNorm2d(256), 
										 nn.GELU())
		
		### Flatten layer
		self.flatten = nn.Flatten(start_dim=1)

		### Linear section
		self.encoder_lin = nn.Sequential(nn.Linear(2 * 2 * 256, 512), 
										 nn.GELU(), 
										 nn.Linear(512, encoded_space_dim),
										 nn.GELU())
		
	def forward(self, x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		x = self.encoder_lin(x)
		return x





class CelebA_decoder(nn.Module):

	def __init__(self, encoded_space_dim):
		super().__init__()
		self.decoder_lin = nn.Sequential(nn.Linear(encoded_space_dim, 512),
										 nn.GELU(), 
										 nn.Linear(512, 2 * 2 * 256), 
										 nn.GELU())

		self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 2, 2))

		self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, output_padding=0), 
										  nn.BatchNorm2d(128), 
										  nn.GELU(), 
										  nn.ConvTranspose2d(128, 64, 4, stride=2, padding=0, output_padding=0), 
										  nn.BatchNorm2d(64), 
										  nn.GELU(), 
										  nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0, output_padding=1), 
										  nn.BatchNorm2d(32), 
										  nn.GELU(), 
										  nn.ConvTranspose2d(32, 3, 4, stride=2, padding=0, output_padding=0))
		
	def forward(self, x):
		x = self.decoder_lin(x)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		return x


class AutoEncoder(nn.Module):
	def __init__(self, in_channels, dec_channels, latent_size):
		super(AutoEncoder, self).__init__()
		
		self.in_channels = in_channels
		self.dec_channels = dec_channels
		self.latent_size = latent_size

		###############
		# ENCODER
		##############
		self.e_conv_1 = nn.Conv2d(in_channels, dec_channels, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_1 = nn.BatchNorm2d(dec_channels)

		self.e_conv_2 = nn.Conv2d(dec_channels, dec_channels*2, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_2 = nn.BatchNorm2d(dec_channels*2)

		self.e_conv_3 = nn.Conv2d(dec_channels*2, dec_channels*4, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_3 = nn.BatchNorm2d(dec_channels*4)

		self.e_conv_4 = nn.Conv2d(dec_channels*4, dec_channels*8, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_4 = nn.BatchNorm2d(dec_channels*8)

		# self.e_conv_5 = nn.Conv2d(dec_channels*8, dec_channels*16, 
		# 						  kernel_size=(4, 4), stride=(2, 2), padding=1)
		# self.e_bn_5 = nn.BatchNorm2d(dec_channels*16)
	   
		self.e_fc_1 = nn.Linear(dec_channels*8*8*8, latent_size)

		###############
		# DECODER
		##############
		
		self.d_fc_1 = nn.Linear(latent_size, dec_channels*8*8*8)

		# self.d_conv_1 = nn.Conv2d(dec_channels*16, dec_channels*8, 
		# 						  kernel_size=(4, 4), stride=(1, 1), padding=0)
		# self.d_bn_1 = nn.BatchNorm2d(dec_channels*8)

		self.d_conv_2 = nn.Conv2d(dec_channels*8, dec_channels*4, 
								  kernel_size=(4, 4), stride=(1, 1), padding=0)
		self.d_bn_2 = nn.BatchNorm2d(dec_channels*4)

		self.d_conv_3 = nn.Conv2d(dec_channels*4, dec_channels*2, 
								  kernel_size=(4, 4), stride=(1, 1), padding=0)
		self.d_bn_3 = nn.BatchNorm2d(dec_channels*2)

		self.d_conv_4 = nn.Conv2d(dec_channels*2, dec_channels, 
								  kernel_size=(4, 4), stride=(1, 1), padding=0)
		self.d_bn_4 = nn.BatchNorm2d(dec_channels)
		
		self.d_conv_5 = nn.Conv2d(dec_channels, in_channels, 
								  kernel_size=(4, 4), stride=(1, 1), padding=0)
		
		
		# Reinitialize weights using He initialization
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.detach())
				m.bias.detach().zero_()
			elif isinstance(m, torch.nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight.detach())
				m.bias.detach().zero_()
			elif isinstance(m, torch.nn.Linear):
				nn.init.kaiming_normal_(m.weight.detach())
				m.bias.detach().zero_()


	def encode(self, x):
		
		#h1
		x = self.e_conv_1(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
		x = self.e_bn_1(x)
		
		#h2
		x = self.e_conv_2(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)	
		x = self.e_bn_2(x)	 

		#h3
		x = self.e_conv_3(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		x = self.e_bn_3(x)
		
		#h4
		x = self.e_conv_4(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		x = self.e_bn_4(x)
		
		#h5
		# x = self.e_conv_5(x)
		# x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		# x = self.e_bn_5(x)		
		
		
		#fc
		x = x.view(-1, self.dec_channels*8*8*8)

		x = self.e_fc_1(x)
		return x

	def decode(self, x):
		
		# h1
		#x = x.view(-1, self.latent_size, 1, 1)
		x = self.d_fc_1(x)
		
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)  
		x = x.view(-1, self.dec_channels*8, 8, 8) 

		

		# h2
		# x = F.interpolate(x, scale_factor=2)
		# x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
		# x = self.d_conv_1(x)
		# x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
		# x = self.d_bn_1(x)
		
		# h3
		x = F.interpolate(x, scale_factor=2)
		x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
		x = self.d_conv_2(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
		x = self.d_bn_2(x)
		
		# h4
		x = F.interpolate(x, scale_factor=2)
		x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
		x = self.d_conv_3(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
		x = self.d_bn_3(x)  

		# h5
		x = F.interpolate(x, scale_factor=2)
		x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
		x = self.d_conv_4(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
		x = self.d_bn_4(x)
		
		
		# out
		x = F.interpolate(x, scale_factor=2)
		x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
		x = self.d_conv_5(x)
		# x = torch.sigmoid(x)
		
		return x

	def forward(self, x):
		z = self.encode(x)
		decoded = self.decode(z)
		return z, decoded





class Discriminator(nn.Module):
	def __init__(self, in_channels=3, dec_channels=32, latent_size=512):
		super(Discriminator, self).__init__()
		
		self.in_channels = in_channels
		self.dec_channels = dec_channels
		self.latent_size = latent_size

		self.e_conv_1 = nn.Conv2d(in_channels, dec_channels, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_1 = nn.BatchNorm2d(dec_channels)

		self.e_conv_2 = nn.Conv2d(dec_channels, dec_channels*2, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_2 = nn.BatchNorm2d(dec_channels*2)

		self.e_conv_3 = nn.Conv2d(dec_channels*2, dec_channels*4, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_3 = nn.BatchNorm2d(dec_channels*4)

		self.e_conv_4 = nn.Conv2d(dec_channels*4, dec_channels*8, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_4 = nn.BatchNorm2d(dec_channels*8)

		self.e_conv_5 = nn.Conv2d(dec_channels*8, dec_channels*16, 
								  kernel_size=(4, 4), stride=(2, 2), padding=1)
		self.e_bn_5 = nn.BatchNorm2d(dec_channels*16)
	   
		self.e_fc_1 = nn.Linear(dec_channels*16*4*4, latent_size)
		self.e_fc_2 = nn.Linear(latent_size, 1)


	def forward(self, x):
		
		#h1
		x = self.e_conv_1(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
		x = self.e_bn_1(x)
		
		#h2
		x = self.e_conv_2(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True)	
		x = self.e_bn_2(x)	 

		#h3
		x = self.e_conv_3(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		x = self.e_bn_3(x)
		
		#h4
		x = self.e_conv_4(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		x = self.e_bn_4(x)
		
		#h5
		x = self.e_conv_5(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		x = self.e_bn_5(x)		
		
		#fc
		x = x.view(-1, self.dec_channels*16*4*4)
		x = self.e_fc_1(x)
		x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
		x = self.e_fc_2(x)
		x = torch.sigmoid(x)

		return x

class Encoder(nn.Module):
	"""
		Input: [batch_size, channels, 157, 189, 156] recommended
		Output: [batch_size, final_latent_space_dim]

		User Guide:
		* conv_in_channels: the channels of the inputs
		conv_out_channels: the channels after consecutive 3D CNNs
					   64, then the sequence length / number of patches will be 1024
					   256, then the sequence length / number of patches will be 4096
		* kernel_size: the kernel size of 3D CNNs and Transpose 3D CNNs
		* padding: whether we need to pad the dimensions
		* batch_norm: whether we need to normalization
		* img_size: the flattened image size before the whole vit block
		* in_channels: the channels go into the patch embedding
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
		self.conv_block= Consecutive3DConvLayerBlock(in_channel = conv_in_channels,
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
	def forward(self, x):
		x, cache = self.conv_block(x)
		x, _ = self.vit_block(x)
		return x

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
				 in_channels: int = 1,
				 out_channels: int = 64,
				 patch_size: int = 16,
				 num_transformer_layer: int = 2,
				 embedding_dim: int = 256,
				 mlp_size: int = 2048,
				 num_heads: int = 16,
				 attention_dropout: float = .0,
				 mlp_dropout: float = .1,
				 embedding_dropout: float = .1,
				 final_latent_space_dim: int = 2048):
		super().__init__()

		self.num_patches = out_channels * patch_size ** 3 // patch_size ** 2

		self.position_embedding = PositionEmbedding(out_channel = out_channels,
													patch_size = patch_size,
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

		self.latent_space = nn.Sequential(
										  nn.LayerNorm(normalized_shape = embedding_dim),
										  nn.Flatten(),
										  nn.Linear(in_features = embedding_dim * self.num_patches,
													out_features = final_latent_space_dim),
										  nn.Dropout(p = mlp_dropout))

	def forward(self, x):
		patch_embeded = self.patch_embedding(x)
		x = self.position_embedding(patch_embeded)
		x = self.embedding_dropout(x)
		x = self.transformer_encoder(x)
		x = self.latent_space(x)
		return x, patch_embeded

	def _get_name(self):
		return "Vision Transformer Encoder"

class Consecutive3DConvLayerBlock(nn.Module):
	# Warning: if out_channel = 64, then sequence length will be 1024;
	#          if out_channel = 256, then sequence length will be 4096;
	def __init__(self,
				 in_channel: int = 1,
				 out_channel: int = 64,
				 kernel_size: int = 3,
				 patch_size: int = 16,
				 padding: bool = True,
				 batch_norm: bool = True) -> None:
		super().__init__()
		self.out_channel = out_channel
		self.padding = padding
		self.batch_norm = batch_norm
		self.patch_size = patch_size

		self.conv1 = nn.Conv3d(in_channels= in_channel,
							   out_channels = 32,
							   kernel_size = kernel_size,
							  stride = kernel_size)

		self.conv2 = nn.Conv3d(32, 64, (kernel_size, kernel_size, kernel_size), padding = 1, stride = 2)
		self.conv3 = nn.Conv3d(64, self.out_channel, (kernel_size, kernel_size, kernel_size), padding = 1,
							   stride = 2)
		if batch_norm:
			self.batch_norm_layer = nn.BatchNorm3d(out_channel)

	def forward(self, x):
		cache = {}
		self.batch_size, self.channels = x.shape[0], x.shape[1]
		cache["original_data"] = x
		if self.padding:
			x = _pad_3D_image_patches_with_channel(x, [x.shape[0], x.shape[1], 158, 189, 156])
			cache["first_padding"] = x
		x = self.conv1(x)
		cache["conv1"] = x
		if self.padding:
			x = _pad_3D_image_patches_with_channel(x, [x.shape[0], x.shape[1], 64, 64, 64])
			cache["second_padding"] = x
		x = self.conv2(x)
		cache["conv2"] = x

		x = self.conv3(x)
		cache["conv3"] = x

		if self.batch_norm:
			x = self.batch_norm_layer(x)

		out_shape = int(math.sqrt(self.out_channel * self.patch_size ** 3))

		return x.reshape(x.shape[0], 1, out_shape, -1), cache

	def _get_name(self):
		return "3D Conv Layers"

class PatchEmbedding(nn.Module):
	def __init__(self,
				 in_channels: int = 1,
				 patch_size: int = 16,
				 embedding_dim: int = 256):
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
				 out_channel: int = 64,
				 patch_size: int = 16,
				 embedding_dimension: int = 256):
		super().__init__()
		num_patches = out_channel * patch_size ** 3 // patch_size ** 2
		self.position_matrix = nn.Parameter(torch.randn(1, num_patches,
													   embedding_dimension), requires_grad = True)

	def forward(self, x):
		return x + self.position_matrix

class MultiheadSelfAttention(nn.Module):
	def __init__(self,
				 embedding_dim: int = 256,
				 num_heads: int = 8,
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
				 embedding_dim: int = 256,
				 mlp_size: int = 2048,
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
				 embedding_dim: int = 256,
				 num_heads: int = 8,
				 mlp_size: int = 2048,
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
	padded = F.pad(img, padding)

	return padded

class InitialResidualNet(nn.Module):
	def __init__(self,
				 final_latent_space_dim: int = 2048,
				 patch_size: int = 16,
				 embedding_dim: int = 256,
				 in_channels: int = 1,
				 out_channels: int = 64):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.num_patches = out_channels * patch_size ** 3 // patch_size ** 2
		sqrt_num_patches = int(math.sqrt(self.num_patches))
		self.patch_size = patch_size
		self.upsampling_fc_layer = nn.Linear(in_features = final_latent_space_dim,
											 out_features = embedding_dim * self.num_patches)
		self.unflatten = nn.Unflatten(dim = 2, unflattened_size = (sqrt_num_patches, sqrt_num_patches))
		self.unpatch_embeded = nn.ConvTranspose2d(in_channels = embedding_dim,
												  out_channels = in_channels,
												  kernel_size = patch_size,
												  stride = patch_size,
												  padding = 0)


	def forward(self, latent_space_vec, patch_and_pos_embedded):
		output = self.upsampling_fc_layer(latent_space_vec).reshape(latent_space_vec.shape[0],
																	self.num_patches,
																	self.embedding_dim)
		assert output.shape == patch_and_pos_embedded.shape, \
			f"[ERROR] Dimensions Mismatch: {output.shape} != {patch_and_pos_embedded}"

		output += patch_and_pos_embedded

		output = self.unflatten(output.permute(0, 2, 1))

		output = self.unpatch_embeded(output).reshape(output.shape[0], -1, self.patch_size,
													  self.patch_size, self.patch_size)

		return output


class DecodeConsecutiveConvNets(nn.Module):
	def __init__(self,
				 in_channel: int = 1,
				 out_channel: int = 64,
				 kernel_size: int = 3,
				 padding: bool = True):
		super().__init__()
		self.padding = padding
		self.unconv_from_conv3 = nn.ConvTranspose3d(in_channels = out_channel,
													out_channels = 64,
													kernel_size = 3,
													padding = 1,
													stride = 2,
													output_padding = 1
													)
		self.unconv_from_conv2 = nn.ConvTranspose3d(in_channels = 64,
													out_channels = 32,
													kernel_size = kernel_size,
													padding = 1,
													stride = 2,
													output_padding = 1)
		self.unconv_from_conv1 = nn.ConvTranspose3d(in_channels = 32,
													out_channels = in_channel,
													kernel_size = kernel_size,
													padding = 0,
													stride = kernel_size,
													dilation = (2, 1, 1)
													)


	def forward(self, x, cache_from_convnets):
		x += cache_from_convnets["conv3"]
		x = self.unconv_from_conv3(x) + cache_from_convnets["conv2"]
		x = self.unconv_from_conv2(x) + cache_from_convnets["second_padding"]
		if self.padding:
			x = _unpad_3D_image_patches_with_channel(x, (52, 63, 52)) + cache_from_convnets["conv1"]
		x = self.unconv_from_conv1(x) + cache_from_convnets["first_padding"]
		if self.padding:
			x = _unpad_3D_image_patches_with_channel(x, (157, 189, 156)) + cache_from_convnets[
			"original_data"]

		return x


def _unpad_3D_image_patches_with_channel(padded_img, original_size):
    # original_size should be a tuple: (original_dim1, original_dim2, original_dim3)
    slices = ()
    for dim_idx in range(-3, 0):  # -3, -2, -1 correspond to dim1, dim2, dim3
        diff = padded_img.size(dim_idx) - original_size[
            dim_idx + 3]  # calculate the difference in each dimension
        padding_before = diff // 2  # calculate padding before
        padding_after = diff - padding_before  # calculate padding after
        slices += slice(padding_before,
                        padding_before + original_size[dim_idx + 3]),  # slice to remove padding

    unpadded = padded_img[..., slices[0], slices[1], slices[2]]  # use ellipsis to keep the first two dimensions (batch and channel) unchanged
    return unpadded
