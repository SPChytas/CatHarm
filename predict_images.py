from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.parametrizations import orthogonal

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import random
import numpy as np 
import pandas as pd

from tqdm import tqdm
import os
import argparse

from utils import metrics
from utils.models import AutoEncoder, AutoEncoderWithoutShortcuts
from utils.data_loader import image_dataset

import sys

import nibabel as nib


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str)
parser.add_argument('--image_file', type=str)

parser.add_argument('--latent_dim', type=int, default=2048)

parser.add_argument('--skip_connections', action='store_true')

parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()




# System set-up
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(f'Selected device: {device}')




try:
	os.makedirs(args.output_file + '/predicted_images')
	os.makedirs(args.output_file + '/original_images')
except:
	pass

paths = pd.read_csv(args.image_file)['PATH']


############################################ <Model> ############################################
latent_dim = args.latent_dim

if (args.skip_connections):
	model = AutoEncoder(final_latent_space_dim=latent_dim, num_transformer_layer=4).to(device)
else:
	model = AutoEncoderWithoutShortcuts(final_latent_space_dim=latent_dim, attention_without_shortcuts=False, num_transformer_layer=4).to(device)

model.load_state_dict(torch.load(args.output_file + '/models/model'))
model.eval()
model.to(device)



up_matrices = []
down_matrices = []

for k in range(2):
	up_matrices.append(torch.load(args.output_file + '/models/up_matrix_%d' %(k)).to(device))
	down_matrices.append(torch.load(args.output_file + '/models/down_matrix_%d' %(k)).to(device))

############################################ </Model> ############################################



max_val = -float('inf')
min_val = float('inf')

for i in range(len(paths)):
		
	image = nib.load(paths[i]).get_fdata()
	min_val = np.nanmin(image)
	max_val = np.nanmax(image)


shift = min_val
scale = max_val - min_val


with torch.no_grad():

	for p in tqdm(paths, total=len(paths)):

		img_name = p.split('/')[-1]
		
		nib_image = nib.load(p)
		image = nib_image.get_fdata()
		image_affine = nib_image.affine

		print (image.shape)

		# shift = np.nanmin(image)
		# scale = np.nanmax(image) - np.nanmin(image)

		image = (image - shift)/scale
		image[np.isnan(image)] = 0

		image = torch.unsqueeze(torch.FloatTensor(image), 0).to(device)


		out_img, _ = model(image)
		out_img = scale*torch.squeeze(out_img).cpu().detach().numpy() + shift

		

		image = scale*image[0].cpu().detach().numpy() + shift
		

		# num_slices = out_img.shape[0]
		# for i in range(num_slices):
			
		# 	plt.subplot(1, 2, 1)
		# 	slice_data1 = image[i, :, :]
		# 	plt.imshow(np.rot90(slice_data1), cmap='gray')
		# 	plt.title(f"Image 1: Slice {i+1}/{num_slices}")
		# 	plt.axis('on')

		# 	plt.subplot(1, 2, 2)
		# 	slice_data2 = out_img[i, :, :]
		# 	plt.imshow(np.rot90(slice_data2), cmap='gray')
		# 	plt.title(f"Image 2: Slice {i+1}/{num_slices}")
		# 	plt.axis('on')

		# 	plt.show()






		nib_img = nib.Nifti1Image(image, affine=image_affine)
		nib_img.to_filename(args.output_file + '/original_images/' + img_name)

		nib_img = nib.Nifti1Image(out_img, affine=image_affine)
		nib_img.to_filename(args.output_file + '/predicted_images/' + img_name)










		
		