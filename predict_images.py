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
from utils.models import AutoEncoder
from utils.data_loader import image_dataset

import sys

import nibabel as nib


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str)
parser.add_argument('--image_file', type=str)

parser.add_argument('--latent_dim', type=int, default=2048)

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
	os.makedirs(args.output_file + '/images')
except:
	pass

paths = pd.read_csv(args.image_file)['PATH']


############################################ <Model> ############################################
latent_dim = args.latent_dim

model = AutoEncoder(final_latent_space_dim=latent_dim)
model.load_state_dict(torch.load(args.output_file + '/models/model'))
model.eval()
model.to(device)



up_matrices = []
down_matrices = []

for k in range(2):
		up_matrices.append(torch.load(args.output_file + '/models/up_matrix_%d' %(k)).to(device))
		down_matrices.append(torch.load(args.output_file + '/models/down_matrix_%d' %(k)).to(device))

############################################ </Model> ############################################






with torch.no_grad():

	for p in tqdm(paths, total=len(paths)):

		img_name = p.split('/')[-1]
		
		image = nib.load(p).get_fdata()
		image[np.isnan(image)] = np.mean(image[~np.isnan(image)])

		div = max(np.max(image), -np.min(image))
		image /= div

		image = torch.unsqueeze(torch.FloatTensor(image), 0).to(device)


		out_img, _ = model(image)
		out_img = div*torch.squeeze(out_img).cpu().detach().numpy()


		nib_img = nib.Nifti1Image(out_img, np.eye(4))


		

		nib_img.to_filename(args.output_file + '/images/' + img_name)










		
		