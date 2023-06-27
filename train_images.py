########################################## <Imports> ############################################
import zipfile
import requests
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
from utils.models import ResNet_encdec
from utils.data_loader import image_dataset
import sys
from pathlib import Path
from utils.models import *
import datetime
from torchinfo import summary
import math
########################################## <Config> ############################################
# Parse Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='data/imaging_data/(157, 189, 156)')
parser.add_argument('--output_file', type=str, default='experiments/imaging_data/(157, 189, 156)')
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--morph_loss_factor', type=float, default=1)
parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--invariant_variables', nargs='+', type=str, default=['mri_coil_name', 'scanner_source'])
parser.add_argument('--equivariant_variables', nargs='+', type=str, default=['sex', 'Age quant'])
parser.add_argument('--age_bin', type=float, default=10)
parser.add_argument("--source", type = str, default = "")
parser.add_argument("--num_warmup_steps", type = int, default = 0)
args = parser.parse_args() # parse arguments


# Save parameters
try:
	os.makedirs(args.output_file)
except:
	pass

args.output_file = args.output_file + '/CatHarm_%d' %(len(os.listdir(args.output_file)))
os.makedirs(args.output_file + '/models')
with open(args.output_file + '/args.txt', 'w') as f:
	for k, v in vars(args).items():
		f.write('%s: %s\n' %(k, str(v)))

################################### <System set-up> ############################################
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')


#################################### <Data Prepare> ############################################
# Training Data Check and Download
image_path = Path(args.file)
data_path = Path("data")
if image_path.is_dir():
	print(f"[INFO] {image_path} directory exists, skipping download.")
else:
	source = args.source
	assert len(source) > 0, f"[ERROR] Invalid URL {source}"
	print(f"[INFO] Didn't find {image_path} directory, creating one...")
	image_path.mkdir(parents = True, exist_ok = True)

	# download data
	target_file = Path(source).name
	with open(data_path / target_file, "wb") as f:
		request = requests.get(source)
		print(f"[INFO] Downloading {target_file} from {source}")
		f.write(request.content)

	# Unzip data and remove .zip file
	with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
		print(f"[INFO] Unzipping {target_file} data...")
		zip_ref.extractall(image_path)

	os.remove(data_path / target_file)


# Load metadata
metadata = pd.read_csv(image_path + '/metadata.csv')

metadata['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)
metadata['Age quant'] = (metadata['mri_age_at_appointment']//args.age_bin).astype(int)
metadata = metadata[['sex', 'Age quant']]
metadata.reset_index(drop=True, inplace=True)

# Load file paths
paths = pd.read_csv(image_path + '/registered_files.csv')
paths = list(paths['PATH'])

# Train/Val split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(paths, np.array(metadata),
															  test_size=0.2, random_state=args.seed)

train_data = image_dataset(X_train, y_train)
test_data = image_dataset(X_test, y_test)

train_dataloader = DataLoader(train_data,
							 batch_size = args.batch_size,
							 shuffle = True,
							 num_workers = os.cpu_count(),
							 pin_memory = True)

test_dataloader = DataLoader(test_data,
						   batch_size = args.batch_size,
						   shuffle = False,
						   num_workers = os.cpu_count(),
						   pin_memory = True)


############################################ <Model> ############################################


					################# Vishnu's Version #################
# Vishnu's Version
# latent_dim = args.latent_dim
#
# model = ResNet_encdec(latent_dim=latent_dim).to(device)
# up_matrices = [Variable(torch.normal(mean=0, std=0.1, size=(latent_dim, latent_dim)).to(device), requires_grad=True) for _ in range(len(args.equivariant_variables))]
# down_matrices = [Variable(torch.normal(mean=0, std=0.1, size=(latent_dim, latent_dim)).to(device), requires_grad=True) for _ in range(len(args.equivariant_variables))]

# loss_mse = torch.nn.MSELoss()
# optimizer = optim.Adam([{'params': model.parameters()}] +
# 					   [{'params': up_matrix} for up_matrix in up_matrices] +
# 					   [{'params': down_matrix} for down_matrix in down_matrices], lr=args.lr)


                    ################# Peiran's Version #################
model = Encoder(conv_in_features = 1, conv_out_features = 128, kernel_size = 3, padding = True,
				batch_norm = True, img_size = 256, in_channels = 8, patch_size = 16, batch_size =
				32, num_transformer_layer = 2, embedding_dim = 2048, mlp_size = 4096, num_heads
				= 16, attention_dropout = .0, mlp_dropout = .1, embedding_dropout = .1,
				final_latent_space_dim = 2048)

summary(model, input_size = (32, 1, 157, 189, 156),
		col_names = ["input_size", "output_size", "num_params", "trainable"],
		col_width = 20,
		row_settings = ["var_names"])

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = args.lr)

num_training_steps = args.epochs * len(train_dataloader)  # assuming train_dataloader is your DataLoader
warmup_steps = args.num_warmup_steps

# Define a function that will be used to adjust the learning rate
# exponential warmup and decay
def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return math.exp(current_step / warmup_steps) - 1
    else:
        return math.exp((warmup_steps - current_step) / (num_training_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

########################################## <Training> ############################################

#### You can direct use utils.train_test_utils.train() ####
# first create a Tensorboard writer by
# writer = create_writer(experiment_name, model_name, extra_annotations)
# then call train() by
# results = train(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, epochs,
# device, writer = writer)
# where the "results" is the last expeiement results, which is a dictionary with keys
# you can directly draw the results by calling plot_loss_curves(results)


train_stats = {'train_loss': [],
			   'train_recon_loss': [],
			   'train_morph_loss': [],
			   'train_inv_loss': [],
			   'val_loss': [],
			   'val_recon_loss': [],
			   'val_morph_loss': []}


max_epochs = args.epochs
for epoch in range(max_epochs):

	# Train
	model.train()


	total_loss = 0
	total_recon_loss = 0
	total_morph_loss = 0
	total_inv_loss = 0


	progress_bar = tqdm(train_generator, total=len(train_generator))
	for batch in progress_bar:


		X = batch[0].to(device)
		mdata = batch[1]

		X_out, _, latent = model(X)

		recon_loss = loss_mse(X, X_out)


		# Morphisms preserving
		morph_loss = 0

		matrices_powers = []
		differences = []

		for variable_index in range(len(args.equivariant_variables)):

			matrices_powers.append({0: torch.eye(latent_dim).to(device)})
			differences.append((np.repeat(np.array(mdata[:, variable_index]).reshape((-1, 1)), mdata.shape[0], 1) - np.array(mdata[:, variable_index])).astype(int))


			max_pos_dif = differences[variable_index].max()
			max_neg_dif = -differences[variable_index].min()

			for p in range(1, max_neg_dif+1):
				matrices_powers[variable_index][p] = torch.linalg.matrix_power(down_matrices[variable_index], p)
			for p in range(1, max_pos_dif+1):
				matrices_powers[variable_index][-p] = torch.linalg.matrix_power(up_matrices[variable_index], p)




		for i in range(mdata.shape[0]):
			for j in range(i+1, mdata.shape[0]):

				current_matrix = torch.eye(latent_dim).to(device)
				for variable_index in range(len(args.equivariant_variables)):
					current_matrix = current_matrix @ matrices_powers[variable_index][differences[variable_index][i, j]]

				latent_mul = torch.matmul(latent[i], current_matrix)
				# Maybe cs would be better
				morph_loss += loss_mse(latent[j], latent_mul)



				current_matrix = torch.eye(latent_dim).to(device)
				for variable_index in range(len(args.equivariant_variables)):
					current_matrix = current_matrix @ matrices_powers[variable_index][differences[variable_index][j, i]]

				latent_mul = torch.matmul(latent[j], current_matrix)
				# Maybe cs would be better
				morph_loss += loss_mse(latent[i], latent_mul)




		morph_loss /= X.shape[0]**2

		# Inverse loss
		inv_loss = 0
		for i in range(len(args.equivariant_variables)):
			inv_loss += loss_mse(torch.matmul(up_matrices[i], down_matrices[i]), torch.eye(args.latent_dim).to(device))
			inv_loss += loss_mse(torch.matmul(down_matrices[i], up_matrices[i]), torch.eye(args.latent_dim).to(device))


		# Update parameters
		loss = recon_loss + args.morph_loss_factor*morph_loss + 1*inv_loss

		# Keep stats
		total_loss += loss.item()
		total_recon_loss += recon_loss.item()
		total_morph_loss += morph_loss.item()
		total_inv_loss += inv_loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		progress_bar.set_description("%3d/%3d - Train loss: %.4f (Recon: %.4f, Morph: %.8f, Inv: %.2f)" % (epoch+1, max_epochs, loss, recon_loss, morph_loss, inv_loss))



	train_stats['train_loss'].append(total_loss/len(train_generator))
	train_stats['train_recon_loss'].append(total_recon_loss/len(train_generator))
	train_stats['train_morph_loss'].append(total_morph_loss/len(train_generator))
	train_stats['train_inv_loss'].append(total_inv_loss/len(train_generator))


	# Eval
	model.eval()

	with torch.no_grad():


		recon_loss = 0
		morph_loss = 0



		progress_bar = tqdm(val_generator, total=len(val_generator))
		for batch in progress_bar:


			X = batch[0].to(device)
			mdata = batch[1]

			X_out, _, latent = model(X)


			recon_loss += loss_mse(X, X_out)


			# Morphisms preserving
			matrices_powers = []
			differences = []

			for variable_index in range(len(args.equivariant_variables)):

				matrices_powers.append({0: torch.eye(latent_dim).to(device)})
				differences.append((np.repeat(np.array(mdata[:, variable_index]).reshape((-1, 1)), mdata.shape[0], 1) - np.array(mdata[:, variable_index])).astype(int))


				max_pos_dif = differences[variable_index].max()
				max_neg_dif = -differences[variable_index].min()

				for p in range(1, max_neg_dif+1):
					matrices_powers[variable_index][p] = torch.linalg.matrix_power(down_matrices[variable_index], p)
				for p in range(1, max_pos_dif+1):
					matrices_powers[variable_index][-p] = torch.linalg.matrix_power(up_matrices[variable_index], p)




			for i in range(mdata.shape[0]):
				for j in range(i+1, mdata.shape[0]):

					current_matrix = torch.eye(latent_dim).to(device)
					for variable_index in range(len(args.equivariant_variables)):
						current_matrix = current_matrix @ matrices_powers[variable_index][differences[variable_index][i, j]]

					latent_mul = torch.matmul(latent[i], current_matrix)
					# Maybe cs would be better
					morph_loss += loss_mse(latent[j], latent_mul)



					current_matrix = torch.eye(latent_dim).to(device)
					for variable_index in range(len(args.equivariant_variables)):
						current_matrix = current_matrix @ matrices_powers[variable_index][differences[variable_index][j, i]]

					latent_mul = torch.matmul(latent[j], current_matrix)
					# Maybe cs would be better
					morph_loss += loss_mse(latent[i], latent_mul)


			morph_loss /= X.shape[0]**2



		recon_loss /=len(val_generator)
		morph_loss /=len(val_generator)

		print (" Recon: %.4f, Morph: %.8f" % (recon_loss, morph_loss))


		train_stats['val_loss'].append(recon_loss.item() + args.morph_loss_factor*morph_loss.item())
		train_stats['val_recon_loss'].append(recon_loss.item())
		train_stats['val_morph_loss'].append(morph_loss.item())



	torch.save(model.state_dict(), args.output_file + '/models/model_%d' %(epoch))

	for i, up_matrix in enumerate(up_matrices):
		torch.save(up_matrix, args.output_file + '/models/up_matrix_%d_%d' %(i, epoch))

	for i, down_matrix in enumerate(down_matrices):
		torch.save(down_matrix, args.output_file + '/models/down_matrix_%d_%d' %(i, epoch))

train_stats_pd = pd.DataFrame.from_dict(train_stats)
train_stats_pd.to_csv(args.output_file + '/train_stats.csv')
