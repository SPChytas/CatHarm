# from data_loader import ADNI


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
from utils.models import MLP_encdec
from utils.data_loader import numerical_dataset

import sys


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--output_file', type=str)

parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--morph_loss_factor', type=float, default=1)

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--invariant_variables', nargs='+', type=str, default=['mri_coil_name', 'scanner_source'])
parser.add_argument('--equivariant_variables', nargs='+', type=str, default=['sex', 'Age quant'])

parser.add_argument('--age_bin', type=float, default=10)

args = parser.parse_args()




# Save parameters
args.output_file = args.output_file + '/CatHarm_%d' %(len(os.listdir(args.output_file)))
os.makedirs(args.output_file)
os.makedirs(args.output_file + '/models')

with open(args.output_file + '/args.txt', 'w') as f:
	for k, v in vars(args).items():
		f.write('%s: %s\n' %(k, str(v)))



# System set-up
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(f'Selected device: {device}')





############################################ <Data> ############################################

# Load data
all_data = pd.read_csv(args.file + '/matched_sample_coil_data.csv')


# Replace str with numbers and drop useless columns
all_data.drop('reggieid', axis=1, inplace=True)
all_data['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)
all_data['diagnosis'].replace({'Control': 0, 'Presumed AD': 1}, inplace=True)
all_data.drop('mri_model_name', axis=1, inplace=True)
all_data['mri_coil_name'].replace({'RM:Nova32ch': 0, '8HRBRAIN': 1}, inplace=True)
all_data['scanner_source'].replace({'Waisman': 0, 'WIMR': 1}, inplace=True)


data = all_data[['gm_vol', 'wm_vol', 'csf_vol', 'global_atrophy', 'icv', 'right_hc_vol', 'left_hc_vol', 'total_hippocampal_volume', 'hippocampal_volume_adj']]
metadata = all_data[['sex', 'mri_age_at_appointment', 'diagnosis', 'mri_coil_name', 'scanner_source']]
metadata['Age quant'] = (metadata['mri_age_at_appointment']//args.age_bin).astype(int)

data.reset_index(drop=True, inplace=True)
metadata.reset_index(drop=True, inplace=True)


print (data.head(10))


# Normalize data
scaler = StandardScaler()	
data_normalized = scaler.fit_transform(np.array(data))

############################################ </Data> ############################################



############################################ <Pre-harmonization evaluation> ############################################

original_stdout = sys.stdout


with open(args.output_file + '/evaluation.txt', 'w') as f:

	sys.stdout = f

	print ('########## <Pre-harmonization evaluation> ##########')

	for v in args.invariant_variables:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data_normalized, metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

		print ('  MMD (rbf): %.4f' %(metrics.mmd_rbf(np.array(data_normalized[metadata[v] == 0]), np.array(data_normalized[metadata[v] == 1]))))
		print ('  MMD (poly): %.4f' %(metrics.mmd_poly(np.array(data_normalized[metadata[v] == 0]), np.array(data_normalized[metadata[v] == 1]))))
		print ('  MMD (linear): %.4f' %(metrics.mmd_linear(np.array(data_normalized[metadata[v] == 0]), np.array(data_normalized[metadata[v] == 1]))))


	# for v in args.equivariant_variables:
	# 	print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data, metadata[v])))
	# 	print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))


	for v in ['diagnosis']:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data_normalized, metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

	print ('-'*20)

	print ('########## </Pre-harmonization evaluation> ##########\n\n')

	sys.stdout = original_stdout

############################################ </Pre-harmonization evaluation> ############################################




# ############################################ <Model> ############################################
latent_dim = args.latent_dim

encoder = MLP_encdec(data_normalized.shape[1], [30, 30], latent_dim, batchnorm=False, dropout=0).to(device)
decoder = MLP_encdec(latent_dim, [30, 30], data_normalized.shape[1], batchnorm=False, dropout=0).to(device)


# Better initialization...
up_matrices = [Variable(torch.normal(mean=0, std=0.1, size=(latent_dim, latent_dim)).to(device), requires_grad=True) for _ in range(len(args.equivariant_variables))]
down_matrices = [Variable(torch.normal(mean=0, std=0.1, size=(latent_dim, latent_dim)).to(device), requires_grad=True) for _ in range(len(args.equivariant_variables))]



loss_mse = torch.nn.MSELoss()

optimizer = optim.Adam([{'params': encoder.parameters()}, 
						{'params': decoder.parameters()}] +
					   [{'params': up_matrix} for up_matrix in up_matrices] +
					   [{'params': down_matrix} for down_matrix in down_matrices], lr=args.lr)





# DataLoader parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 1}

train_data, val_data, train_metadata, val_metadata = train_test_split(data_normalized, np.array(metadata[args.equivariant_variables]), test_size=0.2, random_state=args.seed)


# Create data generators
train_dataset = numerical_dataset(train_data, train_metadata)
train_generator = DataLoader(train_dataset, **params)

val_dataset = numerical_dataset(val_data, val_metadata)
val_generator = DataLoader(val_dataset, **params)
# ############################################ </Model> ############################################





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
	encoder.train()
	decoder.train()


	total_loss = 0
	total_recon_loss = 0
	total_morph_loss = 0 
	total_inv_loss = 0


	progress_bar = tqdm(train_generator, total=len(train_generator))
	for batch in progress_bar:


		X = batch[0].to(device)
		mdata = batch[1]

		latent = encoder(X)
		X_out = decoder(latent)

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
	encoder.eval()
	decoder.eval()

	with torch.no_grad():


		recon_loss = 0
		morph_loss = 0



		progress_bar = tqdm(val_generator, total=len(val_generator))
		for batch in progress_bar:


			X = batch[0].to(device)
			mdata = batch[1]

			latent = encoder(X)
			X_out = decoder(latent)

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
		


	torch.save(encoder.state_dict(), args.output_file + '/models/encoder_%d' %(epoch))
	torch.save(decoder.state_dict(), args.output_file + '/models/decoder_%d' %(epoch))

	for i, up_matrix in enumerate(up_matrices):
		torch.save(up_matrix, args.output_file + '/models/up_matrix_%d_%d' %(i, epoch))

	for i, down_matrix in enumerate(down_matrices):
		torch.save(down_matrix, args.output_file + '/models/down_matrix_%d_%d' %(i, epoch))



train_stats_pd = pd.DataFrame.from_dict(train_stats)
train_stats_pd.to_csv(args.output_file + '/train_stats.csv')

############################################ <Post-harmonization evaluation> ############################################


data_normalized = decoder(encoder(torch.FloatTensor(data_normalized).to(device))).cpu().detach().numpy()


with open(args.output_file + '/evaluation.txt', 'a') as f:

	sys.stdout = f

	print ('########## <Post-harmonization evaluation> ##########')

	for v in args.invariant_variables:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data_normalized, metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

		print ('  MMD (rbf): %.4f' %(metrics.mmd_rbf(np.array(data_normalized[metadata[v] == 0]), np.array(data_normalized[metadata[v] == 1]))))
		print ('  MMD (poly): %.4f' %(metrics.mmd_poly(np.array(data_normalized[metadata[v] == 0]), np.array(data_normalized[metadata[v] == 1]))))
		print ('  MMD (linear): %.4f' %(metrics.mmd_linear(np.array(data_normalized[metadata[v] == 0]), np.array(data_normalized[metadata[v] == 1]))))


	# for v in args.equivariant_variables:
	# 	print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data, metadata[v])))
	# 	print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))



	for v in ['diagnosis']:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data_normalized, metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

	print ('-'*20)

	print ('########## </Post-harmonization evaluation> ##########\n\n')

	sys.stdout = original_stdout

############################################ </Post-harmonization evaluation> ############################################



final_data = pd.DataFrame(scaler.inverse_transform(data_normalized), columns=data.columns)
print (final_data.head(10))
final_data.to_csv(args.output_file + '/harmonized_data.csv', index=False)