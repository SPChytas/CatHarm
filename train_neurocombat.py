# from data_loader import ADNI
from neuroCombat import neuroCombat

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

import time
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--output_file', type=str)

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--invariant_variable', type=str, default='mri_coil_name')
args = parser.parse_args()





# Save parameters
args.output_file = args.output_file + '/ComBat_%d' %(len(os.listdir(args.output_file)))
os.makedirs(args.output_file)

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

data.reset_index(drop=True, inplace=True)
metadata.reset_index(drop=True, inplace=True)

print (data.head(10))

categorical_cols = ['sex', 'diagnosis']
continuous_cols = ['mri_age_at_appointment']
batch_cols = args.invariant_variable


# Normalize data
scaler = StandardScaler()	
data_normalized = scaler.fit_transform(np.array(data))
data = pd.DataFrame(data_normalized, columns=data.columns)


# data.hist()
# plt.show()
############################################ </Data> ############################################



############################################ <Pre-harmonization evaluation> ############################################

original_stdout = sys.stdout

with open(args.output_file + '/evaluation.txt', 'w') as f:

	sys.stdout = f

	print ('########## <Pre-harmonization evaluation> ##########')

	for v in ['mri_coil_name', 'scanner_source']:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(np.array(data), metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

		print ('  MMD (rbf): %.4f' %(metrics.mmd_rbf(np.array(data[metadata[v] == 0]), np.array(data[metadata[v] == 1]))))
		print ('  MMD (poly): %.4f' %(metrics.mmd_poly(np.array(data[metadata[v] == 0]), np.array(data[metadata[v] == 1]))))
		print ('  MMD (linear): %.4f' %(metrics.mmd_linear(np.array(data[metadata[v] == 0]), np.array(data[metadata[v] == 1]))))


	# for v in args.equivariant_variables:
	# 	print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data, metadata[v])))
	# 	print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))



	for v in ['diagnosis']:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(np.array(data), metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

	print ('-'*20)

	print ('########## </Pre-harmonization evaluation> ##########\n\n')

	sys.stdout = original_stdout

############################################ </Pre-harmonization evaluation> ############################################




# ############################################ <Model> ############################################

t = time.time()

data_combat = neuroCombat(dat=data.T,
						  covars=metadata,
						  batch_col=batch_cols,
						  categorical_cols=categorical_cols,
						  continuous_cols=continuous_cols,
						  eb=True,
						  parametric=True,
						  mean_only=False)


t = time.time() - t 

print ('Elapsed time: %.2f' %(t))
# ############################################ </Model> ############################################







############################################ <Post-harmonization evaluation> ############################################


################# HERE SHOULD BE VAL DATA #################
data_normalized = data_combat['data'].T
################# HERE SHOULD BE VAL DATA #################


data = pd.DataFrame(data_normalized, columns=data.columns)
# data.hist()
# plt.show()


with open(args.output_file + '/evaluation.txt', 'a') as f:

	sys.stdout = f

	print ('########## <Post-harmonization evaluation> ##########')

	for v in ['mri_coil_name', 'scanner_source']:
		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data_normalized, metadata[v])))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))
		
		print ('  MMD (rbf): %.4f' %(metrics.mmd_rbf(np.array(data[metadata[v] == 0]), np.array(data[metadata[v] == 1]))))
		print ('  MMD (poly): %.4f' %(metrics.mmd_poly(np.array(data[metadata[v] == 0]), np.array(data[metadata[v] == 1]))))
		print ('  MMD (linear): %.4f' %(metrics.mmd_linear(np.array(data[metadata[v] == 0]), np.array(data[metadata[v] == 1]))))



	# for v in args.equivariant_variables:
	# 	print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable(data_normalized, metadata[v])))
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