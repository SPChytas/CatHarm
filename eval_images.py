

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
from utils.data_loader import image_dataset

import sys

import psutil


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='data/imaging_data/(256, 256, 156)')

parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--seed', type=int, default=42)


args = parser.parse_args()





# System set-up
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(f'Selected device: {device}')



print('RAM Used (GB):', psutil.virtual_memory()[3]/1024**3)


############################################ <Data> ############################################
# Load metadata
metadata = pd.read_csv(args.file + '/metadata.csv')

metadata['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)
metadata['diagnosis'].replace({'Control': 0, 'Presumed AD': 1}, inplace=True)
metadata['mri_coil_name'].replace({'RM:Nova32ch': 0, '8HRBRAIN': 1}, inplace=True)
metadata['scanner_source'].replace({'Waisman': 0, 'WIMR': 1, 'WIMR-FOR RESEARCH USE ONLY': 1, 'UWMF': 1}, inplace=True)


columns = list(metadata.columns)



paths = pd.read_csv(args.file + '/registered_files.csv')
paths = list(paths['PATH'])


# Train/Val split
paths_train, paths_val, metadata_train, metadata_val = train_test_split(paths, np.array(metadata), test_size=0.2, random_state=args.seed)
############################################ </Data> ############################################


print('RAM Used (GB):', psutil.virtual_memory()[3]/1024**3)



original_stdout = sys.stdout

with open(args.file + '/adversarial_evaluation.txt', 'w') as f:

	sys.stdout = f


	for v in ['diagnosis', 'mri_coil_name', 'scanner_source']:
	
		v_index = columns.index(v)

		preload = True
		train_dataset = image_dataset(paths_train, metadata_train[:, v_index], preload)
		val_dataset = image_dataset(paths_val, metadata_val[:, v_index], preload)

		params = {'batch_size': args.batch_size,
				'shuffle': True,
				'num_workers': 1}

		train_generator = DataLoader(train_dataset, **params)
		val_generator = DataLoader(val_dataset, **params)



		print ('-'*20)
		print (v + ': %.2f (+-%.2f)' %(metrics.predict_invariant_variable((train_generator, val_generator), image_data=True)))
		print ('\t random guess is: %.2f' %(max(metadata[v].sum()/metadata.shape[0], 1- metadata[v].sum()/metadata.shape[0])))

		
	print ('-'*20)

	
