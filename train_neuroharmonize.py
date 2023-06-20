
from neuroHarmonize.harmonizationNIFTI import createMaskNIFTI
from neuroHarmonize.harmonizationNIFTI import flattenNIFTIs
from neuroHarmonize.harmonizationNIFTI import applyModelNIFTIs
import neuroHarmonize as nh


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
args.output_file = args.output_file + '/NeuroHarmonizer_%d' %(len(os.listdir(args.output_file)))
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

##### Covariates #####
all_data = pd.read_csv(args.file + '/metadata_(256, 256, 156).csv')

# Replace str with numbers and drop useless columns
all_data.drop('reggieid', axis=1, inplace=True)
all_data['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)
all_data['diagnosis'].replace({'Control': 0, 'Presumed AD': 1}, inplace=True)
all_data.drop('mri_model_name', axis=1, inplace=True)
all_data['mri_coil_name'].replace({'RM:Nova32ch': 0, '8HRBRAIN': 1}, inplace=True)
all_data['scanner_source'].replace({'Waisman': 0, 'WIMR': 1}, inplace=True)


if (args.invariant_variable == 'mri_coil_name'):
	all_data.drop('scanner_source', axis=1, inplace=True)
	all_data.rename(columns={'mri_coil_name': 'SITE'}, inplace=True)
else:
	all_data.drop('mri_coil_name', axis=1, inplace=True)
	all_data.rename(columns={'scanner_source': 'SITE'}, inplace=True)



all_data.reset_index(drop=True, inplace=True)

print (all_data.head(10))



##### MRIs #####
nifti_list = pd.read_csv(args.file + '/files_(256, 256, 156).csv')
nifti_avg, nifti_mask, affine, hdr0 = createMaskNIFTI(nifti_list, threshold=0, output_path=args.output_file + '/thresholded_mask.nii.gz')



############################################ </Data> ############################################






# ############################################ <Model> ############################################

t = time.time()

nifti_array = flattenNIFTIs(nifti_list, mask_path=args.output_file + '/thresholded_mask.nii.gz', output_path=args.output_file + '/flattened_NIFTI_array.npy')
my_model, nifti_array_adj = nh.harmonizationLearn(nifti_array, all_data)
nh.saveHarmonizationModel(my_model, args.output_file + '/MY_MODEL')

t = time.time() - t 

print ('Elapsed time: %.2f' %(t))



applyModelNIFTIs(all_data, my_model, nifti_list, mask_path=args.output_file + '/thresholded_mask.nii.gz')

# ############################################ </Model> ############################################





