from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.parametrizations import orthogonal

import matplotlib.pyplot as plt 
import seaborn as sns

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
parser.add_argument('--original_data', type=str)
parser.add_argument('--harmonized_data', type=str)
args = parser.parse_args()






orig_data = pd.read_csv(args.original_data)
harm_data = pd.read_csv(args.harmonized_data)



print (orig_data.head(5))
print (harm_data.head(5))



# Test 1: preserve the fact that total_hippocampal_volume = right_hc_vol + left_hc_vol

print ('Test 1 - preserve the fact total_hippocampal_volume = right_hc_vol + left_hc_vol:')
print ('\t Original data difference: %.2f (+- %.2f)' %(np.mean(orig_data['total_hippocampal_volume'] - orig_data['right_hc_vol'] - orig_data['left_hc_vol']),
													np.std(orig_data['total_hippocampal_volume'] - orig_data['right_hc_vol'] - orig_data['left_hc_vol'])))
print ('\t Harmonized data difference: %.2f (+- %.2f)' %(np.mean(harm_data['total_hippocampal_volume'] - harm_data['right_hc_vol'] - harm_data['left_hc_vol']),
													  np.std(harm_data['total_hippocampal_volume'] - harm_data['right_hc_vol'] - harm_data['left_hc_vol'])))


print ('Test 2 - preserve the fact global_atrophy = csf_vol/(gm_vol + wm_vol):')
print ('\t Original data difference: %.2f (+- %.2f)' %(np.mean(orig_data['global_atrophy'] - orig_data['csf_vol']/(orig_data['gm_vol'] + orig_data['wm_vol'])),
													  np.std(orig_data['global_atrophy'] - orig_data['csf_vol']/(orig_data['gm_vol'] + orig_data['wm_vol']))))
print ('\t Harmonized data difference: %.2f (+- %.2f)' %(np.mean(harm_data['global_atrophy'] - harm_data['csf_vol']/(harm_data['gm_vol'] + harm_data['wm_vol'])),
													     np.std(harm_data['global_atrophy'] - harm_data['csf_vol']/(harm_data['gm_vol'] + harm_data['wm_vol']))))

for v in harm_data.columns:

	plt.scatter(orig_data[v], harm_data[v])

	m = min(np.min(orig_data[v]), np.min(harm_data[v]))
	M = max(np.max(orig_data[v]), np.max(harm_data[v]))
	line = np.linspace(m, M)
	plt.plot(line, line, 'r--')

	plt.title(v)

	plt.savefig(v + '_line')
	plt.close()

	# plt.show()


	fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 6))
	sns.kdeplot(orig_data[v], hue=orig_data['mri_coil_name'], ax=ax[0, 0], fill=True)
	sns.kdeplot(harm_data[v], hue=orig_data['mri_coil_name'], ax=ax[1, 0], fill=True)

	sns.kdeplot(orig_data[v], hue=orig_data['scanner_source'], ax=ax[0, 1], fill=True)
	sns.kdeplot(harm_data[v], hue=orig_data['scanner_source'], ax=ax[1, 1], fill=True)
	# plt.title(v)


	plt.savefig(v)
	plt.close()

	# plt.show()






