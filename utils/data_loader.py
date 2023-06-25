from torchvision.transforms.functional import rotate
from torchvision.transforms import Resize
import torch.nn.functional as F
import torch
from torchvision.io import read_image
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib
import joblib

def collate_fn(samples):

	original_images = np.array([s[0].numpy() for s in samples])
	rot_images = np.array([s[1].numpy() for s in samples])
	scale_images = np.array([s[2].numpy() for s in samples])

	rotations = [s[3] for s in samples]
	scales = [s[4] for s in samples]

	return (torch.FloatTensor(original_images),
			torch.FloatTensor(rot_images),
			torch.FloatTensor(scale_images),
			rotations,
			scales)

class numerical_dataset(Dataset):
	def __init__(self, data, metadata):
		self.data = data
		self.metadata = metadata

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		return (torch.FloatTensor(self.data[index]), torch.FloatTensor(self.metadata[index]))

class image_dataset(Dataset):
	def __init__(self, files_path, metadata):

		self.files_path = files_path
		self.metadata = metadata

	def __len__(self):
		return len(self.files_path)

	def __getitem__(self, index):

		############ Normalization??
		image = nib.load(self.files_path[index]).get_fdata()
		image[np.isnan(image)] = np.mean(image[~np.isnan(image)])
		image /= max(np.max(image), -np.min(image))

		return torch.unsqueeze(torch.FloatTensor(image), 0), torch.FloatTensor(self.metadata[index])

class ADNI(Dataset):

	def __init__(self, img_names, metadata, attributes=['Age', 'Sex'], return_type='int'):

		self.imgs, self.attributes, self.labels = self._load_X(img_names, metadata, attributes) 

		# Torch transform better?
		X_mean = np.mean(self.imgs)
		X_std = np.std(self.imgs)
		self.imgs = (self.imgs - X_mean) / X_std


		self.img_names = img_names
		self.metadata = metadata
		self.attributes_names = attributes

		self.type = return_type

	def __len__ (self):
		return len(self.labels)

	def __getitem__ (self, index):

		l = 1 if self.labels[index]=='AD' else 0

		if (self.type == 'int'):
			return torch.FloatTensor(self.imgs[index]), torch.FloatTensor([l]), torch.LongTensor(self.attributes.iloc[index].to_numpy(dtype=int)) #torch.FloatTensor(self.attributes.iloc[index].to_numpy(dtype=float))
		else:
			return torch.FloatTensor(self.imgs[index]), torch.FloatTensor([l]), torch.FloatTensor(self.attributes.iloc[index].to_numpy(dtype=float))


	def _load_X(self, img_names, metadata, attributes):

		imgs = []
		attrs = pd.DataFrame(columns=attributes)
		label = []

		for name in tqdm(img_names, total=len(img_names), desc='reading ADNI data in memory'):
			imgs.append(np.load('ADNI/mr_machine/' + name + '.npy'))

			attrs = pd.concat((attrs, metadata[metadata['Subject ID'] == name][attributes]), axis=0, ignore_index=True)
			label.append(metadata[metadata['Subject ID'] == name]['Research Group'].iloc[0])

		imgs = np.array(imgs)

		return imgs, attrs, label


class ADCP(Dataset):

	def __init__(self, img_names, metadata, attributes=['Age', 'Gender'], return_type='int'):

		self.imgs, self.attributes, self.labels = self._load_X(img_names, metadata, attributes) 

		# Torch transform better?
		X_mean = np.mean(self.imgs)
		X_std = np.std(self.imgs)
		self.imgs = (self.imgs - X_mean) / X_std


		self.img_names = img_names
		self.metadata = metadata
		self.attributes_names = attributes

		self.type = return_type

	def __len__ (self):
		return len(self.labels)

	def __getitem__ (self, index):

		l = 1 if self.labels[index]=='AD' else 0

		if (self.type == 'int'):
			return torch.FloatTensor(self.imgs[index]), torch.FloatTensor([l]), torch.LongTensor(self.attributes.iloc[index].to_numpy(dtype=int)) #torch.FloatTensor(self.attributes.iloc[index].to_numpy(dtype=float))
		else:
			return torch.FloatTensor(self.imgs[index]), torch.FloatTensor([l]), torch.FloatTensor(self.attributes.iloc[index].to_numpy(dtype=float))


	def _load_X(self, img_names, metadata, attributes):

		imgs = []
		attrs = pd.DataFrame(columns=attributes)
		label = []

		for name in tqdm(img_names, total=len(img_names), desc='reading ADCP data in memory'):
			imgs.append(np.array(nib.load('ADCP_harmonization/' + name + '/T1w_acpc_dc_restore_1mm.nii.gz').dataobj))

			attrs = pd.concat((attrs, metadata[metadata['Subj_ID'] == name][attributes]), axis=0, ignore_index=True)
			label.append(metadata[metadata['Subj_ID'] == name]['Group'].iloc[0])

		imgs = np.array(imgs)

		return imgs, attrs, label

class GattrDataset(Dataset):
	"""
		Adult dataset, relies on the processed data generated from uci_data.py
		data_path : path to *_proc_gattr.z
		split : 'train'/'test'/'val'
		transform : None at the moment
	"""

	def __init__(self, data_path, split='train'):
		super(GattrDataset, self).__init__()
		self.data_path = data_path
		self.confound_max = 100.0

		train_total_data, split_numbers, train_size, \
		validation_data, validation_labels, validation_confounds, validation_gattr, \
		test_data, test_labels, test_confounds, test_gattr = \
			joblib.load(self.data_path)

		self.split = split
		if self.split == 'train':
			self.data = train_total_data[:, :split_numbers[0]]
			self.labels = train_total_data[:, split_numbers[0]]
			self.confounds = train_total_data[:, split_numbers[1]]
			self.gattr = train_total_data[:, split_numbers[2]]
		elif self.split == 'test':
			self.data = test_data
			self.labels = test_labels
			self.confounds = test_confounds
			self.gattr = test_gattr
		elif self.split == 'val':
			self.data = validation_data
			self.labels = validation_labels
			self.confounds = validation_confounds
			self.gattr = validation_gattr			
		else:
			raise ('Not Implemented Error')

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		x = self.data[index, :].astype(float)
		y = int(self.labels[index])
		c = int(self.confounds[index])
		gattr =  int(np.round(5*self.gattr[index]))




		return torch.tensor(x).float(), torch.tensor(y).float(), torch.tensor(c).long(), torch.tensor(gattr).long()
		