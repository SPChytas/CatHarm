
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




class numerical_dataset(Dataset):

	def __init__(self, data, metadata):
		self.data = data
		self.metadata = metadata

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		return (torch.FloatTensor(self.data[index]), torch.FloatTensor(self.metadata[index]))




class image_dataset(Dataset):

	def __init__(self, files_path, metadata, preload=True):

		self.files_path = files_path
		self.metadata = metadata
		
		self.images = []
		self.preload = preload

		if (self.preload):
			
			for i in tqdm(range(len(self.files_path)), desc='Preloading...'):
				
				image = nib.load(self.files_path[i]).get_fdata()
				image[np.isnan(image)] = np.mean(image[~np.isnan(image)])
				image /= max(np.max(image), -np.min(image))

				self.images.append(image)


	def __len__(self):
		return len(self.files_path)

	def __getitem__(self, index):


		if (self.preload):
			image = self.images[index]
		else:
			############ Normalization??
			image = nib.load(self.files_path[index]).get_fdata()
			image[np.isnan(image)] = np.mean(image[~np.isnan(image)])
			image /= max(np.max(image), -np.min(image))


		if (len(self.metadata.shape) == 1):
			return torch.unsqueeze(torch.FloatTensor(image), 0), torch.FloatTensor([self.metadata[index]])
		else:
			return torch.unsqueeze(torch.FloatTensor(image), 0), torch.FloatTensor(self.metadata[index])


