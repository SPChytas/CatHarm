
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


		self.max_val = -float('inf')
		self.min_val = float('inf')

		for i in range(len(self.files_path)):
				
			image = nib.load(self.files_path[i]).get_fdata()
			self.min_val = np.nanmin(image)
			self.max_val = np.nanmax(image)




		if (self.preload):
			
			for i in tqdm(range(len(self.files_path)), desc='Preloading...'):
				
				image = nib.load(self.files_path[i]).get_fdata()
				image = (image - self.min_val)/(self.max_val - self.min_val)
				image[np.isnan(image) | np.isinf(image)] = 0
				
				self.images.append(image)


	def __len__(self):
		return len(self.files_path)

	def __getitem__(self, index):


		if (self.preload):
			image = self.images[index]
		else:
			image = nib.load(self.files_path[index]).get_fdata()
			image = (image - self.min_val)/(self.max_val - self.min_val)
			image[np.isnan(image) | np.isinf(image)] = 0


		if (len(self.metadata.shape) == 1):
			return torch.unsqueeze(torch.FloatTensor(image), 0), torch.FloatTensor([self.metadata[index]])
		else:
			return torch.unsqueeze(torch.FloatTensor(image), 0), torch.FloatTensor(self.metadata[index])


