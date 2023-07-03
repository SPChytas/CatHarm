import torch
import numpy as np

from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score

from .models import MLP, Encoder
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics

from tqdm.auto import tqdm

import psutil



######################################## <Statistical> ########################################

# Taken from: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
def mmd_linear(X, Y):
	"""MMD using linear kernel (i.e., k(x,y) = <x,y>)
	Note that this is not the original linear MMD, only the reformulated and faster version.
	The original version is:
		def mmd_linear(X, Y):
			XX = np.dot(X, X.T)
			YY = np.dot(Y, Y.T)
			XY = np.dot(X, Y.T)
			return XX.mean() + YY.mean() - 2 * XY.mean()
	Arguments:
		X {[n_sample1, dim]} -- [X matrix]
		Y {[n_sample2, dim]} -- [Y matrix]
	Returns:
		[scalar] -- [MMD value]
	"""
	delta = X.mean(0) - Y.mean(0)
	return delta.dot(delta.T)

def mmd_rbf(X, Y, gamma=1.0):
	"""MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
	Arguments:
		X {[n_sample1, dim]} -- [X matrix]
		Y {[n_sample2, dim]} -- [Y matrix]
	Keyword Arguments:
		gamma {float} -- [kernel parameter] (default: {1.0})
	Returns:
		[scalar] -- [MMD value]
	"""
	XX = metrics.pairwise.rbf_kernel(X, X, gamma)
	YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
	XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
	return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
	"""MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
	Arguments:
		X {[n_sample1, dim]} -- [X matrix]
		Y {[n_sample2, dim]} -- [Y matrix]
	Keyword Arguments:
		degree {int} -- [degree] (default: {2})
		gamma {int} -- [gamma] (default: {1})
		coef0 {int} -- [constant item] (default: {0})
	Returns:
		[scalar] -- [MMD value]
	"""
	XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
	YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
	XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
	return XX.mean() + YY.mean() - 2 * XY.mean()
	
######################################## </Statistical> ########################################










######################################## <Adversarial> ########################################

def _predict_invariant_variable_tabular(X, y, classification, seed):

	if (classification):
		MLP = MLPClassifier
		metric = 'accuracy'
	else:
		MLP = MLPRegressor
		metric = 'r2'

	model = MLP(hidden_layer_sizes=(10*X.shape[1],),
				solver='adam',
				learning_rate_init=1e-2,
				early_stopping=True,
				n_iter_no_change=50,
				max_iter=2000,
				random_state=seed)

	scores = cross_val_score(model, X, y, scoring=metric, cv=10)

	return np.mean(scores), np.std(scores)


def _predict_invariant_variable_image(train_dataloader, val_dataloader, classification, seed):

	
	torch.manual_seed(seed)
	np.random.seed(seed)

	# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	device = torch.device("cpu")
	print(f'Selected device: {device}')


	latent_dim = 2048
	epochs = 100 

	encoder = Encoder(final_latent_space_dim=latent_dim, num_heads=8).to(device)
	pred_head = MLP(latent_dim, [256], 1, classification=classification).to(device)


	param_size = 0
	for param in encoder.parameters():
	    param_size += param.nelement() * param.element_size()
	for param in pred_head.parameters():
	    param_size += param.nelement() * param.element_size()
	
	buffer_size = 0
	for buffer in encoder.buffers():
	    buffer_size += buffer.nelement() * buffer.element_size()
	for buffer in pred_head.buffers():
	    buffer_size += buffer.nelement() * buffer.element_size()

	size_all_gb = (param_size + buffer_size) / 1024**3
	print('model size: {:.3f}GB'.format(size_all_gb))


	optimizer = optim.Adam([{'params': encoder.parameters()}] +
						   [{'params': pred_head.parameters()}])

	loss = nn.BCELoss() if classification else nn.MSELoss()
	metric = torchmetrics.classification.BinaryAccuracy() if classification else nn.MSELoss()

	if (classification):
		best_metric = 0 
	else:
		best_metric = float('inf')


	print('RAM Used (GB):', psutil.virtual_memory()[3]/1024**3)


	for ep in tqdm(range(epochs)):

		encoder.train()
		pred_head.train()

		print('RAM Used (GB):', psutil.virtual_memory()[3]/1024**3)


		for X, y in tqdm(train_dataloader, total=len(train_dataloader)):

			# print (X.shape, y.shape)

			latent = encoder(X.to(device))
			y_out = pred_head(latent)

			l = loss(y_out, y.to(device))

			optimizer.zero_grad()
			l.backward()
			optimizer.step()

			torch.cuda.empty_cache()


		encoder.eval()
		pred_head.eval()

		batch_metric = [] 
		for X, y in val_dataloader:
			latent = encoder(X.to(device))
			y_out = pred_head(latent)

			batch_metric.append(metric(y_out, y.to(device)))

		if (classification):
			best_metric = max(best_metric, np.mean(batch_metric))
		else:
			best_metric = min(best_metric, np.mean(batch_metric))

		torch.cuda.empty_cache()


	return best_metric, 0





def predict_invariant_variable(data, classification=True, seed=42, image_data=False):

	if (not image_data):
		X, y = data
		return _predict_invariant_variable_tabular(X, y, classification, seed)
	else:
		train_dataloader, val_dataloader = data
		return _predict_invariant_variable_image(train_dataloader, val_dataloader, classification, seed)

######################################## </Adversarial> ########################################
