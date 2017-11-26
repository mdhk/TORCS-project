import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

def read_data(*csv_filepaths):
	"""
	Reads data from csv files, returns X and Y as numpy matrices.
	"""
	all_data = np.concatenate([pd.read_csv(csv_file) for csv_file in csv_filepaths], axis=0)

	Y = all_data[:, :4]
	X = all_data[:, 3:]

	return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))

train_data = read_data('train_data/aalborg.csv', 'train_data/alpine-1.csv', 'train_data/f-speedway.csv')
