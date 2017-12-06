import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import cProfile

log_dir = 'bram_logs/'

def read_data(csv_filepath):
	"""
	Reads data from csv file, returns X and Y as torch FloatTensors.
	"""
	all_data = np.asarray(pd.read_csv(csv_filepath))
	Y = torch.FloatTensor(all_data[:, :4])
	X = torch.FloatTensor(all_data[:, 4:])
	return TensorDataset(X, Y)

all_races = []
for csv in glob.glob(log_dir + '*.csv'):
	race = read_data(csv)
	all_races.append(race)
