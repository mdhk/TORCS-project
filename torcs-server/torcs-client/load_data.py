import glob
import dill
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
	for i, data_column in enumerate(all_data[:, 4:].T):
		normalised_column = mean_normalisation(data_column)
		all_data[:, 4:].T[i] = normalised_column
	X = torch.FloatTensor(all_data[:, 4:])
	return TensorDataset(X, Y)

def mean_normalisation(column):
	normalised_column = []
	mean_x = np.mean(column)
	min_x = np.min(column)
	max_x = np.max(column)
	if max_x - min_x == 0:
		return np.zeros(len(column))
	for i, x in enumerate(column):
		normalised_x = (x - mean_x)/(max_x - min_x)
		normalised_column.append(normalised_x)
	return np.asarray(normalised_column)

if __name__ == "__main__":
	all_races = []
	for csv in glob.glob(log_dir + '*.csv'):
		race = read_data(csv)
		all_races.append(race)
	dill.dump(all_races, open("train_data/race_list.pkl", "wb"))
	race_list = dill.load(open("train_data/race_list.pkl", "rb"))
	print(race_list)
