import glob
import cloudpickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import cProfile

log_dir = 'bram_logs/'

def read_data(csv_filepath, normalisation=False):
	"""
	Reads data from csv file, returns X and Y as torch FloatTensors.
	"""
	all_data = np.asarray(pd.read_csv(csv_filepath))
	Y = torch.FloatTensor(all_data[:, :4])
	if normalisation:
		normalised_columns = []
		for i, data_column in enumerate(all_data[:, 4:].T):
			normalised_column = mean_normalisation(data_column)
			normalised_columns.append(normalised_column)
	x = np.asarray(normalised_columns).T
	X = torch.FloatTensor(x)
	return TensorDataset(X, Y)

def mean_normalisation(column):
	normalised_column = []
	mean_x = np.mean(column)
	std_x = np.std(column)
	if std_x == 0:
		return np.zeros(len(column))
	for i, x in enumerate(column):
		normalised_x = (x - mean_x)/std_x
		normalised_column.append(normalised_x)
	return np.asarray(normalised_column)

if __name__ == "__main__":
	all_races = []
	for csv in glob.glob(log_dir + '*.csv'):
		race = read_data(csv, normalisation=True)
		all_races.append(race)
	cloudpickle.dump(all_races, open("train_data/race_list.pkl", "wb"))
	race_list = cloudpickle.load(open("train_data/race_list.pkl", "rb"))
	print(race_list)
