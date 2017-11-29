import numpy as np
import pandas as pd

def read_data(*csv_filepaths):
	"""
	Reads data from csv files, returns X and Y as numpy matrices.
	"""
	X_data = []
	Y_data = []
	for csv_file in csv_filepaths:
		data = pd.read_csv(csv_file, sep=',', header=0)
		Y_data.append(np.matrix(data.values)[:,:3])
		X_data.append(np.matrix(data.values)[:,3:])
	Y = np.concatenate(Y_data, axis=0)
	X = np.concatenate(X_data, axis=0)
	return X, Y

X, Y = read_data('train_data/aalborg.csv', 'train_data/alpine-1.csv', 'train_data/f-speedway.csv')
print(X[:10])



		
