import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import numpy as np
import networks as nw

race_list = dill.load(open('train_data/race_list.pkl', 'rb'))
X = torch.FloatTensor(np.concatenate([tensordataset.data_tensor.numpy() \
    for tensordataset in race_list], axis=0))
Y = torch.FloatTensor(np.concatenate([tensordataset.target_tensor.numpy() \
    for tensordataset in race_list], axis=0))
train_data = TensorDataset(X, Y)
net = nw.MLP(train_data.data_tensor, train_data.target_tensor, N_hidden=50)

epochs = 500
#network = "models/mlp" + str(epochs)
#net.load(network)
if __name__ == "__main__":
    net.train(epochs)
