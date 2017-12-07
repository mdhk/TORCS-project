import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import numpy as np
import networks as nw

race_list = cloudpickle.load(open('train_data/race_list.pkl', 'rb'))
data = np.concatenate([tensordataset.data_tensor.numpy() \
        for tensordataset in race_list], axis=0)
X = torch.FloatTensor(data)
Y = torch.FloatTensor(np.concatenate([tensordataset.target_tensor.numpy() \
    for tensordataset in race_list], axis=0))
train_data = TensorDataset(X, Y)
net = nw.MLP(train_data.data_tensor, train_data.target_tensor, N_hidden=50)

epochs = 50
#network = "models/mlp" + str(epochs)
#net.load(network)
if __name__ == "__main__":
    net.train(epochs)
    data_means = np.mean(data, axis=0)
    data_stds = np.std(data, axis=0)
    stats = {'means': data_means, 'stds': data_stds}
    cloudpickle.dump(stats, open('train_data/stats.pkl', 'wb'))
