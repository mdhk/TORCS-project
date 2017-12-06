import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import rnn_bm as rnnw

race_list = dill.load(open("train_data/race_list.pkl", "rb"))

X, Y = race_list[0].data_tensor, race_list[0].target_tensor

net = rnnw.RNN(input_size=len(X[1]),\
               hidden_size=51, \
               output_size=len(Y[1]),
               n_layers=2)

epochs = 100
network = "models/lstm" + str(epochs)
# net.load(network)
if __name__ == "__main__":
    net.train(epochs, 0.1)
