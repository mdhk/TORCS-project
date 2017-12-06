from load_data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, Dataloader
from torch.autograd import Variable
import numpy as np
import os
import random

X = train_data.data_tensor
Y = train_data.target_tensor

rnn = nn.RNN(input_size=len(X[1]), hidden_size=20, num_layers=2)

# input (seq_len, batch, input_size)
input = Variable(torch.randn(5, 3, 10))

# h_0 (num_layers * num_directions, batch, hidden_size)
h0 = Variable(torch.randn(2, 3, 20))

output, hn = rnn(input, h0)
# print(output)

# class OGDataset(Dataset):
#     def __init__(self, csv_file):
#         self.csv = pd.read_csv(csv_file)
#
#     def __len__(self):
#         return len(self.csv)
#
#     def __getitem(self, idx):
#
#         return sample

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables,
        to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)

    def get_random_data(self, curdir):
        name = "{}/bram_logs".format(curdir)
        length = len([name for name in os.listdir('.') if os.path.isfile(name)])
        i = random.randint(0, length)
        return all_races[i].data_tensor, all_races[i].target_tensor

    def train(self, num_epochs, learning_rate):
        self.train()

        # Get current directory
        curdir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        losses = np.zeros(num_epochs)

        for epoch in range(num_epochs):

            data, targets = self.get_random_data(curdir)

            optimizer.zero_grad()
            outputs, hidden = self(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses[epoch] += loss.data[0]
