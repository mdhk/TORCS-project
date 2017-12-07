from load_data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import numpy as np
import os
import random

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.drop = nn.Dropout(0.5)

        self.encoder = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.hidden = self.init_hidden()

    def load(self, network):
        self.load_state_dict(torch.load(network))

    def forward(self, input):
        print("0")
        print(input)
        pef = self.encoder(input.view(1,-1))
        print("1\n", pef)
        output, hidden = self.rnn(pef.view(1, 1, -1), self.hidden)
        print("2")
        output = self.decoder(output)
        print("3")
        return output, hidden
    # def forward(self, input):
    #     print("0")
    #     emb = self.drop(self.encoder(input))
    #     print("1")
    #     output, hidden = self.rnn(emb, hidden)
    #     print("2")
    #     output = self.decoder(output.view(1, -1))
    #     print("3")
    #     return output, hidden

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
        race_list = dill.load(open("train_data/race_list.pkl", "rb"))
        i = random.randint(0, len(race_list))
        print(len(race_list[i].data_tensor), len(race_list[i].target_tensor),\
              len(race_list[i].data_tensor) * len(race_list[i].target_tensor))

        return race_list[i].data_tensor, race_list[i].target_tensor

    def train(self, num_epochs, learning_rate):

        # Get current directory
        curdir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        losses = np.zeros(num_epochs)

        for epoch in range(num_epochs):

            data, targets = self.get_random_data(curdir)

            hidden = self.init_hidden()

            for line in range(len(data)):


                optimizer.zero_grad()
                outputs, hidden = self(data[line])
                print("HOOR?")
                loss = criterion(outputs, targets[line])
                loss.backward()
                optimizer.step()
                losses[epoch] += loss.data[0]


        torch.save(self.state_dict(), 'models/lstm' + str(N_epochs))
