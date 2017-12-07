import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""
Geen uitvoerende code binnen dit bestand!
Alles binnen classes schrijven zodat je het kan importeren zonder dat er
iets gebeurt.
"""

class neural_net(nn.Module):
    """
    The standard framework for our neural nets.
    contains 1 hidden layer?
    """
    def __init__(self, X, Y, N_hidden):
        super(neural_net, self).__init__()
        self.X = X
        self.Y = Y

        self.N_in = len(self.X[1])
        self.N_out = len(self.Y[1])
        self.N_hidden = N_hidden

        self.hidden = nn.Linear(self.N_in, self.N_hidden)
        self.hidden2 = nn.Linear(self.N_hidden, self.N_hidden)
        self.out = nn.Linear(self.N_hidden, self.N_out)

    def load(self, network):
        self.load_state_dict(torch.load(network))

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.tanh(self.hidden2(x))
        x = self.out(x)
        return x

    def predict(self, x):
        x = Variable(torch.FloatTensor(np.asarray(x)))
        y = self.forward(x).data.numpy()[0]
        return y


class MLP(neural_net):
    """
    A normal multi layer perceptron. Might break because I restylized it.
    Contains one hidden layer
    """
    def train(self, N_epochs):

        # self.load_state_dict(torch.load('models/mlp10'))

        x = Variable(self.X)
        out = self(x)
        target = Variable(self.Y)
        print(out.size(), target.size())

        ## choose a loss function, e.g. nn.MSELoss() or nn.NLLLoss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        for epoch in range(1, N_epochs + 1):
            optimizer.zero_grad()
            loss = criterion(out, target)

            # print every 10th epoch
            # if epoch == 1 or epoch % 10 == 0:
            print('epoch', epoch, ', loss:', loss)

            loss.backward()
            optimizer.step()

            out = self(x)

        torch.save(self.state_dict(), 'models/mlp' + str(N_epochs))
