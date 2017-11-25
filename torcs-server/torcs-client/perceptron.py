from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

X = train_data.data_tensor
Y = train_data.target_tensor
N_in = len(X[1])
N_out = len(Y[1])
N_hidden = 30
N_epochs = 500

class Perceptron(nn.Module):

    def __init__(self):
        super(ExampleNN, self).__init__()

        # an affine operation: y = Wx + b
        self.linear = nn.Linear(22, 3)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.hidden = nn.Linear(N_in, N_hidden)
                self.hidden2 = nn.Linear(N_hidden, N_hidden)
                self.out   = nn.Linear(N_hidden, N_out)

            def forward(self, x):
                x = F.relu(self.hidden(x))
                x = F.tanh(self.hidden2(x))
                x = self.out(x)
                return x

net = MLP()
x = Variable(X)
out = net(x)
target = Variable(Y)

criterion = nn.MSELoss()

for epoch in range(N_epochs):
    loss = criterion(out, target)
    print('epoch', epoch + 1, ', loss:', loss)
    net.zero_grad()  # reset gradients
    loss.backward()  # compute gradients
    # update weights
    learning_rate = 0.01
    for f in net.parameters():
        # for each parameter, take a small step in the opposite dir of the gradient
        # sub_ substracts in-place
        f.data.sub_(f.grad.data * learning_rate)

    out = net(x)
