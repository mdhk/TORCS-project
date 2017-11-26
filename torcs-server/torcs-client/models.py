from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

X = train_data.data_tensor
Y = train_data.target_tensor
N_in = len(X[1])
N_out = len(Y[1])
N_hidden = 95

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

        def predict(self, x):
            x = Variable(torch.FloatTensor(np.asarray(x)))
            y = self.forward(x).data.numpy()[0]
            return y

## TODO RNN
# class RNN(nn.Module):
#     def __init__(self, input, hidden_size, output_size):
#         super(RNN, self).__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax()
#
#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden
#
#     def initHidden(self):
#         return Variable(torch.zeros(1, self.hidden_size))
#
#     def lineToTensor(line):
#         tensor = torch.zeros(len(line), 1, N_input)
#         for li, letter in enumerate(line):
#             tensor[li][0][letterToIndex(letter)] = 1
#         return tensor

net = MLP()

### UNCOMMENT FOR LOADING A TRAINED MODEL
net.load_state_dict(torch.load('models/mlp100'))

# UNCOMMENT FOR TRAINING
# N_epochs = 100
# x = Variable(X)
# out = net(x)
# target = Variable(Y)
#
# ## choose a loss function, e.g. nn.MSELoss() or nn.NLLLoss()
# criterion = nn.MSELoss()
#
# for epoch in range(N_epochs):
#     loss = criterion(out, target)
#     if epoch % 10 == 0:
#         print('epoch', epoch + 1, ', loss:', loss)
#     net.zero_grad()  # reset gradients
#     loss.backward()  # compute gradients
#     # update weights
#     learning_rate = 0.02
#     for f in net.parameters():
#         # for each parameter, take a small step in the opposite dir of the gradient
#         # sub_ substracts in-place
#         f.data.sub_(f.grad.data * learning_rate)
#
#     out = net(x)
#
# torch.save(net.state_dict(), 'models/mlp100')
