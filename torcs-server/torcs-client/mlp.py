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

net = MLP()

### UNCOMMENT FOR LOADING A TRAINED MODEL
net.load_state_dict(torch.load('models/mlp10'))

# UNCOMMENT FOR TRAINING
# N_epochs = 10
# x = Variable(X)
# out = net(x)
# target = Variable(Y)
#
# ## choose a loss function, e.g. nn.MSELoss() or nn.NLLLoss()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
#
# for epoch in range(1, N_epochs + 1):
#     optimizer.zero_grad()
#     loss = criterion(out, target)
#
#     # print every 10th epoch
#     if epoch == 1 or epoch % 10 == 0:
#         print('epoch', epoch, ', loss:', loss)
#
#     loss.backward()
#     optimizer.step()
#
#     out = net(x)
#
# torch.save(net.state_dict(), 'models/mlp' + str(N_epochs))
