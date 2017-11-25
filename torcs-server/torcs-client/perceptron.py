from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ExampleNN(nn.Module):

    def __init__(self):
        super(ExampleNN, self).__init__()

        # an affine operation: y = Wx + b
        self.linear = nn.Linear(22, 3)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)  # output values are squashed between 0 and 1
        return x

net = ExampleNN()

x = Variable(train_data.data_tensor)
out = net(x)
target = Variable(train_data.target_tensor)

criterion = nn.MSELoss()
loss = criterion(out, target)

print("output:", out)
print("loss:", loss)
