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

net.zero_grad()  # reset gradients
loss.backward()  # compute gradients

# update weights
learning_rate = 0.5
for f in net.parameters():
    # for each parameter, take a small step in the opposite dir of the gradient
    # sub_ substracts in-place
    f.data.sub_(f.grad.data * learning_rate)

new_out = net(x)
new_loss = criterion(new_out, target)

print("target:", target)
print("out:", out)
print("new out (should be closer to target):", new_out)

print("\nloss:", loss)
print("new loss (should be lower):", new_loss)
