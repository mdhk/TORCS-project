from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import networks as nw

net = nw.MLP(train_data.data_tensor, train_data.target_tensor, N_hidden=30)
net.train(1000)
