from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

        self.hidden = self.initHidden()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = F.tanh(self.i2h(combined))
        output = F.tanh(self.i2o(combined))
        return output, hidden

    def predict(self, input):
        input = Variable(torch.FloatTensor(np.asarray(input)))
        output = self.forward(input)
        return output.data.numpy()[0]

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def train_step(target_commands, training_sample, learning_rate, hidden):
    recurrent_net.zero_grad()

    for i in range(2):
        output, hidden = recurrent_net(training_sample.unsqueeze(0), hidden)
        print("output: ", i, output)
        print("hidden: ", i, hidden)

    return output

def backprop(output, target_commands):
    loss = criterion(output, target_commands)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in recurrent_net.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return loss.data[0]

def split_data(data_tensor, n_frames):
    data_tensor = torch.cat((torch.zeros(n_frames-1, len(data_tensor[0])), \
        data_tensor), 0)
    train_samples = []
    for i in range(n_frames, len(data_tensor)):
        train_samples.append(data_tensor[i-n_frames:i])
    return train_samples

def train(data_tensor, target_tensor, learning_rate=0.005, print_every=100, epochs=1):
    total_loss = 0
    hidden = recurrent_net.initHidden()
    for _ in range(epochs):
        for i in range(len(data_tensor)):
            outputs = []
            for j in range(10):
                output = train_step(Variable(target_tensor[j]), \
                    Variable(data_tensor[j]), learning_rate, hidden)
                outputs.append(output)
            outputs = torch.cat(outputs, 0)
            target_commands = target_tensor[i:i+10]
            loss = backprop(outputs, Variable(target_commands))
            total_loss += loss
            if i % print_every == 0:
                print('loss:', total_loss/print_every)
                total_loss = 0

def warmup(data_tensor):
    hidden = recurrent_net.initHidden()
    for i in range(len(data_tensor)):
        recurrent_net.zero_grad()
        output, hidden = recurrent_net(Variable(data_tensor[i].unsqueeze(0)), hidden)

training = False

learning_rate = 0.002
n_frames = 1
n_hidden = 128
n_input = 22
n_output = 3

recurrent_net = RNN(n_input, n_hidden, n_output)

if training:
    criterion = nn.MSELoss()
    warmup(train_data.data_tensor[:2])
    train(train_data.data_tensor[2:], train_data.target_tensor[2:], learning_rate=learning_rate)
    torch.save(recurrent_net.state_dict(), 'models/rnn')
else:
    recurrent_net.load_state_dict(torch.load('models/rnn'))
