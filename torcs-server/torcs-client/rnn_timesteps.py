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

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 1)
        self.hidden = F.tanh(self.i2h(combined))
        output = F.tanh(self.i2o(combined))
        return output

    def predict(self, input):
        input = Variable(torch.FloatTensor(np.asarray(input)))
        output = self.forward(input)
        return output.data.numpy()[0]

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train_step(target_commands, time_frame_data, learning_rate):
    recurrent_net.hidden = recurrent_net.initHidden()

    recurrent_net.zero_grad()

    for i in range(time_frame_data.size()[0]):
        output = recurrent_net(time_frame_data[i].unsqueeze(0))

    loss = criterion(output, target_commands)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in recurrent_net.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]


def split_data(data_tensor, n_frames):
    data_tensor = torch.cat((torch.zeros(n_frames-1, len(data_tensor[0])), \
        data_tensor), 0)
    train_samples = []
    for i in range(n_frames, len(data_tensor)):
        train_samples.append(data_tensor[i-n_frames:i])
    return train_samples

def train(training_samples, target_tensor, learning_rate=0.005, print_every=100, epochs=1):
    total_loss = 0
    for _ in range(epochs):
        for i in range(len(training_samples)):
            output, loss = train_step(Variable(target_tensor[i]), \
                Variable(training_samples[i]), learning_rate)
            total_loss += loss
            if i % print_every == 0:
                print('loss:', total_loss/print_every)
                total_loss = 0

training = True

learning_rate = 0.5
n_frames = 100
n_hidden = 128
n_input = 22
n_output = 3

recurrent_net = RNN(n_input, n_hidden, n_output)

if training:
    criterion = nn.MSELoss()
    training_samples = split_data(train_data.data_tensor, n_frames)
    train(training_samples, train_data.target_tensor, learning_rate=learning_rate)
    torch.save(recurrent_net.state_dict(), 'models/rnn')
else:
    recurrent_net.load_state_dict(torch.load('models/rnn'))
