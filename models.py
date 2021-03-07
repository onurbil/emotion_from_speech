import numpy as np

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class RecurrentModel(torch.nn.Module):
    def __init__(self, recurrent, linear_input_size, linear_size, classes):
        super(RecurrentModel, self).__init__()

        self.recurrent = recurrent
        self.fc1 = torch.nn.Linear(linear_input_size, linear_size)
        self.fc2 = torch.nn.Linear(linear_size, classes)

    def forward(self, x):
        x, hid = self.recurrent(x)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:, -1, :]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=1)

        return x


class LSTM(RecurrentModel):
    def __init__(self, num_layers, feature_num, hidden_size, linear_size, classes, bidirectional=False):
        recurrent = torch.nn.LSTM(num_layers=num_layers, input_size=feature_num, hidden_size=hidden_size,
                                  bidirectional=bidirectional, batch_first=True)
        linear_input_size = hidden_size * (2 if bidirectional else 1)

        super(LSTM, self).__init__(recurrent, linear_input_size, linear_size, classes)


class GRU(RecurrentModel):
    def __init__(self, num_layers, feature_num, hidden_size, linear_size, classes, bidirectional=False):
        recurrent = torch.nn.GRU(num_layers=num_layers, input_size=feature_num, hidden_size=hidden_size,
                                 bidirectional=bidirectional, batch_first=True)
        linear_input_size = hidden_size * (2 if bidirectional else 1)

        super(GRU, self).__init__(recurrent, linear_input_size, linear_size, classes)


class VanillaRNN(RecurrentModel):
    def __init__(self, num_layers, feature_num, hidden_size, linear_size, classes, bidirectional=False):
        recurrent = torch.nn.RNN(num_layers=num_layers, input_size=feature_num, hidden_size=hidden_size,
                                 bidirectional=bidirectional, batch_first=True)
        linear_input_size = hidden_size * (2 if bidirectional else 1)

        super(VanillaRNN, self).__init__(recurrent, linear_input_size, linear_size, classes)
