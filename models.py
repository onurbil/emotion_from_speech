class LSTM(torch.nn.Module):
    def __init__(self, num_layers, feature_num, hidden_size, linear_size, classes):
        super(Model, self).__init__()

        self.lstm = torch.nn.LSTM(num_layers=num_layers, input_size=feature_num, hidden_size=hidden_size,
                                  batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, linear_size)
        self.fc2 = torch.nn.Linear(linear_size, classes)

    def forward(self, x):
        x, hid = self.lstm(x)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:, -1, :]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=1)

        return x