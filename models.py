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


class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()

        self.models = models

    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            # print(f'\tpred: {pred.shape}')
            predictions.append(model(x))

        predictions_array = torch.sum(torch.stack(predictions), 0) / len(self.models)
        # print('predictions: ', predictions_array.shape)
        return predictions_array


def load_gru(file_path, model_args, feature_num=None, classes_num=None):
    if feature_num is not None:
        model_args['feature_num'] = feature_num
    if classes_num is not None:
        model_args['classes'] = classes_num

    model = GRU(**model_args)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model


if __name__ == '__main__':
    import os
    import training

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print('Current device:', torch.cuda.get_device_name(device))
    else:
        print('Failed to find GPU. Will use CPU.')
        device = 'cpu'

    dataset_path = os.path.join('big_data_list.npy')
    train_loader, valid_loader, test_loader, feature_num, le = training.load_dataset(dataset_path, 64, True, random_seed=42)
    classes_num = len(le.classes_)

    model_defs = [
        ('./models/BiGRU_BigData_lay-4_hs-256.pt', 4, 256, 128),
        ('./models/BiGRU_BigData_lay-4_hs-512.pt', 4, 512, 128),
        ('./models/BiGRU_BigData-Augmented_lay-4_hs-512.pt', 4, 512, 128),
        ('./models/BiGRU_BigData-Augmented_lay-8_hs-512.pt', 8, 512, 128),
    ]

    models = []
    for file_path, num_layers, hidden_size, linear_size in model_defs:
        model_args = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'linear_size': linear_size,
            'bidirectional': True,
        }
        model = load_gru(file_path, model_args, feature_num, classes_num)
        model.to(device)
        accuracy, classes_accuracies = training.test_model(model, test_loader, len(le.classes_), le, device)
        print('\tmodel accuracy', accuracy)
        print('\tmodel classes_accuracies', classes_accuracies)

        models.append(model)

    ensemble = Ensemble(models)
    accuracy, classes_accuracies = training.test_model(ensemble, test_loader, len(le.classes_), le, device)
    print('accuracy', accuracy)
    print('classes_accuracies', classes_accuracies)
