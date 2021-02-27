import numpy as np

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class PadSequence:
    def __call__(self, batch):
        x = [ix[0] for ix in batch]
        y = [iy[1] for iy in batch]
        y = torch.LongTensor(y)

        padded = rnn_utils.pad_sequence(x, batch_first=True)
        sorted_batch_lengths = [len(p) for p in padded]
        packed = rnn_utils.pack_padded_sequence(padded, sorted_batch_lengths, batch_first=True, enforce_sorted=False)
        return packed, y


def load_dataset(dataset_path, batch_size, shuffle_dataset=True, random_seed=42):
    data_list = np.load(dataset_path, allow_pickle=True)
    x = data_list[:, 0]
    x = [torch.Tensor(arr) for arr in x]

    y = data_list[:, 1]
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    dataset = list(zip(x, y))

    indices = list(range(len(dataset)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:2000], indices[2000:2200], indices[2200:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=PadSequence())
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=valid_sampler, collate_fn=PadSequence())
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler, collate_fn=PadSequence())

    feature_num = x[0].shape[1]
    classes = len(list(le.classes_))
    return train_loader, valid_loader, test_loader, feature_num, classes


def train_model(model, num_epochs, loss, optimizer, train_loader, valid_loader, device, verbose=1):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total, correct = 0, 0
        for batch in train_loader:
            x_train = batch[0].to(device)
            y_train = batch[1].to(device)

            optimizer.zero_grad()
            y_train_pred = model(x_train)

            train_loss = loss(y_train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            if verbose >= 2:
                print('Epoch {}:  Train loss: {}'.format(epoch, train_loss.item()))

            _, y_train_pred = torch.max(y_train_pred.data, 1)
            total += y_train_pred.size(0)
            correct += (y_train_pred == y_train).sum().item()
        if verbose >= 1:
            accuracy = 100 * correct / total
            print('Epoch {}:  Train Accuracy: {}%'.format(epoch, '%.4f' % accuracy))

        """
        Validation part:
        """
        model.eval()
        total, correct = 0, 0
        for batch in valid_loader:
            x_valid = batch[0].to(device)
            y_valid = batch[1].to(device)
            y_valid_pred = model(x_valid)
            _, y_valid_pred = torch.max(y_valid_pred.data, 1)

            total += y_valid_pred.size(0)
            correct += (y_valid_pred == y_valid).sum().item()
        if verbose >= 1:
            accuracy = 100 * correct / total
            print('Epoch {}:  Validation Accuracy: {}%\n'.format(epoch, '%.4f' % accuracy))


def test_model(model, test_loader, classes, device, verbose=1):
    model = model.to(device)

    class_correct = [0] * classes
    class_total = [0] * classes
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            x_test = batch[0].to(device)
            y_test = batch[1].to(device)
            y_test_pred = model(x_test)
            _, y_test_pred = torch.max(y_test_pred.data, 1)

            tp = (y_test_pred == y_test).squeeze()

            for i in range(y_test_pred.size(0)):
                label = y_test[i].item()
                class_correct[label] += tp[i].item()
                class_total[label] += 1

        if verbose >= 2:
            for i in range(classes):
                print('Test Accuracy of {} {}: {}'.format(i, le.inverse_transform([i])[0], 
                                                    100*class_correct[i]/class_total[i]))

        accuracy = sum(class_correct) / sum(class_total)
        if verbose >= 1:
            print('Test Accuracy: {}%'.format('%.4f' % (100 * accuracy)))

    return accuracy


def test_hyper_parameters(num_runs, dataset_path, batch_size,
                          num_epochs, learning_rate, weight_decay,
                          model_cls, model_args, device, verbose=1):
    train_loader, valid_loader, test_loader, feature_num, classes = load_dataset(dataset_path, batch_size)
    model_args['feature_num'] = feature_num
    model_args['classes'] = classes

    loss = torch.nn.CrossEntropyLoss()

    accuracies = []
    for run in range(num_runs):
        model = model_cls(**model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_model(model, num_epochs, loss, optimizer, train_loader, valid_loader, device, verbose)
        accuracy = test_model(model, test_loader, classes, device, verbose)
        accuracies.append(accuracy)

    return np.mean(accuracies)


if __name__ == '__main__':
    import os
    from training import test_hyper_parameters
    import torch

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print('Current device:', torch.cuda.get_device_name(device))
    else:
        print('Failed to find GPU. Will use CPU.')
        device = 'cpu'

    dataset_path = os.path.join('data_list.npy')


    class Model(torch.nn.Module):
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


    """
    Parameters:
    """
    batch_size = 64
    num_epochs = 3
    learning_rate = 0.001
    weight_decay = 0.01
    model_args = {
        'num_layers': 1,
        'hidden_size': 256,
        'linear_size': 128,
    }

    mean_accuracy = test_hyper_parameters(num_runs=3, dataset_path=dataset_path, batch_size=batch_size,
                                          num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                          model_cls=Model, model_args=model_args, device=device, verbose=0)

    print(mean_accuracy)
