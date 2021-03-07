import numpy as np

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    return train_loader, valid_loader, test_loader, feature_num, le


def train_model(model, num_epochs, loss, optimizer, train_loader, valid_loader, device, patience, verbose=1):
    stopping = 0
    max_val_acc = 0
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

        """	
        Early stopping:
        """
        if max_val_acc < accuracy:
            max_val_acc = accuracy
            weights = copy.deepcopy(model.state_dict())
            stopping = 0
        else:
            stopping = stopping + 1

        if stopping == patience:
            print('Early stopping...')
            print('Restoring best weights')
            model.load_state_dict(weights)
            break


def plot_conf_matrix(y_pred, y_test, le):

    cm = confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    ax.set_xticklabels([''] + le.classes_,rotation=90)
    ax.set_yticklabels([''] + le.classes_, rotation=0)
    plt.show()


def test_model(model, test_loader, classes, le, device, verbose=1):
    model = model.to(device)

    class_correct = [0] * classes
    class_total = [0] * classes
    all_test_pred = []
    all_labels = []
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            x_test = batch[0].to(device)
            y_test = batch[1].to(device)
            y_test_pred = model(x_test)
            _, y_test_pred = torch.max(y_test_pred.data, 1)

            tp = (y_test_pred == y_test).squeeze()
            all_test_pred.extend(y_test_pred.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

            for i in range(y_test_pred.size(0)):
                label = y_test[i].item()
                class_correct[label] += tp[i].item()
                class_total[label] += 1

        classes_accuracies = []
        for i in range(classes):
            class_accuracy = class_correct[i] / class_total[i]
            classes_accuracies.append(class_accuracy)
            if verbose >= 2:
                print('Test Accuracy of {} {}: {}'.format(i, le.inverse_transform([i])[0], 100 * class_accuracy))


        accuracy = sum(class_correct) / sum(class_total)
        if verbose >= 1:
            print('Test Accuracy: {}%'.format('%.4f' % (100 * accuracy)))
            plot_conf_matrix(np.array(all_test_pred),np.array(all_labels),le)

    return accuracy, classes_accuracies


def test_hyper_parameters(num_runs, dataset_path, batch_size,
                          num_epochs, learning_rate, weight_decay,
                          model_cls, model_args, device, patience, verbose=1):
    train_loader, valid_loader, test_loader, feature_num, le = load_dataset(dataset_path, batch_size)
    classes = len(list(le.classes_))

    model_args['feature_num'] = feature_num
    model_args['classes'] = classes

    loss = torch.nn.CrossEntropyLoss()

    accuracies = []
    class_accuracies = []
    for run in range(num_runs):
        model = model_cls(**model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_model(model, num_epochs, loss, optimizer, train_loader, valid_loader, device, patience, verbose)
        accuracy, class_acc = test_model(model, test_loader, classes, le, device, verbose)
        accuracies.append(accuracy)
        class_accuracies.append(class_acc)

    if verbose >= 1:
        print(
            'Test Accuracy	0-Accuracy	1-Accuracy	2-Accuracy	3-Accuracy	4-Accuracy	5-Accuracy	6-Accuracy')
        for run in range(num_runs):
            print_string = f'{accuracies[run]}'
            class_acc = class_accuracies[run]
            for acc in class_acc:
                print_string += f'	{acc}'

            print(print_string)

    return np.mean(accuracies), accuracies, class_accuracies


if __name__ == '__main__':
    import os
    from training import load_dataset
    from training import train_model
    from training import test_hyper_parameters
    from models import LSTM

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print('Current device:', torch.cuda.get_device_name(device))
    else:
        print('Failed to find GPU. Will use CPU.')
        device = 'cpu'

    dataset_path = os.path.join('data_list.npy')

    ## Parameters:
    num_runs = 3
    patience = 20
    batch_size = 64
    num_epochs = 300
    learning_rate = 0.001
    weight_decay = 0.01
    model_args = {
        'num_layers': 1,
        'hidden_size': 256,
        'linear_size': 128,
    }

    mean_accuracy, accuracies, class_accuracies = test_hyper_parameters(num_runs=num_runs, dataset_path=dataset_path,
                                                                        batch_size=batch_size,
                                                                        num_epochs=num_epochs,
                                                                        learning_rate=learning_rate,
                                                                        weight_decay=weight_decay,
                                                                        model_cls=LSTM, model_args=model_args,
                                                                        device=device, patience=patience, verbose=1)

    print(mean_accuracy)
