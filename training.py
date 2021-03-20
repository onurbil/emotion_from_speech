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
from sklearn.model_selection import train_test_split


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

    train_indices, val_indices, test_indices = indices[:round(len(indices)*0.7143)], indices[round(len(indices)*0.7143):round(len(indices)*0.7858)], indices[round(len(indices)*0.7858):]

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

    print(f'Dataset loaded - train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')
    print(f'\tfeature_num: {feature_num}, classes_num: {len(le.classes_)}')
    return train_loader, valid_loader, test_loader, feature_num, le


def load_augmented_dataset(train_dataset_paths, test_dataset_path, batch_size, shuffle_dataset=True, random_seed=42):
    test_data_list = np.load(test_dataset_path, allow_pickle=True)
    x = []
    ys = []
    data_list_size = None
    for path in train_dataset_paths:
        data_list = np.load(path, allow_pickle=True)
        x += [torch.Tensor(arr) for arr in data_list[:, 0]]
        ys.append(data_list[:, 1])

        if data_list_size is None:
            data_list_size = data_list.shape[0]
        elif data_list_size != data_list.shape[0]:
            raise Exception(f'Datasets are of different sizes: {data_list_size} and {data_list.shape[0]}')

    y = np.concatenate(ys)

    test_x = [torch.Tensor(arr) for arr in test_data_list[:, 0]]
    test_y = test_data_list[:, 1]

    le = LabelEncoder()
    le.fit(test_y)
    y = le.transform(y)
    test_y = le.transform(test_y)
    dataset = list(zip(x, y))
    # test_dataset = list(zip(test_x, test_y))

    indices = list(range(len(test_x)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_end_index = round(len(indices)*0.7143)
    test_start_index = round(len(indices)*0.7858)
    train_indices, val_indices, test_indices = indices[:train_end_index], indices[train_end_index:test_start_index], indices[test_start_index:]

    train_ds = np.random.randint(0, len(train_dataset_paths) - 1, len(train_indices))
    train_indices = [index + ds * data_list_size for index, ds in zip(train_indices, train_ds)]
    val_ds = np.random.randint(0, len(train_dataset_paths) - 1, len(val_indices))
    val_indices = [index + ds * data_list_size for index, ds in zip(val_indices, val_ds)]
    test_ds = np.random.randint(0, len(train_dataset_paths) - 1, len(test_indices))
    test_indices = [index + ds * data_list_size for index, ds in zip(test_indices, test_ds)]

    print('data_list_size', data_list_size)
    print('np.min(train_ds)', np.min(train_ds), 'np.max(train_ds)', np.max(train_ds))
    print('np.min(val_ds)', np.min(val_ds), 'np.max(val_ds)', np.max(val_ds))
    print('np.min(test_ds)', np.min(test_ds), 'np.max(test_ds)', np.max(test_ds))

    print('np.max(train_indices)', np.max(train_indices))
    print('np.max(val_indices)', np.max(val_indices))
    print('np.max(test_indices)', np.max(test_indices))

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

    print(f'Augmented dataset loaded - train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')
    print(f'\tfeature_num: {feature_num}, classes_num: {len(le.classes_)}')
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


def plot_conf_matrix(y_pred, y_test, le, title=None):
    if title is None:
        title = 'Confusion matrix'

    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    ax.set_xticklabels([''] + le.classes_, rotation=90)
    ax.set_yticklabels([''] + le.classes_, rotation=0)
    plt.show()


def test_model(model, test_loader, classes, le, device, confusion_matrix_title=None, verbose=1):
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
            plot_conf_matrix(np.array(all_test_pred), np.array(all_labels), le, confusion_matrix_title)

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

        train_model(model, num_epochs, loss, optimizer, train_loader, valid_loader, device, patience, verbose=verbose)
        accuracy, class_acc = test_model(model, test_loader, classes, le, device, verbose=verbose)
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

    return np.mean(accuracies), accuracies, class_accuracies, model


def test_hyper_parameters_augmented(num_runs, clean_dataset_path, train_dataset_paths, test_dataset_path, batch_size,
                                    num_epochs, learning_rate, weight_decay,
                                    model_cls, model_args, device, patience, verbose=1):
    clean_train_loader, clan_valid_loader, clean_test_loader, clean_feature_num, clean_le = load_dataset(
        clean_dataset_path, batch_size)
    train_loader, valid_loader, test_loader, feature_num, le = load_augmented_dataset(train_dataset_paths,
                                                                                      test_dataset_path, batch_size)
    clean_classes = len(list(clean_le.classes_))
    classes = len(list(le.classes_))

    model_args['feature_num'] = feature_num
    model_args['classes'] = classes

    loss = torch.nn.CrossEntropyLoss()

    noise_accuracies = []
    noise_class_accuracies = []
    clean_accuracies = []
    clean_class_accuracies = []
    for run in range(num_runs):
        model = model_cls(**model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_model(model, num_epochs, loss, optimizer, train_loader, valid_loader, device, patience, verbose=verbose)
        noise_accuracy, noise_class_acc = test_model(model, test_loader, classes, le, device,
                                                     confusion_matrix_title='Confusion Matrix Augmented',
                                                     verbose=verbose)
        clean_accuracy, clean_class_acc = test_model(model, clean_test_loader, clean_classes, clean_le, device,
                                                     confusion_matrix_title='Confusion Matrix Clean', verbose=verbose)
        noise_accuracies.append(noise_accuracy)
        noise_class_accuracies.append(noise_class_acc)
        clean_accuracies.append(clean_accuracy)
        clean_class_accuracies.append(clean_class_acc)

    if verbose >= 1:
        print('Noise test results:')
        print_test_results(noise_accuracies, noise_class_accuracies)
        print('Clean test results:')
        print_test_results(clean_accuracies, clean_class_accuracies)

    return (np.mean(clean_accuracies), clean_accuracies, clean_class_accuracies,
            np.mean(noise_accuracies), noise_accuracies, noise_class_accuracies,
            model)


def print_test_results(accuracies, class_accuracies):
    print(
        'Test Accuracy	0-Accuracy	1-Accuracy	2-Accuracy	3-Accuracy	4-Accuracy	5-Accuracy	6-Accuracy')
    for run in range(len(accuracies)):
        print_string = f'{accuracies[run]}'
        class_acc = class_accuracies[run]
        for acc in class_acc:
            print_string += f'	{acc}'

        print(print_string)


if __name__ == '__main__':
    import os
    from training import load_dataset
    from training import train_model
    from training import test_hyper_parameters
    from models import LSTM, GRU, VanillaRNN

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print('Current device:', torch.cuda.get_device_name(device))
    else:
        print('Failed to find GPU. Will use CPU.')
        device = 'cpu'

    dataset_path = os.path.join('big_data_list.npy')
    train_dataset_paths = [
        dataset_path,
        os.path.join('big_data_list_engine-0.3.npy'),
        os.path.join('big_data_list_piano-0.4.npy'),
        os.path.join('big_data_list_l_noise-0.3.npy'),
    ]
    test_dataset_path = os.path.join('big_data_list_talking-0.4.npy')

    ## Parameters:
    num_runs = 2
    patience = 20
    batch_size = 64
    num_epochs = 1
    learning_rate = 0.001
    weight_decay = 0.01
    model_args = {
        'num_layers': 1,
        'hidden_size': 256,
        'linear_size': 128,
        'bidirectional': True,
    }

    results = test_hyper_parameters_augmented(num_runs=num_runs,
                                              clean_dataset_path=dataset_path,
                                              train_dataset_paths=train_dataset_paths,
                                              test_dataset_path=test_dataset_path,
                                              batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              learning_rate=learning_rate,
                                              weight_decay=weight_decay,
                                              model_cls=LSTM,
                                              model_args=model_args,
                                              device=device, patience=patience,
                                              verbose=1)

    print(results[0])
    print(results[3])
