import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn.utils.rnn as rnn_utils 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.sampler import SubsetRandomSampler

from debugging_tools import *




def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor: Numpy array must begin from 0."""
    return np.eye(num_classes, dtype='uint8')[y]


class PadSequence:
    def __call__(self, batch):

        x=[ix[0] for ix in batch]
        y=[iy[1] for iy in batch]
        padded = rnn_utils.pad_sequence(x, batch_first=True)
        sorted_batch_lengths = [len(p) for p in padded]
        packed = rnn_utils.pack_padded_sequence(padded, sorted_batch_lengths, batch_first=True, enforce_sorted=False)
        return packed, y


data_list = np.load('data_list.npy', allow_pickle=True)
x = data_list[:,0]
x = [torch.Tensor(arr) for arr in x]



y = data_list[:,1]
le = LabelEncoder()
le.fit(y)
y_cat = le.transform(y)
y = to_categorical(y_cat, len(set(y_cat)))
dataset = list(zip(x,y))


"""
Parameters:
"""
batch_size=32
num_epochs = 1
shuffle_dataset = True
random_seed = 42



indices = list(range(len(dataset)))
if shuffle_dataset :
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


lstm = torch.nn.LSTM(input_size=20, hidden_size=10, batch_first=True)

for epoch in range(num_epochs):
    for batch in train_loader:
        x = batch[0]
        y = batch[1]
        print(x)
        print(y)
        lstm(x)


# Loss function missing, backward chain and updating weights missing...

