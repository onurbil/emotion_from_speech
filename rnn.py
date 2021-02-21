import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn.utils.rnn as rnn_utils 
from torch.nn.utils.rnn import pack_padded_sequence



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
dataset=list(zip(x,y))

loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=32, collate_fn=PadSequence())



lstm = torch.nn.LSTM(input_size=20, hidden_size=3, batch_first=True)
for d in loader:
    x=d[0]
    y=d[1]
    print(x)
    print(y)
    lstm(x)

