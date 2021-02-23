import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn.utils.rnn as rnn_utils 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F



class PadSequence:
    def __call__(self, batch):

        x=[ix[0] for ix in batch]
        y=[iy[1] for iy in batch]
        y=torch.LongTensor(y)

        padded = rnn_utils.pad_sequence(x, batch_first=True)
        sorted_batch_lengths = [len(p) for p in padded]
        packed = rnn_utils.pack_padded_sequence(padded, sorted_batch_lengths, batch_first=True, enforce_sorted=False)
        return packed, y


"""
Create x and y:
"""
data_list = np.load('data_list.npy', allow_pickle=True)
x = data_list[:,0]
x = [torch.Tensor(arr) for arr in x]


y = data_list[:,1]
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
dataset = list(zip(x,y))


"""
Parameters:
"""
batch_size=64
num_epochs = 5
shuffle_dataset = True
random_seed = 42
feature_num = x[0].shape[1]


"""
Create train, validation and test sets:
"""
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



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=feature_num, hidden_size=256, batch_first=True)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 7)


    def forward(self, x):

        x, hid = self.lstm(x)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:,-1,:]
                
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        
        return x


"""
Train model:
"""
model = Model()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)


for epoch in range(num_epochs):
    model.train()
    total, correct = 0, 0
    for batch in train_loader:
        x_train = batch[0]
        y_train = batch[1]

        optimizer.zero_grad()
        y_train_pred = model(x_train)
        
        train_loss = loss(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()
        print('Epoch {}:  Train loss: {}'.format(epoch, train_loss.item()))
        
        _, y_train_pred = torch.max(y_train_pred.data, 1)
        total += y_train_pred.size(0)
        correct += (y_train_pred == y_train).sum().item()
    accuracy = 100 * correct/total
    print('Train Accuracy: {}%'.format('%.4f' % accuracy))

    
    """
    Validation part:
    """
    model.eval()
    total, correct = 0, 0
    for batch in valid_loader:
        x_valid = batch[0]
        y_valid = batch[1]
        y_valid_pred = model(x_valid)
        _, y_valid_pred = torch.max(y_valid_pred.data, 1)
        
        total += y_valid_pred.size(0)
        correct += (y_valid_pred == y_valid).sum().item()
    accuracy = 100 * correct/total
    print('Validation Accuracy: {}%\n'.format('%.4f' % accuracy))



"""
Test model:
"""
total, correct = 0, 0
with torch.no_grad():
    model.eval()
    for batch in test_loader:
        x_test = batch[0]
        y_test = batch[1]
        y_test_pred = model(x_test)
        _, y_test_pred = torch.max(y_test_pred.data, 1)
        
        total += y_test_pred.size(0)
        correct += (y_test_pred == y_test).sum().item()
    accuracy = 100 * correct/total
    print('Test Accuracy: {}%'.format('%.4f' % accuracy))
