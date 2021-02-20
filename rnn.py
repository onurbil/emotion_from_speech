import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder


data_list = np.load('data_list.npy', allow_pickle=True)


y = data_list[:,1]
y = [label.casefold() for label in y]
y = [l.replace('pleasant_surprised', 'pleasant_surprise') for l in y]
le = LabelEncoder()
le.fit(y)
y_cat = le.transform(y)
y = keras.utils.to_categorical(y_cat, len(set(y_cat)))


x = data_list[:,-1]
x = [arr.tolist() for arr in x]
x=x[:32]
y=y[:32]

x = tf.ragged.constant(x)
print(x.bounding_shape())
print(x.shape)
print(y.shape)

# For ragged tensor , get maximum sequence length
max_seq = x.bounding_shape()[-1]
print(max_seq)

model = tf.keras.Sequential([
    # Input Layer with shape = [Any,  maximum sequence length]                      
    tf.keras.layers.Input(shape=[None,max_seq], dtype=tf.float32, ragged=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(7, activation='softmax')
])

# CategoricalCrossentropy
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

for layer in model.layers:
    print(layer.output_shape)

model.summary()
history = model.fit(x, y, epochs=10)


####################################################

# Pytorch example:

import torch 
import torch.nn.utils.rnn as rnn_utils 
a = torch.Tensor([[1], [2], [3]]) 
b = torch.Tensor([[4], [5]]) 
c = torch.Tensor([[6]]) 
d = torch.Tensor([[7],[8],[9],[10]])
batch = [a,b,c,d]

# In a shape(3,1) as sequence x window size. In our case: sequence x 20
print(a.shape)

padded= rnn_utils.pad_sequence(batch, batch_first=True)
print('padded: \n', padded)
sorted_batch_lengths = [len(x) for x in padded]
print('sorted batch length: \n', sorted_batch_lengths)

packed = rnn_utils.pack_padded_sequence(padded, sorted_batch_lengths, batch_first=True, enforce_sorted=False)
print('packed: \n', packed)
lstm = torch.nn.LSTM(input_size=1, hidden_size=3, batch_first=True)
lstm(packed)