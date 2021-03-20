import torch 
import os
import download
import dataset
from paths import WORKING_DIR
from models import LSTM, GRU, VanillaRNN
from training import load_dataset
from training import train_model
from training import test_hyper_parameters

# Download:
download.main()

# Process:
dataset_path = os.path.join(WORKING_DIR,'data_list.npy')
if not os.path.exists(dataset_path):
    print('Dataset in preprocessing...')
    dataset.process_dataset()

# Train:
print('Dataset in training...')


if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(device))
else:
    print('Failed to find GPU. Will use CPU.')
    device = 'cpu'


## Parameters:
num_runs = 1
patience = 20
batch_size = 64
num_epochs = 300
learning_rate = 0.0001
weight_decay = 0
model_cls = GRU
model_args = {
    'num_layers': 4,
    'hidden_size': 512,
    'linear_size': 128,
    'bidirectional': True,
}

results = test_hyper_parameters(num_runs=num_runs, dataset_path=dataset_path,
                                batch_size=batch_size,
                                num_epochs=num_epochs,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                model_cls=model_cls, model_args=model_args,
                                device=device, patience=patience, verbose=1)
print(results[0])


