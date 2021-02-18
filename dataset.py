import os
import numpy as np
from pathlib import Path
from paths import DATA_FOLDER
import librosa

"""
Folder structure:
Folder structure must be as following:
<folder_name> is the parent folder of github repository and may have any name.

<folder_name>
└─── emotion_from_speech/
│    └─── dataset.py
│    └─── ...
│
└─── TESS Toronto emotional speech set data/
│    └─── OAF_Fear
│    └─── OAF_Pleasant_surprise
│    └─── ...
│
└─── ...
"""

# Parameters:
sampling_rate=24414
mfcc_number = 20

# Data path:
# DATA_FOLDER = Path('../TESS Toronto emotional speech set data')
class_dir = os.listdir(DATA_FOLDER)

file_class = []
features = np.zeros((0,mfcc_number,0))
for folder in class_dir:
    
    cls =  folder.split('_',1)[-1]
    cls_path = os.path.join(DATA_FOLDER, folder)
    
    data_row = []
    for file in os.listdir(cls_path):
        if file.endswith('.wav'):
            
            file_path = os.path.join(cls_path, file)
            data, sr = librosa.load(file_path, sr=None)            
            file_class.append([file,cls])
            
            if sr != sampling_rate:
                # This loop is for OAF_food_fear.wav, which has sr of 96000 Hz.
                data, sr = librosa.load(file_path, sr=sampling_rate)
                
            mfccs = librosa.feature.mfcc(data, sr=sr, n_mfcc=20)
            # Concatenate:
            if features.shape[-1]>=mfccs.shape[-1]:
                padded = np.zeros((features.shape[1],features.shape[-1]))
                padded[:,:mfccs.shape[-1]] = mfccs
                mfccs = padded
            else:
                padded = np.zeros((features.shape[0],features.shape[1],mfccs.shape[-1]))
                padded[:,:,:features.shape[-1]] = features
                features = padded
            mfccs = np.expand_dims(mfccs, axis=0)
            features = np.concatenate((features,mfccs),axis=0)
                
file_class = np.array(file_class)

print(features)
print(features.shape)

print(file_class)
print(file_class.shape)

            















