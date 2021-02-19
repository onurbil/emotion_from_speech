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

data_list = []
for folder in class_dir:
    
    cls =  folder.split('_',1)[-1]
    cls_path = os.path.join(DATA_FOLDER, folder)
    
    data_row = []
    for file in os.listdir(cls_path):
        if file.endswith('.wav'):
            
            file_path = os.path.join(cls_path, file)
            data, sr = librosa.load(file_path, sr=None)            
            
            if sr != sampling_rate:
                # This loop is for OAF_food_fear.wav, which has sr of 96000 Hz.
                data, sr = librosa.load(file_path, sr=sampling_rate)
                
            mfccs = librosa.feature.mfcc(data, sr=sr, n_mfcc=20)
            # For later first order and second order derivatives. 
            # Energy not implemented. Chroma for music bc of harmonic, melodic features.
            # mfcc_delta = librosa.feature.delta(mfcc, order=1)
            # mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            data_list.append([file,cls,mfccs])

                
print(data_list)
print(len(data_list))



            















