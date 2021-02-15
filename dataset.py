import os
import numpy as np
from scipy.io import wavfile
from pathlib import Path


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

DATA_FOLDER = Path('../TESS Toronto emotional speech set data')

class_dir = os.listdir(DATA_FOLDER)

wav_array = []
broken_files = []
for folder in class_dir:
    
    cls =  folder.split('_',1)[-1]
    cls_path = os.path.join(DATA_FOLDER, folder)
    
    data_row = []
    for file in os.listdir(cls_path):
        if file.endswith('.wav'):
            
            file_path = os.path.join(cls_path, file)
            try:
                sample_rate, data = wavfile.read(file_path)
            except: 
                broken_files.append(file_path)
                continue
                
            wav_array.append([file,cls, sample_rate, data])
            

"""
wav_array: [file_name, class, sample_rate, data_array]
"""
wav_array = np.array(wav_array,dtype=object)
print(wav_array)
print(wav_array.shape)
print(broken_files)

            
            















