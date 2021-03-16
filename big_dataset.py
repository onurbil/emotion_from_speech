import os
import numpy as np
from pathlib import Path
from paths import BIG_DATA_FOLDER
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
└─── Emotions/
│    └─── Angry
│    └─── Disgusted
│    └─── ...
│
└─── ...
"""


def calculate_features(data, sr, n_mfcc, order):
    mfcc = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
    # First order and second order and energy features.
    mfcc_delta = librosa.feature.delta(mfcc, order=order)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=order)
    energy = librosa.feature.rms(data)

    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, energy), axis=0)
    # Bring to sample x sequence x features for LSTM:
    features = features.T
    return features


def process_dataset():
    # Parameters:
    sampling_rate = 24414
    mfcc_number = 20

    # Data path:
    class_dir = os.listdir(BIG_DATA_FOLDER)

    data_list = []
    for folder in class_dir:

        cls_path = os.path.join(BIG_DATA_FOLDER, folder)
        cls = folder
        data_row = []
        for file in os.listdir(cls_path):
            if file.endswith('.wav'):

                file_path = os.path.join(cls_path, file)
                data, sr = librosa.load(file_path, sr=None)

                if sr != sampling_rate:
                    data, sr = librosa.load(file_path, sr=sampling_rate)

                features = calculate_features(data, sr, n_mfcc=20, order=1)
                data_list.append([features, cls])


    data_list = np.array(data_list, dtype=object)
    np.save('big_data_list.npy', data_list)


if __name__ == '__main__':
    process_dataset()

            















