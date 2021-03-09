import os

from pathlib import Path
import numpy as np
import librosa
import soundfile
import tqdm

from paths import DATA_FOLDER
import dataset


def augment_audio_files(original, second, second_ratio, augmented_fade_out=0):
    augmented = np.zeros(original.shape)
    if original.size > second.size:
        start = 0
        second_copy_size = second.size
        while start < augmented.size:
            augmented[start:second_copy_size] = (original[start:second_copy_size] * (1 - second_ratio)
                                                 + second[:second_copy_size] * second_ratio)
            start += second.size
            second_copy_size = min(original.size - start, second.size)
    else:
        augmented = original * (1 - second_ratio) + second[:original.size] * second_ratio

    if augmented_fade_out > 0:
        for n in range(1, augmented_fade_out + 1):
            augmented[-n] *= n / (augmented_fade_out + 1)

    return augmented


def augment_and_save_files_from_dir(source_dir, target_dir, augmentation_audio, augmentation_rate, sampling_rate,
                                    augmentation_fade_out=0):
    class_dirs = os.listdir(source_dir)

    for folder in tqdm.tqdm(class_dirs):
        target_folder = os.path.join(target_dir, folder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        cls_path = os.path.join(source_dir, folder)

        for file in os.listdir(cls_path):
            if not file.endswith('.wav'):
                continue

            file_path = os.path.join(cls_path, file)
            data, sampling_rate = librosa.load(file_path, sr=sampling_rate)

            augmented_audio = augment_audio_files(data, augmentation_audio, augmentation_rate, augmentation_fade_out)
            target_path = os.path.join(target_folder, file)
            soundfile.write(target_path, augmented_audio, sampling_rate, subtype='PCM_16')


def augment_and_process_files_from_dir(source_dir, data_filename, augmentation_audio, augmentation_rate, sampling_rate,
                                       augmentation_fade_out=0):
    class_dirs = os.listdir(source_dir)

    data_list = []
    for folder in tqdm.tqdm(class_dirs):
        cls_path = os.path.join(source_dir, folder)

        for file in os.listdir(cls_path):
            if not file.endswith('.wav'):
                continue

            cls = Path(file).stem.split('_', 2)[-1]
            file_path = os.path.join(cls_path, file)
            data, sampling_rate = librosa.load(file_path, sr=sampling_rate)

            augmented_audio = augment_audio_files(data, augmentation_audio, augmentation_rate, augmentation_fade_out)

            features = dataset.calculate_features(augmented_audio, sampling_rate, n_mfcc=20, order=1)
            data_list.append([features, cls])

    print(data_list)
    data_list = np.array(data_list, dtype=object)
    np.save(data_filename, data_list)


def load_augmentation(file_path, sampling_rate):
    augmentation, sr = librosa.load(file_path, sr=sampling_rate)
    return augmentation


sampling_rate = 24414

augumentations = [
    ('Engine.wav', 'engine', .3),
    ('High Noise.wav', 'h_noise', .3),
    ('Lenas Piano.wav', 'piano', .4),
    ('Low Noise.wav', 'l_noise', .3),
    ('People Talking.wav', 'talking', .4),
]

for file, name, ratio in augumentations:
    augmentation = load_augmentation(os.path.join('augmentation_sounds', file), sampling_rate)
    augment_and_save_files_from_dir(DATA_FOLDER, f'../speech_augumented/{name} TESS Toronto',
                                    augmentation, ratio, sampling_rate, 100)

# augment_and_process_files_from_dir(DATA_FOLDER, 'data_list_piano-0.5.npy', augmentation, .5, sampling_rate)
