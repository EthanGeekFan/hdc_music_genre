import numpy as np
import librosa
from hdc import HDC, hdc_preproc
from nn_model import Model
import get_data
import os
from datasets import load_dataset, load_from_disk, Dataset
import torch

"""
This function augmentates the training data but does not change the testing data
"""

# NOISE
def noise(data):
    # import pdb; pdb.set_trace()
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data
# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# get file name
HDC_DIM = 1024
split = True
num_split_pieces = 10
if not split:
    hdc_ds_path = f'augmented_gtzan_hdc_standard_{HDC_DIM}'
else:
    hdc_ds_path = f'augmented_gtzan_hdc_split_{num_split_pieces}_{HDC_DIM}'


# get data without features extraction
raw_data = get_data.load_gtzan()
ds = raw_data.train_test_split(test_size=0.2, seed=229, stratify_by_column='genre')
raw_train = ds['train']
raw_test = ds['test']

# extract features
data_mfcc = get_data.std_proc(raw_train)

import pdb; pdb.set_trace()


# hdc process

# reduce training and testing data




# if not os.path.exists(hdc_ds_path):
#     if split:
#         dataset = get_data.split_pieces(num_split_pieces)
#     else:
#         dataset = get_data.standard()
#     dataset.set_format('torch', columns=['features'])
#     dataset = hdc_preproc(dataset, HDC_DIM)
#     dataset.save_to_disk(hdc_ds_path)
# else:
#     dataset = load_from_disk(hdc_ds_path)
