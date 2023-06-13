# Preprocess the dataset
# there are 1000 audios in the dataset, each audio has 30 seconds, 22050 sample rate
# we cut each audio into 10 pieces, each piece has 3 seconds, 22050 sample rate
# then use various methods to extract features from each piece
# finally, we get 10000 pieces of data

import os
from datasets import load_dataset, Dataset, load_from_disk, ClassLabel
import librosa
import numpy as np
from tqdm import tqdm


def load_gtzan():
    '''
    Load GTZAN dataset
    Returns:
        ds: dataset
    '''
    return load_dataset('marsyas/gtzan', split='train')


def split(dataset, split_num):
    min_length = min([len(x['audio']['array']) for x in dataset])
    print('Minimum length: ', min_length)
    piece_length = min_length // split_num
    print('Piece length: ', piece_length)
    audios = []
    genres = []
    for i in tqdm(range(len(dataset)), desc="Cut audio"):
        audio = dataset[i]['audio']['array'][:min_length]
        for j in range(split_num):
            audios.append({'array': np.array(audio[j*piece_length:(j+1)*piece_length])})
            genres.append(dataset[i]['genre'])
    return Dataset.from_dict({'audio': audios, 'genre': genres}).cast_column('genre', ClassLabel(num_classes=10))


def std_proc(dataset):
    hop_length = 512  # change with hdc dimensions
    n_fft = 2048
    n_mfcc = 13
    min_length = min([len(x['audio']['array']) // hop_length for x in dataset])
    print('Minimum time series length: ', min_length)
    audios = []
    genres = []
    for i in tqdm(range(len(dataset)), desc="Extract features"):
        audio = np.array(dataset[i]['audio']['array'])
        # mfcc = librosa.feature.mfcc(y=audio, sr=22050, hop_length=hop_length, n_fft=n_fft, n_mfcc=n_mfcc)
        # spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050, hop_length=hop_length, n_fft=n_fft)
        # chroma = librosa.feature.chroma_stft(y=audio, sr=22050, hop_length=hop_length, n_fft=n_fft)
        # spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050, hop_length=hop_length, n_fft=n_fft)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=22050, hop_length=hop_length, n_fft=n_fft)
        # num_features = mfcc.shape[0] + spectral_centroid.shape[0] + chroma.shape[0] + spectral_contrast.shape[0]
        # num_features = mfcc.shape[0]+ chroma.shape[0]
        # concatenate features
        # features = np.zeros((num_features, min_length))
        # features[:mfcc.shape[0], :] = mfcc[:, :min_length]
        # features[mfcc.shape[0]:mfcc.shape[0]+chroma.shape[0], :] = chroma[:, :min_length]
        # features[mfcc.shape[0]:mfcc.shape[0]+spectral_centroid.shape[0], :] = spectral_centroid[:, :min_length]
        # features[mfcc.shape[0]+spectral_centroid.shape[0]:mfcc.shape[0]+spectral_centroid.shape[0]+chroma.shape[0], :] = chroma[:, :min_length]
        # features[mfcc.shape[0]+spectral_centroid.shape[0]+chroma.shape[0]:, :] = spectral_contrast[:, :min_length]
        # features = mel_spectrogram
        audios.append(mel_spectrogram)
        genres.append(dataset[i]['genre'])
    return Dataset.from_dict({'features': audios, 'genre': genres}).cast_column('genre', ClassLabel(num_classes=10))



# get GTZAN and keep 30 second intervals
def standard():
    if not os.path.exists('gtzan_standard'):
        print('Processing GTZAN dataset...')
        dataset = load_gtzan()
        print('After loading', dataset)
        dataset = std_proc(dataset)
        print('After features extraction', dataset)
        dataset.save_to_disk('gtzan_standard')
    else:
        print('Loading processed GTZAN dataset...')
        dataset = load_from_disk('gtzan_standard')
    print(dataset)
    print('Done!')
    return dataset

# get GTZAN and split into 3 second intervals
def split_pieces(num_pieces):
    if not os.path.exists('gtzan_10pieces'):
        print('Processing GTZAN dataset...')
        dataset = load_gtzan()
        print('After loading', dataset)
        dataset = split(dataset, num_pieces)
        print('After splitting', dataset)
        dataset = std_proc(dataset)
        print('After features extraction', dataset)
        dataset.save_to_disk('gtzan_10pieces')
    else:
        print('Loading processed GTZAN dataset...')
        dataset = load_from_disk('gtzan_10pieces')
    print(dataset)
    print('Done!')
    return dataset


if __name__ == '__main__':
    standard()
