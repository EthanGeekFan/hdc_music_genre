from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
import librosa
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pre
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

# torch.manual_seed(529)

torch.manual_seed(8263655087879647638)
# torch.manual_seed(5064919006405696560)
# print('Seed: ', seed)
print('Check seed: ', torch.initial_seed())

N = 4
print('N-gram: ', N)

HDC_DIM = 1024
print('HDC Dimension: ', HDC_DIM)

SCALE = 3
print('Scaling: ', SCALE)

class HDC:
    '''
    Use FHRR representation
    Use 12 chrome vectors to represent 12 pitches
    encode intensity with a shared base vector
    '''
    
    def __init__(self, dim, scaling=3):
        '''
        Initialize HDC encoder
        '''
        self.dim = dim
        self.timeseries_length = None
        self.scaling = scaling
        # choose device based on availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # NVIDIA GPU
            print('HDC Encoder Using CUDA GPU: ', torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple GPU
            print('HDC Encoder Using Apple GPU')
        else:
            self.device = torch.device('cpu') # CPU
            print('HDC Encoder Using CPU')
        self.pitches = torch.rand(12, HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        # self.intenisty_base = torch.rand(HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        # self.another_intensity_base = torch.rand(HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        self.intensity_bases = torch.rand(1, HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        self.scaling = scaling
        self.mfcc_feature_symbols = None
        self.mel_feature_symbols = None
        self.chroma_feature_symbols = None
        self.spec_feature_symbols = None
        # profile vectors accumulator
        self.profile_vectors = {}
        self.final_profile_vectors = {}


    def extract_mel(self, audio, n_mels=128):
        '''
        Returns:
            mel: mel spectrogram of the audio
        '''
        # audio = audio.to(self.device)
        # mel_spectrogram = T.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=n_mels).to(self.device)
        # mel = mel_spectrogram(audio)
        mel = librosa.feature.melspectrogram(y=audio.cpu().numpy(), sr=22050, n_fft=2048, hop_length=512, n_mels=128)
        mel = librosa.power_to_db(mel)
        # mel = np.log(mel + 1e-9)
        normalized = librosa.util.normalize(mel)
        # values to M levels between -1 and 1
        M = 10
        step = 2 / (M - 1)
        for i in range(M):
            # values -1 + step * i - step / 2 <= normalized <= -1 + step * i + step / 2 should be mapped to -1 + step * i
            normalized[(normalized >= -1 + step * i - step / 2) & (normalized <= -1 + step * i + step / 2)] = -1 + step * i
        mel = torch.tensor(normalized, device=self.device)
        return mel
    

    def extract_spectrogram(self, audio, n_fft=200):
        '''
        Returns:
            spectrogram: spectrogram of the audio
        '''
        audio = audio.to(self.device)
        # spectrogram = T.Spectrogram(n_fft=200, hop_length=512).to(self.device)
        spec = librosa.stft(y=audio.cpu().numpy(), n_fft=200, hop_length=512)
        spec = librosa.power_to_db(np.abs(spec) ** 2)
        normalized = librosa.util.normalize(spec)
        M = 10
        step = 2 / (M - 1)
        for i in range(M):
            # values -1 + step * i - step / 2 <= normalized <= -1 + step * i + step / 2 should be mapped to -1 + step * i
            normalized[(normalized >= -1 + step * i - step / 2) & (normalized <= -1 + step * i + step / 2)] = -1 + step * i
        # spec = spectrogram(audio)
        spec = torch.tensor(normalized, device=self.device)
        return spec
    

    def extract_centroid(self, audio):
        '''
        Returns:
            centroid: spectral centroid vector of the audio
        '''
        audio = audio.to(self.device)
        centroid = librosa.feature.spectral_centroid(y=audio.cpu().numpy(), sr=22050, n_fft=2048, hop_length=512)
        centroid = torch.tensor(centroid, device=self.device)
        return centroid

    

    def extract_chroma(self, audio):
        '''
        Returns:
            chroma: chroma vector of the audio
        '''
        audio = audio.to(self.device)
        chroma = librosa.feature.chroma_stft(y=audio.cpu().numpy(), sr=22050, n_fft=2048, hop_length=512)
        normalized = librosa.util.normalize(chroma)
        M = 10
        step = 2 / (M - 1)
        for i in range(M):
            # values -1 + step * i - step / 2 <= normalized <= -1 + step * i + step / 2 should be mapped to -1 + step * i
            normalized[(normalized >= -1 + step * i - step / 2) & (normalized <= -1 + step * i + step / 2)] = -1 + step * i
        # chroma = torch.tensor(chroma, device=self.device)
        chroma = torch.tensor(normalized, device=self.device)
        return chroma
    

    def extract_mfcc(self, audio):
        '''
        Returns:
            mfcc: mfcc vector of the audio
        '''
        audio = audio.to(self.device)
        mfcc = librosa.feature.mfcc(y=audio.cpu().numpy(), sr=22050, n_mfcc=617)
        normalized = librosa.util.normalize(mfcc)
        # values to M levels between -1 and 1
        M = 10
        step = 2 / (M - 1)
        for i in range(M):
            # values -1 + step * i - step / 2 <= normalized <= -1 + step * i + step / 2 should be mapped to -1 + step * i
            normalized[(normalized >= -1 + step * i - step / 2) & (normalized <= -1 + step * i + step / 2)] = -1 + step * i

        mfcc = torch.tensor(normalized, device=self.device)
        # mfcc = torch.tensor(mfcc, device=self.device)
        return mfcc


    def extract_contrast(self, audio):
        '''
        Returns:
            contrast: spectral contrast vector of the audio
        '''
        audio = audio.to(self.device)
        contrast = librosa.feature.spectral_contrast(y=audio.cpu().numpy(), sr=22050, n_fft=2048, hop_length=512)
        contrast = torch.tensor(contrast, device=self.device)
        return contrast
    

    def extract_features(self, audio):
        # mel = self.extract_mel(audio)
        # spectrogram = self.extract_spectrogram(audio)
        # chroma = self.extract_chroma(audio)
        # # concatenate features
        # features = torch.cat((mel, spectrogram, chroma), dim=0)
        # n_fft = 200
        mfcc = self.extract_mfcc(audio)
        # # fractional bind
        fb = (self.intensity_bases[0] * self.scaling).unsqueeze(0).unsqueeze(0).repeat(mfcc.shape[0], mfcc.shape[1], 1) * mfcc.unsqueeze(2)
        
        # try level encoding

        # bind features
        if self.mfcc_feature_symbols is None:
            self.mfcc_feature_symbols = torch.rand(mfcc.shape[0], HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        features = fb + self.mfcc_feature_symbols.unsqueeze(1).repeat(1, mfcc.shape[1], 1)


        # mel = self.extract_mel(audio)
        # fb = (self.intensity_bases[1] * self.scaling).unsqueeze(0).unsqueeze(0).repeat(mel.shape[0], mel.shape[1], 1) * mel.unsqueeze(2)
        # if self.mel_feature_symbols is None:
        #     self.mel_feature_symbols = torch.rand(mel.shape[0], HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        # mel_features = fb + self.mel_feature_symbols.unsqueeze(1).repeat(1, mel.shape[1], 1)

        # spec = self.extract_spectrogram(audio)
        # fb = (self.intensity_bases[2] * self.scaling).unsqueeze(0).unsqueeze(0).repeat(spec.shape[0], spec.shape[1], 1) * spec.unsqueeze(2)
        # if self.spec_feature_symbols is None:
        #     self.spec_feature_symbols = torch.rand(spec.shape[0], HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        # spec_features = fb + self.spec_feature_symbols.unsqueeze(1).repeat(1, spec.shape[1], 1)

        # chroma = self.extract_chroma(audio)
        # fb = (self.intensity_bases[3] * self.scaling).unsqueeze(0).unsqueeze(0).repeat(chroma.shape[0], chroma.shape[1], 1) * chroma.unsqueeze(2)
        # if self.chroma_feature_symbols is None:
        #     self.chroma_feature_symbols = torch.rand(chroma.shape[0], HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        # chroma_features = fb + self.chroma_feature_symbols.unsqueeze(1).repeat(1, chroma.shape[1], 1)

        return torch.cat((features,), dim=0)

        # return features

    
    def init_shape(self, shape):
        '''
        Initialize time vectors
        '''
        self.num_features, self.timeseries_length = shape
        self.feature_vecs = torch.rand(self.num_features, HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        self.time_vecs = torch.rand(self.timeseries_length, HDC_DIM, device=self.device) * 2 * np.pi - np.pi


    def extract(self, audio):
        feature = self.extract_features(audio)
        # if self.timeseries_length is None:
        #     self.init_shape(feature.shape)
        # # try permutation
        # # fractional bind
        # fb = (self.intenisty_base * self.scaling).unsqueeze(0).unsqueeze(0).repeat(self.num_features, self.timeseries_length, 1) * feature.unsqueeze(2)
        # # permute and bind together
        # for i in range(self.num_features - 1):
        #     fb[i] = torch.roll(fb[i], self.num_features - i - 1, dims=1)
        # bp_sum = torch.sum(fb, dim=0)

        
        # # fractional bind
        # fb = (self.intenisty_base * self.scaling).unsqueeze(0).unsqueeze(0).repeat(self.num_features, self.timeseries_length, 1) * feature.unsqueeze(2)
        # # bind features
        # bp = fb + self.feature_vecs.unsqueeze(1).repeat(1, self.timeseries_length, 1)
        # # bundle features
        # bp_sin = torch.sin(bp)
        # bp_cos = torch.cos(bp)
        # bp_sin_sum = torch.sum(bp_sin, dim=0)
        # bp_cos_sum = torch.sum(bp_cos, dim=0)
        # bp_sum = torch.atan2(bp_sin_sum, bp_cos_sum)

        # bundle features
        bp_sin = torch.sin(feature)
        bp_cos = torch.cos(feature)
        bp_sin_sum = torch.sum(bp_sin, dim=0)
        bp_cos_sum = torch.sum(bp_cos, dim=0)
        bp_sum = torch.atan2(bp_sin_sum, bp_cos_sum)

        # f_sin = torch.sin(feature)
        # f_cos = torch.cos(feature)
        # bp_sum = torch.atan2(f_sin, f_cos).T

        return bp_sum
        
    
    def encode(self, audio, label):
        '''
        Encode audio to a single HDC vector
        '''
        label = int(label)
        f = self.extract(audio)

        # instead of bundling time, we build trigrams and add to the profile vector
        # from the begging to the end, we have 3 * (timeseries_length - 2) trigrams
        # we use a dictionary to store the profile vectors
        # the key is the label, the value is a list of profile vectors
        # we will bundle the profile vectors to get the final profile vector

        if label not in self.profile_vectors:
            self.profile_vectors[label] = []
        # build trigrams r(r(v1)) * r(v2) * v3 where r is permutation and v is the feature vector of a time step
        ngrams = []
        for i in range(f.shape[0] - N + 1):
            vs = []
            for j in range(N):
                vs.append(f[i + j])
            # # permutation
            # for j in range(N - 1):
            #     vs[j] = torch.roll(vs[j], N - j - 1)
            # bind
            v = torch.sum(torch.stack(vs), dim=0)
            # normalize back to -pi to pi
            v = torch.atan2(torch.sin(v), torch.cos(v))
            self.profile_vectors[label].append(v)
        #     ngrams.append(v)
        # # build second order ngrams
        # for i in range(len(ngrams) - N + 1):
        #     vs = []
        #     for j in range(N):
        #         vs.append(ngrams[i + j])
        #     # permutation
        #     for j in range(N - 1):
        #         vs[j] = torch.roll(vs[j], N - j - 1)
        #     # bind
        #     v = torch.sum(torch.stack(vs), dim=0)
        #     # normalize back to -pi to pi
        #     v = torch.atan2(torch.sin(v), torch.cos(v))
        #     self.profile_vectors[label].append(v)

    
    def unbundle(self, audio, label):
        label = int(label)
        f = self.extract(audio)
        if label not in self.profile_vectors:
            self.profile_vectors[label] = []
        # build trigrams r(r(v1)) * r(v2) * v3 where r is permutation and v is the feature vector of a time step
        ngrams = []
        for i in range(f.shape[0] - N + 1):
            vs = []
            for j in range(N):
                vs.append(f[i + j])
            # # permutation
            # for j in range(N - 1):
            #     vs[j] = torch.roll(vs[j], N - j - 1)
            # bind
            v = torch.sum(torch.stack(vs), dim=0)
            # normalize back to -pi to pi
            v = torch.atan2(torch.sin(v), torch.cos(v))
            self.profile_vectors[label].append(-v)


    def train(self, audios, labels):
        seen_labels = set()
        num_seen = {}
        for audio, label in tqdm(zip(audios, labels), "Training HDC", total=len(audios)):
            # if int(label) not in seen_labels:
            #     seen_labels.add(int(label))
            # else:
            #     if int(label) not in num_seen:
            #         num_seen[int(label)] = 0
            #     if num_seen[int(label)] >= 10:
            #         continue
            #     num_seen[int(label)] += 1
            self.encode(audio, label)
                
        # bundle profile vectors
        for label in self.profile_vectors:
            pv_sin = torch.sin(torch.stack(self.profile_vectors[label]))
            pv_cos = torch.cos(torch.stack(self.profile_vectors[label]))
            pv_sin_sum = torch.sum(pv_sin, dim=0)
            pv_cos_sum = torch.sum(pv_cos, dim=0)
            pv_sum = torch.atan2(pv_sin_sum, pv_cos_sum)
            self.final_profile_vectors[label] = pv_sum

        # accuracy = 0
        # # fine tune on the training set
        # for audio, label in tqdm(zip(audios, labels), "Fine Tuning", total=len(audios)):
        #     prediction = self.recognize(audio)
        #     if prediction != int(label):
        #         # self.unbundle(audio, label)
        #         self.encode(audio, label)
        #     else:
        #         accuracy += 1

        # print('Train Accuracy: ', accuracy / len(audios))
        
        # # bundle profile vectors
        # for label in self.profile_vectors:
        #     pv_sin = torch.sin(torch.stack(self.profile_vectors[label]))
        #     pv_cos = torch.cos(torch.stack(self.profile_vectors[label]))
        #     pv_sin_sum = torch.sum(pv_sin, dim=0)
        #     pv_cos_sum = torch.sum(pv_cos, dim=0)
        #     pv_sum = torch.atan2(pv_sin_sum, pv_cos_sum)
        #     self.final_profile_vectors[label] = pv_sum

        # accuracy = 0
        # # fine tune on the training set
        # for audio, label in tqdm(zip(audios, labels), "Fine Tuning", total=len(audios)):
        #     prediction = self.recognize(audio)
        #     if prediction != int(label):
        #         self.unbundle(audio, label)
        #         self.encode(audio, label)
        #     else:
        #         accuracy += 1
        
        # print('Train Accuracy: ', accuracy / len(audios))

        
        # # bundle profile vectors
        # for label in self.profile_vectors:
        #     pv_sin = torch.sin(torch.stack(self.profile_vectors[label]))
        #     pv_cos = torch.cos(torch.stack(self.profile_vectors[label]))
        #     pv_sin_sum = torch.sum(pv_sin, dim=0)
        #     pv_cos_sum = torch.sum(pv_cos, dim=0)
        #     pv_sum = torch.atan2(pv_sin_sum, pv_cos_sum)
        #     self.final_profile_vectors[label] = pv_sum
    

    def recognize(self, audio):
        '''
        Recognize the genre of the audio
        '''
        f = self.extract(audio)

        # after feature extraction, we will build trigrams and add to the profile vector
        profile_vector = []
        # build trigrams r(r(v1)) * r(v2) * v3 where r is permutation and v is the feature vector of a time step
        ngrams = []
        for i in range(f.shape[0] - N + 1):
            vs = []
            for j in range(N):
                vs.append(f[i + j])
            # # permutation
            # for j in range(N - 1):
            #     vs[j] = torch.roll(vs[j], N - j - 1)
            # bind
            v = torch.sum(torch.stack(vs), dim=0)
            # normalize back to -pi to pi
            v = torch.atan2(torch.sin(v), torch.cos(v))
            profile_vector.append(v)
        #     ngrams.append(v)
        # # build second order ngrams
        # for i in range(len(ngrams) - N + 1):
        #     vs = []
        #     for j in range(N):
        #         vs.append(ngrams[i + j])
        #     # permutation
        #     for j in range(N - 1):
        #         vs[j] = torch.roll(vs[j], N - j - 1)
        #     # bind
        #     v = torch.sum(torch.stack(vs), dim=0)
        #     # normalize back to -pi to pi
        #     v = torch.atan2(torch.sin(v), torch.cos(v))
        #     profile_vector.append(v)
        # bundle profile vectors
        pv_sin = torch.sin(torch.stack(profile_vector))
        pv_cos = torch.cos(torch.stack(profile_vector))
        pv_sin_sum = torch.sum(pv_sin, dim=0)
        pv_cos_sum = torch.sum(pv_cos, dim=0)
        pv_sum = torch.atan2(pv_sin_sum, pv_cos_sum)
        # calculate the similarity between the profile vector and the profile vectors of each genre
        # FHRR similarity is calculated by the average angular distance
        similarities = {}
        for label in self.final_profile_vectors:
            similarities[label] = torch.mean(torch.cos(pv_sum - self.final_profile_vectors[label]))
        # print("Similarities: ", similarities)
        # return the genre with the minimum distance
        return max(similarities, key=similarities.get)



def main(num_data):
    torch.manual_seed(8263655087879647638)
    # Load dataset
    dataset = pre.raw_same_len()
    dataset.set_format('torch', columns=['audio', 'genre'])
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=229, stratify_by_column='genre')
    train = dataset['train']
    test = dataset['test']
    # Build a ANN with 2048 input dimension and 10 output dimension, 1 hidden layer with 1024 dimension
    model = HDC(HDC_DIM, scaling=SCALE)
    # Train the model
    model.train(train['audio'][:num_data], train['genre'][:num_data])
    # Test the model
    correct = 0
    for audio, label in tqdm(zip(test['audio'], test['genre']), "Testing", total=len(test['genre'])):
        prediction = model.recognize(audio)
        if prediction == int(label):
            correct += 1
    print('Test Accuracy: ', correct / len(test['genre']))
    # Train accuracy
    # correct = 0
    # for audio, label in tqdm(zip(train['features'], train['genre']), "Training", total=len(train['genre'])):
    #     prediction = model.recognize(audio)
    #     if prediction == int(label):
    #         correct += 1
    # print('Train Accuracy: ', correct / len(train['genre']))
    # Test model and print a confusion matrix
    # confusion_matrix = np.zeros((10, 10))
    # for audio, label in tqdm(zip(test['audio'], test['genre']), "Testing", total=len(test['genre'])):
    #     prediction = model.recognize(audio)
    #     confusion_matrix[int(label)][prediction] += 1
    # print('Confusion Matrix: ')
    # print(confusion_matrix)
    return correct / len(test['genre'])
    

if __name__ == '__main__':
    accs = []
    num_data = [i for i in range(100, 901, 100)]
    for i in num_data:
        accs.append(main(i))
    num_data = [0] + num_data
    accs = [0.1] + accs
    plt.plot(num_data, accs)
    plt.xlabel('Number of Training Data')
    plt.ylabel('Accuracy')
    plt.title('HDC Accuracy vs Number of Training Data')
    plt.savefig('efficiency.png')