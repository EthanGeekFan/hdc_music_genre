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


class ANN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(ANN, self).__init__()
        # 1 hidden layer and softmax output
        # loss function: cross entropy
        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=1),
        )
        # choose device based on availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # NVIDIA GPU
            print('ANN Using CUDA GPU: ', torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple GPU
            print('ANN Using Apple GPU')
        else:
            self.device = torch.device('cpu') # CPU
            print('ANN Using CPU')
        self.to(self.device)
        
    def forward(self, x):
        logits = self.layers(x)
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores
    
    def predict(self, x):
        return self.forward(x).argmax(dim=1)
    
    def loss(self, x, y):
        return nn.NLLLoss()(self.forward(x), y)
    
    def accuracy(self, x, y):
        return (self.predict(x) == y).float().mean()
    
    def train(self, x, y, optimizer, x_test, y_test, epochs=100, batch_size=32):
        print('Training on', self.device)
        print('X shape:', x.shape)
        print('Y shape:', y.shape)
        # convert Y to one-hot
        # y = F.one_hot(y, num_classes=10)
        # y_test = F.one_hot(y_test, num_classes=10)
        # print('Y shape:', y.shape)
        # print(y[0])
        # print('Y test shape:', y_test.shape)
        x = x.to(self.device)
        y = y.to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        losses = []
        accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            for i in range(0, x.shape[0], batch_size):
                # forward
                loss = self.loss(x[i:i+batch_size], y[i:i+batch_size])
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch}: loss {loss.item()}')
            losses.append(loss.item())
            accuracies.append(self.accuracy(x, y).cpu().item())
            test_accuracies.append(self.accuracy(x_test, y_test).cpu().item())
        print(f'Train Accuracy: {self.accuracy(x, y)}')
        print(f'Test Accuracy: {self.accuracy(x_test, y_test)}')
        return losses, accuracies, test_accuracies
        
    def test(self, x, y):
        print(f'Test Accuracy: {self.accuracy(x.to(self.device), y.to(self.device))}')
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        torch.load_state_dict(torch.load(path))


def load_gtzan():
    '''
    Load GTZAN dataset
    Returns:
        train: training set
        test: test set
    '''
    dataset = load_dataset('marsyas/gtzan', split='train')
    return dataset


HDC_DIM = 512

class HDC:
    '''
    Use FHRR representation
    Use 12 chrome vectors to represent 12 pitches
    encode intensity with a shared base vector
    '''
    
    def __init__(self, dim, scaling=6):
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
        self.intenisty_base = torch.rand(HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        self.scaling = 1
        self.n_mels = 128
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=self.n_mels).to(self.device)
        self.mel_vectors = torch.rand(self.n_mels, HDC_DIM, device=self.device) * 2 * np.pi - np.pi

    
    def init_shape(self, shape):
        '''
        Initialize time vectors
        '''
        self.num_features, self.timeseries_length = shape
        self.feature_vecs = torch.rand(self.num_features, HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        self.time_vecs = torch.rand(self.timeseries_length, HDC_DIM, device=self.device) * 2 * np.pi - np.pi
        
    
    def encode(self, audio):
        '''
        Encode audio to a single HDC vector
        '''
        # librosa returns 12 chroma vectors
        # audio = librosa.feature.chroma_stft(y=audio, sr=22050)
        if self.timeseries_length is None:
            self.init_shape(audio.shape)
        assert audio.shape[0] == self.num_features
        if not audio.shape[1] == self.timeseries_length:
            print('Warning: audio shape does not match time windows')
            print('Audio shape: ', audio.shape)
            print('Time series length: ', self.timeseries_length)
            raise ValueError
        audio = audio.to(self.device)
        # fractional bind
        fb = (self.intenisty_base * self.scaling).unsqueeze(0).unsqueeze(0).repeat(self.num_features, audio.shape[1], 1) * audio.unsqueeze(2)
        # bind features
        bp = fb + self.feature_vecs.unsqueeze(1).repeat(1, self.timeseries_length, 1)
        # bundle features
        bp_sin = torch.sin(bp)
        bp_cos = torch.cos(bp)
        bp_sin_sum = torch.sum(bp_sin, dim=0)
        bp_cos_sum = torch.sum(bp_cos, dim=0)
        bp_sum = torch.atan2(bp_sin_sum, bp_cos_sum)
        # bind time
        et = bp_sum + self.time_vecs
        # bundle time
        et_sin = torch.sin(et)
        et_cos = torch.cos(et)
        et_sin_sum = torch.sum(et_sin, dim=0)
        et_cos_sum = torch.sum(et_cos, dim=0)
        et_sum = torch.atan2(et_sin_sum, et_cos_sum)
        return et_sum
    

def chroma_stft_preproc(ds):
    '''
    Do chroma stft for each audio
    Args:
        ds: dataset
    Returns:
        ds: dataset with chroma stft
    '''
    ds = ds.map(lambda x: {'stft': librosa.feature.chroma_stft(y=x['audio']['array'], sr=22050)}, desc="Chroma STFT").with_format('torch', columns=['stft'])
    return ds


def hdc_preproc(ds):
    '''
    Do chroma stft and HDC encode for each audio
    Args:
        ds: dataset
    Returns:
        ds: dataset with chroma stft
    '''
    hdc_encoder = HDC(HDC_DIM)
    ds = ds.map(lambda x: {'hdc': hdc_encoder.encode(x['features'])}, desc="HDC encode")
    return ds


def main():
    # Load dataset
    dataset = None
    # if not os.path.exists('gtzan_chroma_stft'):
    #     dataset = load_gtzan()
    #     dataset = chroma_stft_preproc(dataset)
    #     dataset.save_to_disk('gtzan_chroma_stft')
    # find the minimum length of all audio
    hdc_ds_path = f'gtzan_hdc_standard_{HDC_DIM}'
    if not os.path.exists(hdc_ds_path):
        dataset = pre.standard()
        dataset.set_format('torch', columns=['features'])
        # preprocess
        dataset = hdc_preproc(dataset)
        dataset.save_to_disk(hdc_ds_path)
    else:
        dataset = load_from_disk(hdc_ds_path)
    dataset.set_format('torch', columns=['hdc', 'genre'])
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=229, stratify_by_column='genre')
    train = dataset['train']
    test = dataset['test']
    # Build a ANN with 2048 input dimension and 10 output dimension, 1 hidden layer with 1024 dimension
    model = ANN(HDC_DIM, 30, 10)
    # Train the model
    losses, accuracies, test_accuracies = model.train(train['hdc'], train['genre'], 
                                                      torch.optim.Adam(model.parameters(), lr=0.0001),
                                                      test['hdc'], test['genre'],
                                                      epochs=100)
    # plot the loss, accuracy, test accuracy curve in two subplots
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('loss_acc.png')
    # Test the model
    model.test(test['hdc'], test['genre'])
    # Save the model
    model.save('model.pt')
    

if __name__ == '__main__':
    main()