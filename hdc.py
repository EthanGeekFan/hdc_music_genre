from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
import librosa
import os
import torch
from torch import nn
import matplotlib.pyplot as plt


class ANN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(ANN, self).__init__()
        # 1 hidden layer and softmax output
        # loss function: cross entropy
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def predict(self, x):
        return self.forward(x).argmax(dim=1)
    
    def loss(self, x, y):
        return nn.CrossEntropyLoss()(self.forward(x), y)
    
    def accuracy(self, x, y):
        return (self.predict(x) == y).float().mean()
    
    def train(self, x, y, optimizer, x_test, y_test, epochs=100, batch_size=32):
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
            accuracies.append(self.accuracy(x, y))
            test_accuracies.append(self.accuracy(x_test, y_test))
        print(f'Train Accuracy: {self.accuracy(x, y)}')
        print(f'Test Accuracy: {self.accuracy(x_test, y_test)}')
        return losses, accuracies, test_accuracies
        
    def test(self, x, y):
        print(f'Test Accuracy: {self.accuracy(x, y)}')
        
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
    dataset = dataset.train_test_split(test_size=0.2, seed=229, stratify_by_column='genre')
    train = dataset['train']
    test = dataset['test']
    return train, test


HDC_DIM = 2048

class HDC:
    '''
    Use FHRR representation
    Use 12 chrome vectors to represent 12 pitches
    encode intensity with a shared base vector
    '''
    
    def __init__(self):
        self.pitches = []
        for _ in range(12):
            pitch_hv = np.random.uniform(-np.pi, np.pi, HDC_DIM)
            self.pitches.append(pitch_hv)
        self.intenisty_base = np.random.uniform(-np.pi, np.pi, HDC_DIM)
        self.time_base = np.random.uniform(-np.pi, np.pi, HDC_DIM)
        self.scaling = 1
        
    
    def encode(self, audio):
        '''
        Encode audio to a single HDC vector
        '''
        assert audio.shape[0] == 12
        # for each frame, bind pitch
        result = np.zeros((audio.shape[1], HDC_DIM))
        for i in range(audio.shape[1]):
            frame = audio[:, i]
            intermediate = np.zeros((12, HDC_DIM))
            for j in range(12):
                intermediate[j, :] = HDC.bind(self.pitches[j], HDC.frac_bind(self.intenisty_base, frame[j]))
            # bundle
            result[i, :] = HDC.bind(HDC.bundle(intermediate), HDC.frac_bind(self.time_base, i * self.scaling / audio.shape[1]))
        return HDC.bundle(result)
        
        
    @staticmethod
    def bind(a, b):
        '''
        Bind two HDC vectors
        Args:
            a: HDC vector
            b: HDC vector
        Returns:
            c: HDC vector
        '''
        return a + b
    
    
    @staticmethod
    def unbind(a, b):
        '''
        Unbind two HDC vectors
        Args:
            a: HDC vector
            b: HDC vector
        Returns:
            c: HDC vector
        '''
        return b - a
    
    
    @staticmethod
    def bundle(vectors):
        '''
        Bundle multiple HDC vectors
        Args:
            a: HDC vector
            b: HDC vector
        Returns:
            c: HDC vector
        '''
        return np.angle(np.sum(np.exp(1j * vectors), axis=0))
    
    
    @staticmethod
    def frac_bind(base, val):
        '''
        Fractional bind
        Args:
            base: HDC vector
            val: real number
        Returns:
            c: HDC vector
        '''
        return base * val


hdc_encoder = HDC()


def preproc(ds: Dataset):
    '''
    Do chroma stft and HDC encode for each audio
    Args:
        ds: dataset
    Returns:
        ds: dataset with chroma stft
    '''
    ds = ds.map(lambda x: {'stft': librosa.feature.chroma_stft(y=x['audio']['array'], sr=22050)}, desc="Chroma STFT").with_format('numpy')
    ds = ds.map(lambda x: {'hdc': hdc_encoder.encode(x['stft'])}, desc="HDC encode")
    return ds


def main():
    if not os.path.exists('gtzan_train') or not os.path.exists('gtzan_test'):
        # preprocess
        train, test = load_gtzan()
        train = preproc(train)
        test = preproc(test)
        train.save_to_disk('gtzan_train')
        test.save_to_disk('gtzan_test')
    else:
        train = load_from_disk('gtzan_train').with_format('torch', columns=['hdc', 'genre'])
        test = load_from_disk('gtzan_test').with_format('torch', columns=['hdc', 'genre'])
    # Build a ANN with 2048 input dimension and 10 output dimension, 1 hidden layer with 1024 dimension
    model = ANN(HDC_DIM, 3000, 10)
    # Train the model
    losses, accuracies, test_accuracies = model.train(train['hdc'], train['genre'], 
                                                      torch.optim.SGD(model.parameters(), lr=0.01),
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