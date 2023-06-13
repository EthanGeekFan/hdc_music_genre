import torch
import numpy as np
import torchaudio.transforms as T

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
        self.pitches = torch.rand(12, dim, device=self.device) * 2 * np.pi - np.pi
        self.intenisty_base = torch.rand(dim, device=self.device) * 2 * np.pi - np.pi
        self.scaling = 1
        self.n_mels = 128
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=self.n_mels).to(self.device)
        self.mel_vectors = torch.rand(self.n_mels, dim, device=self.device) * 2 * np.pi - np.pi

    
    def init_shape(self, shape):
        '''
        Initialize time vectors
        '''
        self.num_features, self.timeseries_length = shape
        self.feature_vecs = torch.rand(self.num_features, self.dim, device=self.device) * 2 * np.pi - np.pi
        self.time_vecs = torch.rand(self.timeseries_length, self.dim, device=self.device) * 2 * np.pi - np.pi
        
    
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


def hdc_preproc(ds, dim):
    '''
    Do chroma stft and HDC encode for each audio
    Args:
        ds: dataset
    Returns:
        ds: dataset with chroma stft
    '''
    # dim = 512
    hdc_encoder = HDC(dim)
    ds = ds.map(lambda x: {'hdc': hdc_encoder.encode(x['features'])}, desc="HDC encode")
    return ds