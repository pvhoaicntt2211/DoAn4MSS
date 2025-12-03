# file: src/dataset.py

import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
import config

class MUSDBDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sample_rate = config.SAMPLE_RATE
        self.chunk_samples = config.CHUNK_SAMPLES
        self.song_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, index):
        song_folder = os.path.join(self.root_dir, self.song_folders[index])
        
        try:
            mixture_path = os.path.join(song_folder, "mixture.wav")
            vocals_path = os.path.join(song_folder, "vocals.wav")
            
            mixture_wav, _ = librosa.load(mixture_path, sr=self.sample_rate, mono=True)
            vocals_wav, _ = librosa.load(vocals_path, sr=self.sample_rate, mono=True)
            
            if len(mixture_wav) < self.chunk_samples:
                print(f"Warning: Skipping short song {song_folder}")
                return self.__getitem__((index + 1) % len(self))

            start = np.random.randint(0, len(mixture_wav) - self.chunk_samples)
            end = start + self.chunk_samples
            mixture_chunk = mixture_wav[start:end]
            vocals_chunk = vocals_wav[start:end]

            mixture_spec = librosa.stft(mixture_chunk, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
            vocals_spec = librosa.stft(vocals_chunk, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
            
            mixture_mag = np.abs(mixture_spec)
            vocals_mag = np.abs(vocals_spec)

            mixture_mag = torch.FloatTensor(mixture_mag).unsqueeze(0)
            vocals_mag = torch.FloatTensor(vocals_mag).unsqueeze(0)

            return mixture_mag, vocals_mag

        except Exception as e:
            print(f"Error loading song {song_folder}: {e}")
            return self.__getitem__((index + 1) % len(self))