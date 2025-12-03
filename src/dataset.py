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
        self.stems = config.STEMS  # ['vocals', 'drums', 'bass', 'other']
        self.song_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, index):
        song_folder = os.path.join(self.root_dir, self.song_folders[index])
        
        try:
            mixture_path = os.path.join(song_folder, "mixture.wav")
            
            # Load mixture
            mixture_wav, _ = librosa.load(mixture_path, sr=self.sample_rate, mono=True)
            
            if len(mixture_wav) < self.chunk_samples:
                print(f"Warning: Skipping short song {song_folder}")
                return self.__getitem__((index + 1) % len(self))

            # Random chunk selection
            start = np.random.randint(0, len(mixture_wav) - self.chunk_samples)
            end = start + self.chunk_samples
            mixture_chunk = mixture_wav[start:end]

            # Load all 4 stems
            stems_chunks = []
            for stem in self.stems:
                stem_path = os.path.join(song_folder, f"{stem}.wav")
                stem_wav, _ = librosa.load(stem_path, sr=self.sample_rate, mono=True)
                stem_chunk = stem_wav[start:end]
                stems_chunks.append(stem_chunk)

            # Compute spectrograms
            mixture_spec = librosa.stft(mixture_chunk, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
            mixture_mag = np.abs(mixture_spec)

            # Stack all stem spectrograms: shape (4, F, T)
            stems_specs = []
            for stem_chunk in stems_chunks:
                stem_spec = librosa.stft(stem_chunk, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
                stem_mag = np.abs(stem_spec)
                stems_specs.append(stem_mag)
            
            stems_mag = np.stack(stems_specs, axis=0)  # Shape: (4, F, T)

            # Convert to tensors
            mixture_mag = torch.FloatTensor(mixture_mag).unsqueeze(0)  # (1, F, T)
            stems_mag = torch.FloatTensor(stems_mag)  # (4, F, T)

            return mixture_mag, stems_mag

        except Exception as e:
            print(f"Error loading song {song_folder}: {e}")
            return self.__getitem__((index + 1) % len(self))