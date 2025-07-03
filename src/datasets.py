import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from src.embedder import Embedder
from abc import abstractmethod
from pathlib import Path
import pickle
import random

MNIST_MEL_SPEC = (128, 33)
MNIST_MFCC = (13, 33)
URBANSOUND_MEL_SPEC = (128, 128)
URBANSOUND_MFCC = (40, 128)


class AudioDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", sample_rate: int = 16000, feature_type: str = "melspectrogram", n_mels: int = 128, n_mfcc: int = 40, hop_length: int = 512):
        """
        Base class for audio datasets with support for MelSpectrogram and MFCC feature extraction.

        Args:
            root_dir (str): Path to the dataset directory.
            split (str): Dataset split. Options: "train" or "val".
            sample_rate (int): Sampling rate for audio files.
            feature_type (str): Type of feature to extract. Options: "melspectrogram" or "mfcc".
            n_mels (int): Number of mel bands for MelSpectrogram.
            n_mfcc (int): Number of MFCC features to extract.
        """
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.feature_type = feature_type.lower()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.classes, self.label_map, self.file_paths, self.labels = self._load_dataset()

        self._init_transforms(n_mels, hop_length, n_mfcc)

        # For padding/truncating audio files
        self.max_length = int(sample_rate * 3.09)  # Most UrbanSound8K clips are 4 seconds or less

    @abstractmethod
    def _load_dataset(self):
        pass

    
    def __adjust_dims(self, features):
        # second_dim = MNIST_MEL_SPEC[1]
        # _, n_mels, time = features.shape 
        
        # if time < second_dim:
        #     # Pad with zeros
        #     pad = torch.zeros((1, n_mels, second_dim - time))
        #     features = torch.cat([features, pad], dim=2)

        return features
    
    def _init_transforms(self, n_mels, hop_length, n_mfcc):
        """Initialize audio feature transforms."""
        # Define transforms
        if self.feature_type == "melspectrogram":
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=hop_length
            )
        elif self.feature_type == "mfcc":
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={"n_fft": 2048, "hop_length": hop_length, "n_mels": n_mels}
            )
        elif self.feature_type == 'combined':
            # Use both MelSpectrogram and MFCC
            self.spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=hop_length
            )
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={"n_fft": 2048, "hop_length": hop_length, "n_mels": n_mels}
            )
    
    def _load_audio(self, file_path):
        """Load audio file for a given track."""
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad/truncate to consistent length
        if waveform.shape[1] < self.max_length:
            # Pad with zeros
            pad_size = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[1] > self.max_length:
            # Truncate
            waveform = waveform[:, :self.max_length]
        
        return waveform
    
    def _extract_features(self, waveform):
        """Extract features from audio."""
        
        if self.feature_type == 'combined':
            spec_features = self.__adjust_dims(self.spec_transform(waveform))
            mfcc_features = self.__adjust_dims(self.mfcc_transform(waveform))
            return spec_features, mfcc_features
        else:
            features = self.transform(waveform)
            features = self.__adjust_dims(features)

            return features
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        track_id = self.file_paths[idx]
        # Load audio
        audio = self._load_audio(track_id)
        
        # Extract features
        label = self.labels[idx]
        if self.feature_type == 'combined':
            spec_features, mfcc_features = self._extract_features(audio)
            return spec_features, mfcc_features, label
        else:
            features = self._extract_features(audio)
            return features, label
    
    def get_class_name(self, idx):
        """Get class name for a given index."""
        return self.classes[idx]
    
    def get_dataset_info(self):
        """Get information about the dataset."""
        info = {
            'num_classes': len(self.classes),
            'classes': self.classes,
            'num_samples': len(self.track_ids),
            'split': self.split,
            'feature_type': self.feature_type,
            'sample_rate': self.sample_rate
        }
        return info

class GTZANDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", sample_rate: int = 16000, feature_type: str = "melspectrogram", n_mels: int = 128, n_mfcc: int = 13, chunk_size: int = 128):
        """
        GTZAN Dataset with support for MelSpectrogram and MFCC feature extraction.

        Args:
            root_dir (str): Path to the 'data/raw' directory containing genre subfolders.
            split (str): Dataset split. Options: "train" or "val".
            sample_rate (int): Sampling rate for audio files.
            feature_type (str): Type of feature to extract. Options: "melspectrogram" or "mfcc".
            n_mels (int): Number of mel bands for MelSpectrogram.
            n_mfcc (int): Number of MFCC features to extract.
            chunk_size (int): Size of the feature map (chunk_size x chunk_size).
        """
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.feature_type = feature_type.lower()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.chunk_size = chunk_size

        # Get genre labels from subfolder names (ignore non-directory files like .DS_Store)
        self.genres = [genre for genre in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, genre))]
        self.label_map = {genre: idx for idx, genre in enumerate(self.genres)}

        # Collect all file paths and labels
        self.file_paths = []
        self.labels = []
        for genre in self.genres:
            genre_folder = os.path.join(root_dir, genre)
            files = sorted(os.listdir(genre_folder))
            
            # Split files into train and val
            if self.split == "train":
                files = files[:int(0.8 * len(files))]  # First 80% for training
            elif self.split == "val":
                files = files[int(0.8 * len(files)):]  # Last 20% for validation
            else:
                raise ValueError("Invalid split. Choose 'train' or 'val'.")

            for file in files:
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(genre_folder, file))
                    self.labels.append(self.label_map[genre])

        # Define transforms
        if self.feature_type == "melspectrogram":
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=2048,
                hop_length=512
            )
        elif self.feature_type == "mfcc":
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": self.n_mels}
            )
        else:
            raise ValueError("Invalid feature_type. Choose 'melspectrogram' or 'mfcc'.")

    def __len__(self):
        return len(self.file_paths)*7

    def __getitem__(self, idx):
        """Loads an audio file, extracts features, and returns a tensor."""
        chunk_idx = idx%7
        idx = idx//7
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio file
        waveform, sr = torchaudio.load(file_path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract features
        features = self.transform(waveform)

        # Convert to log scale
        features = torch.log(features + 1e-6)

        # Split into chunks
        chunks = self._split_into_chunks(features)
        
        return chunks[chunk_idx], torch.tensor(label, dtype=torch.long)

    def _split_into_chunks(self, features):
        """Splits the feature map into chunks of size chunk_size x chunk_size."""
        n_frames = features.shape[2]
        chunks = []
        for i in range(0, n_frames, self.chunk_size):
            if i + self.chunk_size <= n_frames:
                chunk = features[:, :, i:i + self.chunk_size]
                if chunk.shape[2] < self.chunk_size:
                    # Pad the last chunk if necessary
                    pad_size = self.chunk_size - chunk.shape[2]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                chunks.append(chunk)
        return torch.stack(chunks)

class AudioMNISTDataset(AudioDataset):
    def __init__(self, root_dir: str, split="train", train_size=0.80, sample_rate=16000, 
                 feature_type='melspectrogram', hop_length=512, n_mfcc=13, n_mels=128):
        self.train_size = train_size
        super().__init__(root_dir, split, sample_rate, feature_type, n_mels, n_mfcc, hop_length)
        self.max_length = sample_rate + 500
        
    def _load_dataset(self):
        """Load AudioMNIST dataset metadata and prepare file paths/labels."""
        file_list = []
        label_list = []
        
        # Walk through directory to find all WAV files
        for dirname, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    file_list.append(os.path.join(dirname, filename))
                    label_list.append(int(filename[0]))  # First character is the digit label
        
        # Split into train/val based on train_size
        total_len = len(file_list)
        if self.split == 'train':
            file_list = file_list[:int(self.train_size * total_len)]
            label_list = label_list[:int(self.train_size * total_len)]
        elif self.split == 'val':
            file_list = file_list[int(self.train_size * total_len):]
            label_list = label_list[int(self.train_size * total_len):]
        
        # AudioMNIST has digits 0-9 as classes
        classes = [str(i) for i in range(10)]
        label_map = {str(i): i for i in range(10)}
        
        return classes, label_map, file_list, label_list

    def __adjust_dims(self, features):
        """Special dimension adjustment for AudioMNIST to match MNIST_MEL_SPEC size."""
        second_dim = MNIST_MEL_SPEC[1]
        _, n_mels, time = features.shape
        
        if time < second_dim:
            # Pad with zeros
            pad = torch.zeros((1, n_mels, second_dim - time))
            features = torch.cat([features, pad], dim=2)

        return features
class UrbanSound8KDataset(AudioDataset):
    def __init__(self, root_dir: str, split="train", train_size=0.80, sample_rate=16000, 
                 feature_type='melspectrogram', hop_length=512, n_mfcc=13, n_mels=128):
        super().__init__(root_dir, split, sample_rate, feature_type, n_mels, n_mfcc, hop_length)
        self.max_length = int(sample_rate * 4.09)
       
    def _load_dataset(self):
        """Load UrbanSound8K dataset metadata and prepare file paths/labels."""
        # Load metadata
        metadata_path = os.path.join(self.root_dir, 'metadata', 'UrbanSound8K.csv')
        metadata = pd.read_csv(metadata_path)
        
        # Create label map from class names
        classes = sorted(metadata['class'].unique())
        label_map = {cls: idx for idx, cls in enumerate(classes)}
        
        # Handle folds parameter (specific to UrbanSound8K)
        if hasattr(self, 'folds') and self.folds is not None:
            folds = self.folds
        else:
            if self.split == "train":
                folds = [1, 2, 3, 4, 5, 6, 7, 8]  # First 8 folds for training
            elif self.split == "val":
                folds = [9, 10]  # Last 2 folds for validation
        
        # Filter metadata for selected folds
        metadata = metadata[metadata['fold'].isin(folds)]
        
        # Collect all file paths and labels
        file_paths = []
        labels = []
        for _, row in metadata.iterrows():
            fname = row['slice_file_name']
            file_path = os.path.join(self.root_dir, 'audio', f'fold{row["fold"]}', fname)
            file_paths.append(file_path)
            labels.append(label_map[row['class']])
        
        return classes, label_map, file_paths, labels

class MedleySolosDataset(AudioDataset):
    def _load_dataset(self):
        """Load Medley-solos dataset metadata and prepare file paths/labels."""
        # Load metadata
        metadata = pd.read_csv(os.path.join(self.root_dir, 'annotation', 'Medley-solos-DB_metadata.csv'))
        
        # Get instrument classes
        classes = ['Clarinet', 'Distorted Electric Guitar', 'Female Singer', 
                  'Flute', 'Piano', 'Tenor Saxophone', 'Trumpet', 'Violin']
        label_map = {cls: idx for idx, cls in zip(metadata.instrument, metadata.instrument_id)}
        
        # Handle split parameter
        split = 'validation' if self.split == 'val' else self.split
        split = 'training' if split == 'train' else split
        
        # Collect all file paths and labels
        file_paths = []
        labels = []
        for _, row in metadata.iterrows():
            if row['subset'] != split:
                continue
            fname = f'Medley-solos-DB_{row["subset"]}-{row["instrument_id"]}_{row["uuid4"]}.wav'
            file_path = os.path.join(self.root_dir, 'audio', fname)
            file_paths.append(file_path)
            labels.append(row['instrument_id'])
        
        return classes, label_map, file_paths, labels

class GIANTSTEPS_TEMPO(AudioDataset):
    def __init__(self, root_dir: str, split: str = "train", sample_rate: int = 16000, 
                 feature_type: str = "melspectrogram", n_mels: int = 128, n_mfcc: int = 40, 
                 hop_length: int = 512, chunk_duration: float = 3.0):
        """
        GiantSteps Tempo Dataset with:
        - Chunked spectrograms
        - Discretized tempo labels (3 classes)
        - Custom train/val split: 50 songs per class for VAL, rest for TRAIN
        """
        self.chunk_duration = chunk_duration
        self.frames_per_chunk = int(chunk_duration * sample_rate / hop_length)
        self.songs_per_class_val = 50  # 50 songs per class for VALIDATION
        
        super().__init__(root_dir, split, sample_rate, feature_type, n_mels, n_mfcc, hop_length)
        self.max_length = int(sample_rate * 120)
        # Discretize labels before chunking
        self.original_labels = [self._discretize_tempo(bpm) for bpm in self.labels]
        self.classes = ['slow', 'medium', 'fast']  # Update class names
        self.label_map = {'slow': 0, 'medium': 1, 'fast': 2}
        
        # Generate chunk indices
        self.chunk_indices = self._generate_chunk_indices()

    def _discretize_tempo(self, bpm: float) -> int:
        """Convert continuous BPM to discrete class (0, 1, or 2)"""
        if bpm < 120:
            return 0  # 'slow'
        elif 120 <= bpm <= 140:
            return 1  # 'medium'
        else:
            return 2  # 'fast'

    def _load_dataset(self):
        """Load GiantSteps tempo annotations with custom train/val split."""
        audio_dir = os.path.join(self.root_dir, 'audio')
        tempo_dir = os.path.join(self.root_dir, 'tempo')
        
        # First, load all files and their BPMs
        all_files = []
        all_bpms = []
        
        for audio_file in Path(audio_dir).glob('*.mp3'):
            tempo_file = Path(tempo_dir) / f"{audio_file.stem}.bpm"
            if tempo_file.exists():
                with open(tempo_file, 'r') as f:
                    bpm = float(f.read().strip())
                    all_files.append(str(audio_file))
                    all_bpms.append(bpm)
        
        # Group files by discretized tempo class
        files_by_class = {0: [], 1: [], 2: []}  # slow, medium, fast
        bpms_by_class = {0: [], 1: [], 2: []}
        
        for file_path, bpm in zip(all_files, all_bpms):
            tempo_class = self._discretize_tempo(bpm)
            files_by_class[tempo_class].append(file_path)
            bpms_by_class[tempo_class].append(bpm)
        
        # Print class distribution
        print("Class distribution:")
        for class_idx, class_name in enumerate(['slow', 'medium', 'fast']):
            print(f"  {class_name}: {len(files_by_class[class_idx])} songs")
        
        # Create train/val split: 50 songs per class for VAL, rest for TRAIN
        train_files = []
        train_bpms = []
        val_files = []
        val_bpms = []
        
        for class_idx in range(3):
            class_files = files_by_class[class_idx]
            class_bpms = bpms_by_class[class_idx]
            
            # Ensure we have enough files for this class
            available_files = len(class_files)
            val_count = min(self.songs_per_class_val, available_files)
            
            if available_files < self.songs_per_class_val:
                print(f"Warning: Class {['slow', 'medium', 'fast'][class_idx]} has only "
                      f"{available_files} songs, using all for validation")
            
            # Sort for reproducibility
            combined = list(zip(class_files, class_bpms))
            combined.sort(key=lambda x: x[0])  # Sort by filename
            
            # Split: first val_count for VALIDATION, rest for TRAINING
            val_portion = combined[:val_count]
            train_portion = combined[val_count:]
            
            val_files.extend([f for f, _ in val_portion])
            val_bpms.extend([b for _, b in val_portion])
            train_files.extend([f for f, _ in train_portion])
            train_bpms.extend([b for _, b in train_portion])
            
            print(f"  Split for {['slow', 'medium', 'fast'][class_idx]}: "
                  f"{len(train_portion)} train, {len(val_portion)} val")
        
        # Select files based on split
        if self.split == 'train':
            selected_files = train_files
            selected_bpms = train_bpms
        elif self.split == 'val':
            selected_files = val_files
            selected_bpms = val_bpms
        else:
            # For 'all' or other splits, use all data
            selected_files = all_files
            selected_bpms = all_bpms
        
        print(f"Selected {len(selected_files)} songs for {self.split} split")
        
        # Store original file paths and BPMs for chunk generation
        self.original_file_paths = selected_files
        self.original_bpms = selected_bpms
        
        # Return dummy classes (will be updated in __init__ after discretization)
        return ['tempo'], {'tempo': 0}, selected_files, selected_bpms

    def _generate_chunk_indices(self):
        """Generate indices for audio chunks."""
        chunk_indices = []
        
        for idx, file_path in enumerate(self.original_file_paths):

            try:
                # Load audio to determine number of chunks
                waveform = self._load_audio(file_path)
                features = self._extract_features(waveform)
                
                if isinstance(features, tuple):  # For combined features
                    features = features[0]  # Use spectrogram for chunk calculation
                
                # Calculate number of chunks
                total_frames = features.shape[2]
                n_chunks = total_frames // self.frames_per_chunk
                
                # Only add chunks if we have at least one complete chunk
                if n_chunks > 0:
                    for chunk_idx in range(n_chunks):
                        chunk_indices.append((idx, chunk_idx))
                        
            except Exception as e:
                print(f"  Warning: Could not process {file_path}: {e}")
                continue
        
        print(f"Generated {len(chunk_indices)} chunks total")
        
        # Print chunk distribution by class
        chunk_counts = {0: 0, 1: 0, 2: 0}
        for file_idx, _ in chunk_indices:
            tempo_class = self.original_labels[file_idx]
            chunk_counts[tempo_class] += 1
        
        print("Chunk distribution by class:")
        for class_idx, class_name in enumerate(['slow', 'medium', 'fast']):
            print(f"  {class_name}: {chunk_counts[class_idx]} chunks")
        
        return chunk_indices

    def _get_chunk(self, features, chunk_idx):
        """Extract a chunk from features."""
        start = chunk_idx * self.frames_per_chunk
        end = start + self.frames_per_chunk
        return features[:, :, start:end]

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        file_idx, chunk_idx = self.chunk_indices[idx]
        
        # Load original audio
        audio = self._load_audio(self.original_file_paths[file_idx])
        
        # Extract features
        if self.feature_type == 'combined':
            spec_features, mfcc_features = self._extract_features(audio)
            spec_chunk = self._get_chunk(spec_features, chunk_idx)
            mfcc_chunk = self._get_chunk(mfcc_features, chunk_idx)
            
            # Ensure chunks have correct size (pad if necessary)
            if spec_chunk.shape[2] < self.frames_per_chunk:
                pad_size = self.frames_per_chunk - spec_chunk.shape[2]
                spec_chunk = torch.nn.functional.pad(spec_chunk, (0, pad_size))
                mfcc_chunk = torch.nn.functional.pad(mfcc_chunk, (0, pad_size))
                
            return spec_chunk, mfcc_chunk, self.original_labels[file_idx]
        else:
            features = self._extract_features(audio)
            chunk = self._get_chunk(features, chunk_idx)
            
            # Pad if necessary
            if chunk.shape[2] < self.frames_per_chunk:
                pad_size = self.frames_per_chunk - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
            return chunk, self.original_labels[file_idx]
    
    def get_split_info(self):
        """Get information about the train/val split."""
        return {
            'songs_per_class_val': self.songs_per_class_val,  # Updated variable name
            'split': self.split,
            'total_songs': len(self.original_file_paths),
            'total_chunks': len(self.chunk_indices),
            'chunk_duration': self.chunk_duration
        }

def get_dataset(dataset, data_path, feature_type='melspectrogram', batch_size=256, chunk_duration=3.0):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'GTZAN':
        channel = 1  # Mel-Spectrogram is single-channel
        im_size = (128, 128)
        num_classes = 10
        mean = [0.0]
        std = [1.0]

        # Load GTZAN dataset manually
        dst_train = GTZANDataset(root_dir=os.path.join(data_path, 'GTZAN'), split = 'train')
        dst_test = GTZANDataset(root_dir=os.path.join(data_path, 'GTZAN'), split ='val')
        class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    elif dataset == 'AUDIO_MNIST':
        channel = 1
        im_size =  MNIST_MEL_SPEC if feature_type == 'melspectrogram' else MNIST_MFCC
        # im_size =  MNIST_MEL_SPEC 
        num_classes = 10
        mean = [0.0]
        std = [1.0]
        if feature_type == 'combined':
            dst_train = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), feature_type= 'combined', split = 'train', hop_length=512)
            dst_test = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), feature_type= 'melspectrogram', split ='val', hop_length=512)
        else:
            dst_train = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), feature_type= feature_type, split = 'train', hop_length=512)
            dst_test = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), feature_type= feature_type, split ='val', hop_length=512)
        class_names = [str(i) for i in range(num_classes)]
    elif dataset == 'URBANSOUND8K':
        channel = 1
        im_size = URBANSOUND_MEL_SPEC if feature_type == 'melspectrogram' else URBANSOUND_MFCC
        num_classes = 10
        mean = [0.0]
        std = [1.0]
        if feature_type == 'combined':
            dst_train = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), feature_type= 'combined', split = 'train', hop_length=512)
            dst_test = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), feature_type= 'melspectrogram', split ='val', hop_length=512)
        else:
            dst_train = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), feature_type= feature_type, split='train')
            dst_test = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), split='val')
        class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
    elif dataset == 'MedleySolos':
        channel = 1
        im_size = (128, 97)
        num_classes = 8
        mean = [0.0]
        std = [1.0]
        
        if feature_type == 'combined':
            dst_train = MedleySolosDataset(root_dir=os.path.join(data_path, 'Medley-solos-DB'), split='train', feature_type='combined')
            dst_test = MedleySolosDataset(root_dir=os.path.join(data_path, 'Medley-solos-DB'), split='test', feature_type='melspectrogram')
        else:
            dst_train = MedleySolosDataset(root_dir=os.path.join(data_path, 'Medley-solos-DB'), split='train', feature_type=feature_type)
            dst_test = MedleySolosDataset(root_dir=os.path.join(data_path, 'Medley-solos-DB'), split='test', feature_type=feature_type)
        class_names = dst_train.classes

    elif dataset == 'GIANTSTEPS_TEMPO':
        channel = 1 
        im_size = (128, int(chunk_duration * 16000 / 512))  
        num_classes = 3  # slow/medium/fast
        mean = [0.0]
        std = [1.0]
        
        if feature_type == 'combined':
            dst_train = GIANTSTEPS_TEMPO(
                root_dir=os.path.join(data_path, 'GIANTSTEPS_TEMPO'),
                feature_type='combined',
                split='train',
                chunk_duration=chunk_duration
            )
            dst_test = GIANTSTEPS_TEMPO(
                root_dir=os.path.join(data_path, 'GIANTSTEPS_TEMPO'),
                feature_type='melspectrogram',
                split='val',
                chunk_duration=chunk_duration
            )
        else:
            dst_train = GIANTSTEPS_TEMPO(
                root_dir=os.path.join(data_path, 'GIANTSTEPS_TEMPO'),
                feature_type=feature_type,
                split='train',
                chunk_duration=chunk_duration
            )
            dst_test = GIANTSTEPS_TEMPO(
                root_dir=os.path.join(data_path, 'GIANTSTEPS_TEMPO'),
                feature_type=feature_type,
                split='val',
                chunk_duration=chunk_duration
            )
        class_names = ['slow (<120bpm)', 'medium (120-140bpm)', 'fast (>140bpm)']

    elif dataset == "EmbeddingsDataset_AUDIO_MNIST":
        channel = 1
        im_size = (128, 128)
        num_classes = 10
        mean = [0.0]
        std = [1.0]

        # Load GTZAN dataset manually
        dst_train = EmbeddingsDataset(root_dir="data", dataset_name="AUDIO_MNIST", split="train")
        dst_test = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), split ='val', hop_length=512)
        class_names = [str(i) for i in range(num_classes)]
    else:
        exit('unknown dataset: %s'%dataset)
    

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    


def load_synthetic_data(path, device='cpu'):
    """Load synthetic prototypes"""
    data = torch.load(path, weights_only=False)
    synth_data = data['data'][0][0].to(device)
    synth_labels = data['data'][0][1].to(device)
    return synth_data, synth_labels

def load_real_data(dataset_class, root_dir, split, model, device='cpu', max_samples=None):
    """Load real data and extract embeddings"""
    dataset = dataset_class(root_dir=root_dir, split=split)
    real_embs = []
    real_labels = []
    
    for i, (data, label) in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        if i % 1000 == 0:
            print(f"Processing {i}th sample")
        emb = model.embed(data.unsqueeze(0).to(device)).detach().cpu().numpy()
        real_embs.append(emb)
        real_labels.append(label)
        
    return np.concatenate(real_embs, axis=0), np.array(real_labels)

class EmbeddingsDataset(Dataset):
    """
    Base class for datasets using AST embeddings
    """
    def __init__(self, 
                 root_dir: str, 
                 dataset_name: str = "GTZAN",
                 split: str = "train",
                 sample_rate: int = 16000, 
                 embeddings_file: str = None,
                 embedding_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 cache_dir: str = "./embeddings_cache",
                 samples_per_class: int = None):
        """
        Base dataset class for audio embeddings datasets.

        Args:
            root_dir (str): Path to the dataset directory.
            dataset_name (str): Name of the dataset ("GTZAN" or "MoodsMIREX").
            split (str): Dataset split. Options: "train", "val", "all", or "subset".
            sample_rate (int): Sampling rate for audio files.
            embeddings_file (str): Optional path to a file with pre-computed embeddings.
            embedding_model (str): Name of the model to use for embeddings.
            cache_dir (str): Directory to cache embeddings.
            samples_per_class (int): Number of samples per class when split="subset".
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.samples_per_class = samples_per_class
        
        # These attributes will be set by child classes
        self.embeddings = None
        self.all_labels = None
        self.indices = None
        
        # Setup embeddings
        self._setup_embeddings(embeddings_file)
    
    def _setup_embeddings(self, embeddings_file):
        """Set up the dataset for embeddings"""
        if self.dataset_name not in ["GTZAN", "MoodsMIREX", "AUDIO_MNIST"]:
            raise ValueError("dataset_name must be either 'GTZAN' or 'MoodsMIREX'")
        
        if embeddings_file is None:
            # Default embeddings file based on model
            model_name = self.embedding_model.replace('/', '_')
            embeddings_file = self.cache_dir / f"{self.dataset_name.lower()}_embeddings_{model_name}.pkl"
        else:
            embeddings_file = Path(embeddings_file)
        
        # Check if embeddings file exists
        if embeddings_file.exists():
            print(f"Loading embeddings from {embeddings_file}")
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.all_labels = data['labels']
                self.label_to_idx = data.get('label_to_idx', {label: label for label in np.unique(self.all_labels)})
        else:
            print(f"Embeddings file not found. Generating embeddings...")
            # Create embedder and extract embeddings
            embedder = Embedder(model_name=self.embedding_model, cache_dir=self.cache_dir)
            if self.dataset_name == "GTZAN":
                self.embeddings, self.all_labels, self.label_to_idx = embedder.get_embeddings_GTZAN(
                    base_path=self.root_dir, force_recalculate=True
                )
            elif self.dataset_name == "MoodsMIREX":
                self.embeddings, self.all_labels, self.label_to_idx = embedder.get_embeddings_MoodsMIREX(
                    base_path=self.root_dir, force_recalculate=True
                )
            elif self.dataset_name == "AUDIO_MNIST":
                self.embeddings, self.all_labels = embedder.get_embeddings_AUDIO_MNIST(
                    base_path=self.root_dir, force_recalculate=True
                )

        self.class_names = list(self.label_to_idx.keys())
        # Create indices based on split type
        self._create_indices()

    def _create_indices(self):
        """
        Create indices based on the specified split
        """
        if self.split == "all":
            # Use all data
            self.indices = torch.arange(len(self.all_labels))
        elif self.split == "subset":
            # Create a balanced subset with samples_per_class samples per class
            if self.samples_per_class is None:
                raise ValueError("samples_per_class must be specified when split='subset'")
            self._create_subset_indices()
        else:
            # Create train/val split
            self._split_indices()
    
    def _create_subset_indices(self):
        """
        Create indices for a balanced subset with samples_per_class samples per class,
        but only selecting from the training portion of the data.
        """
        # First, create train/val split by class
        class_indices = {}
        train_indices_by_class = {}
        
        # Get indices for each class
        for class_name in self.class_names:
            class_idx = self.label_to_idx[class_name]
            class_indices[class_name] = torch.where(self.all_labels == class_idx)[0]
            
            # Split into train/val (80/20)
            split_idx = int(0.8 * len(class_indices[class_name]))
            # Only store the training indices for sampling
            train_indices_by_class[class_name] = class_indices[class_name][:split_idx].tolist()
        
        # Now sample from training indices only
        subset_indices = []
        for class_name, indices in train_indices_by_class.items():
            if len(indices) < self.samples_per_class:
                print(f"Warning: Class {class_name} has only {len(indices)} training samples, "
                    f"which is less than the requested {self.samples_per_class}. "
                    f"Using all available training samples for this class.")
                subset_indices.extend(indices)
            else:
                # Sample randomly without replacement from training set
                sampled_indices = random.sample(indices, self.samples_per_class)
                subset_indices.extend(sampled_indices)
        
        # Convert to tensor
        self.indices = torch.tensor(subset_indices)
        
        print(f"Created subset dataset with {len(self.indices)} samples "
            f"({len(class_indices)} classes with ~{self.samples_per_class} samples each), "
            f"sampled only from training data")

    def _split_indices(self):
        """
        Create train/val split indices ensuring class balance
        """
        # Create indices for splits by class
        class_indices = {}
        for class_name in self.class_names:
            class_idx = self.label_to_idx[class_name]
            class_indices[class_name] = torch.where(self.all_labels == class_idx)[0]
        
        # Split by class to ensure balance
        train_indices = []
        val_indices = []
        for class_name, indices in class_indices.items():
            split_idx = int(0.8 * len(indices))
            train_indices.append(indices[:split_idx])
            val_indices.append(indices[split_idx:])
        
        # Combine indices
        train_indices = torch.cat(train_indices)
        val_indices = torch.cat(val_indices)
        
        # Select appropriate indices based on split
        if self.split == "train":
            self.indices = train_indices
        else:  # "val"
            self.indices = val_indices
 
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Get embedding and label directly
        index = self.indices[idx]
        embedding = self.embeddings[index]
        label = self.all_labels[index]
        return embedding, label


if __name__ == "__main__":

    # dataset_subset = UrbanSound8KDataset(
    #     root_dir="data/URBANSOUND8K",
    #     split="subset",
    #     sample_rate=16000,
    #     feature_type="mfcc",
    #     folds=[1, 2, 3, 4, 5, 6, 7, 8],  # Use first 8 folds for training
    # )
    # print(f"Dataset length: {len(dataset_subset)}")
    # for i in range(5):
    #     features, label = dataset_subset[i]
    #     print(f"Sample {i}: features shape {features.shape}, label {label}")

    # Create training dataset (50 songs per class)
    train_dataset = GIANTSTEPS_TEMPO(
        root_dir="data/GIANTSTEPS_TEMPO",
        split='train',
        chunk_duration=3.0,
        feature_type='melspectrogram'
    )

    # Create validation dataset (remaining songs)
    val_dataset = GIANTSTEPS_TEMPO(
        root_dir="data/GIANTSTEPS_TEMPO",
        split='val',
        chunk_duration=3.0,
        feature_type='melspectrogram'
    )

    # Check split information
    print("Training set info:")
    print(train_dataset.get_split_info())
    print(f"Training chunks: {len(train_dataset)}")

    print("\nValidation set info:")
    print(val_dataset.get_split_info())
    print(f"Validation chunks: {len(val_dataset)}")

    # Test a sample
    chunk, label = train_dataset[0]
    print(f"\nSample chunk shape: {chunk.shape}")
    print(f"Sample label: {label} ({train_dataset.classes[label]})")

