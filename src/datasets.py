import os
import torch
import torchaudio
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir: str, split="train", train_size=0.80, sample_rate=16000):
        self.root_dir = root_dir
        self.file_list = []
        self.label_list = []
        self.sample_rate = sample_rate
        for dirname, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename[-3:] == "wav":
                    self.file_list.append(os.path.join(dirname, filename))
                    self.label_list.append(int(filename[0]))
        
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
        )
        
        total_len = len(self.file_list)
        
        if split == 'train':
            self.file_list, self.label_list = self.file_list[:int(train_size * total_len)], self.label_list[:int(train_size * total_len)]
        elif split == 'val':
            self.file_list, self.label_list = self.file_list[int(train_size * total_len):], self.label_list[int(train_size * total_len):]
                    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_list[idx])
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        features = self.transform(waveform)
        
        # Current shape is [n_mels, time], we want [128, 128]
        _, n_mels, time = features.shape
        
        # Option 1: Truncate or pad time dimension to 128
        if time < 128:
            # Pad with zeros
            pad = torch.zeros((1, n_mels, 128 - time))
            features = torch.cat([features, pad], dim=2)
        
        return features, self.label_list[idx]
    
    def __len__(self):
        return len(self.file_list)
    
class UrbanSound8KDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", sample_rate: int = 16000, 
                 feature_type: str = "melspectrogram", n_mels: int = 128, 
                 n_mfcc: int = 13, folds=None):
        """
        UrbanSound8K Dataset with support for MelSpectrogram and MFCC feature extraction.

        Args:
            root_dir (str): Path to the UrbanSound8K directory containing 'audio' and 'metadata' folders.
            split (str): Dataset split. Options: "train" or "val".
            sample_rate (int): Sampling rate for audio files.
            feature_type (str): Type of feature to extract. Options: "melspectrogram" or "mfcc".
            n_mels (int): Number of mel bands for MelSpectrogram.
            n_mfcc (int): Number of MFCC features to extract.
            chunk_size (int): Size of the feature map (chunk_size x chunk_size).
            folds (list): List of folds to include. If None, uses standard train/val split.
        """
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.feature_type = feature_type.lower()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Load metadata
        metadata_path = os.path.join(root_dir, 'metadata', 'UrbanSound8K.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Create label map from class names
        self.classes = sorted(self.metadata['class'].unique())
        self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Filter folds based on split if folds not specified
        if folds is None:
            if split == "train":
                folds = [1, 2, 3, 4, 5, 6, 7, 8]  # First 8 folds for training
            elif split == "val":
                folds = [9, 10]  # Last 2 folds for validation
            else:
                raise ValueError("Invalid split. Choose 'train' or 'val'.")
        
        # Filter metadata for selected folds
        self.metadata = self.metadata[self.metadata['fold'].isin(folds)]
        
        # Collect all file paths and labels
        self.file_paths = []
        self.labels = []
        for _, row in self.metadata.iterrows():
            fold = row['fold']
            fname = row['slice_file_name']
            file_path = os.path.join(root_dir, 'audio', f'fold{fold}', fname)
            self.file_paths.append(file_path)
            self.labels.append(self.label_map[row['class']])
        
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
        
        # For padding/truncating audio files
        self.max_length = int(sample_rate * 4.09)  # Most UrbanSound8K clips are 4 seconds or less

    def __len__(self):
        return len(self.file_paths)  # Return 3 chunks per file

    def __getitem__(self, idx):
        """Loads an audio file, extracts features, and returns a tensor."""
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

        # Pad/truncate to consistent length
        if waveform.shape[1] < self.max_length:
            # Pad with zeros
            pad_size = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[1] > self.max_length:
            # Truncate
            waveform = waveform[:, :self.max_length]

        # Extract features
        features = self.transform(waveform)

        return features, torch.tensor(label, dtype=torch.long)
    
        return torch.stack(chunks)
def get_dataset(dataset, data_path):
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
        mean = [0.5]
        std = [0.5]

        # Load GTZAN dataset manually
        dst_train = GTZANDataset(root_dir=os.path.join(data_path, 'GTZAN'), split = 'train')
        dst_test = GTZANDataset(root_dir=os.path.join(data_path, 'GTZAN'), split ='val')
        class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    elif dataset == 'AUDIO_MNIST':
        channel = 1
        im_size = (128, 128)
        num_classes = 10
        mean = [0.5]
        std = [0.5]
        # Load GTZAN dataset manually
        dst_train = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), split = 'train')
        dst_test = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), split ='val')
        class_names = [str(i) for i in range(num_classes)]
    elif dataset == 'URBANSOUND8K':
        channel = 1
        im_size = (128, 128)
        num_classes = 10
        mean = [0.5]
        std = [0.5]
        dst_train = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), split='train')

        dst_test = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), split='val')
        class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
    else:
        exit('unknown dataset: %s'%dataset)
    

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    
if __name__ == "__main__":
    # load BIRD dataset
    dataset = UrbanSound8KDataset(root_dir='data/URBANSOUND8K', split='train', sample_rate=16000, feature_type='melspectrogram')
    print(f"Number of samples: {len(dataset)}")
    for i in range(5):
        features, label = dataset[i]
        print(f"Sample {i}: features shape {features.shape}, label {label.item()}")


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