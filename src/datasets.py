import os
import torch
import torchaudio
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from src.embedder import Embedder
from pathlib import Path
import pickle
import random

MNIST_MEL_SPEC = (128, 33)
MNIST_MFCC = (13, 33)
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
    def __init__(self, root_dir: str, split="train", train_size=0.80, sample_rate=16000, feature_type = 'melspectrogram', hop_length=512, n_mfcc=13, n_mels=128):
        self.root_dir = root_dir
        self.file_list = []
        self.label_list = []
        self.sample_rate = sample_rate
        self.feature_type = feature_type.lower()

        for dirname, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename[-3:] == "wav":
                    self.file_list.append(os.path.join(dirname, filename))
                    self.label_list.append(int(filename[0]))
    
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
        second_dim = MNIST_MEL_SPEC[1]
        _, n_mels, time = features.shape
        
        if time < second_dim:
            # Pad with zeros
            pad = torch.zeros((1, n_mels, second_dim - time))
            features = torch.cat([features, pad], dim=2)
    
        # if self.feature_type == 'mfcc':
        #     # compute first order and second order deltas
        #     mfcc_first = torchaudio.functional.compute_deltas(features, win_length=3)
        #     mfcc_second = torchaudio.functional.compute_deltas(mfcc_first, win_length=3)
        #     # concatenate them
        #     features = torch.cat([features, mfcc_first, mfcc_second], dim=1)
        #     # compute mean accross time
        #     features = torch.mean(features, dim=2, keepdim=False)
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

        features = self.transform(waveform)
        
        if self.feature_type == 'mfcc':
            # compute first order and second order deltas
            mfcc_first = torchaudio.functional.compute_deltas(features, win_length=3)
            mfcc_second = torchaudio.functional.compute_deltas(mfcc_first, win_length=3)
            # concatenate them
            features = torch.cat([features, mfcc_first, mfcc_second], dim=1)
            # compute mean accross time
            features = torch.mean(features, dim=2, keepdim=False)

        return features, torch.tensor(label, dtype=torch.long)
    
        return torch.stack(chunks)
def get_dataset(dataset, data_path, feature_type='melspectrogram'):
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
        im_size =  MNIST_MEL_SPEC if feature_type == 'melspectrogram' else MNIST_MFCC
        # im_size =  MNIST_MEL_SPEC 
        num_classes = 10
        mean = [0.5]
        std = [0.5]
        # Load GTZAN dataset manually
        dst_train = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), feature_type= feature_type, split = 'train', hop_length=512)
        dst_test = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), feature_type= feature_type, split ='val', hop_length=512)
        class_names = [str(i) for i in range(num_classes)]
    elif dataset == 'URBANSOUND8K':
        channel = 1
        im_size = (128, 128)
        num_classes = 10
        mean = [0.5]
        std = [0.5]
        dst_train = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), feature_type= feature_type, split='train')

        dst_test = UrbanSound8KDataset(root_dir=os.path.join(data_path, 'URBANSOUND8K'), split='val')
        class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
    elif dataset == "EmbeddingsDataset_AUDIO_MNIST":
        channel = 1
        im_size = (128, 128)
        num_classes = 10
        mean = [0.5]
        std = [0.5]

        # Load GTZAN dataset manually
        dst_train = EmbeddingsDataset(root_dir="data", dataset_name="AUDIO_MNIST", split="train")
        dst_test = AudioMNISTDataset(root_dir=os.path.join(data_path, 'AUDIO_MNIST'), split ='val', hop_length=512)
        class_names = [str(i) for i in range(num_classes)]
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

    dataset_subset = AudioMNISTDataset(root_dir="data/AUDIO_MNIST", split='train', feature_type='melspectrogram', hop_length=512)
    print(f"Number of samples in subset dataset: {len(dataset_subset)}")
    mx = 0
    for i in range(len(dataset_subset)):
        mx = max(mx, int(dataset_subset[i][0].shape[1]))
    print(f"Max feature length in subset dataset: {mx}")