import os
import torch
import torchaudio
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