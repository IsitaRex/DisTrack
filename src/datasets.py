import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchaudio.prototype.transforms import ChromaSpectrogram
from abc import abstractmethod
from sklearn.model_selection import train_test_split

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
        elif self.feature_type == 'chromagram':
            self.transform = ChromaSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=2048,
                hop_length=hop_length,
                n_chroma=12
            )
        elif self.feature_type == 'log-melspectrogram':

            print('SELECTED LOG-MEL')
            self.transform = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=hop_length
                ),
                torchaudio.transforms.AmplitudeToDB(),
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
            spec_features = self.spec_transform(waveform)
            mfcc_features = self.mfcc_transform(waveform)
            return spec_features, mfcc_features
        else:
            features = self.transform(waveform)

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
class TinySOL(AudioDataset):
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
        self.max_length = int(sample_rate * 4)
        
        # # Generate chunk indices
        # self.chunk_indices = self._generate_chunk_indices()

    def _load_dataset(self):
        """Load Medley-solos dataset metadata and prepare file paths/labels."""
        # Load metadata
        metadata = pd.read_csv(os.path.join(self.root_dir, 'annotation', 'TinySOL_metadata.csv'))
        # Transform to 12 pitch classes
        metadata['Pitch_transformed'] = metadata['Pitch'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        # Get instrument classes
        classes = metadata['Pitch_transformed'].unique().tolist()
        label_map = {cls: idx for idx, cls in enumerate(classes)}

        # Collect all file paths and labels
        file_paths = []
        labels = []
        for _, row in metadata.iterrows():
            file_path = os.path.join(self.root_dir, 'audio', row['Path'])
            file_paths.append(file_path)
            labels.append(label_map[row['Pitch_transformed']])
        
        # Split into train/val using train_test_split
        if self.split == 'train':
            file_paths, _, labels, _ = train_test_split(file_paths, labels, train_size=0.8, stratify=labels, random_state=42)
        elif self.split == 'val':
            _, file_paths, _, labels = train_test_split(file_paths, labels, train_size=0.8, stratify=labels, random_state=42)

        return classes, label_map, file_paths, labels
        

    # def _generate_chunk_indices(self):
    #     """Generate indices for audio chunks."""
    #     chunk_indices = []
        
    #     for idx, file_path in enumerate(self.file_paths):

    #         try:
    #             # Load audio to determine number of chunks
    #             waveform = self._load_audio(file_path)
    #             features = self._extract_features(waveform)
                
    #             if isinstance(features, tuple):  # For combined features
    #                 features = features[0]  # Use spectrogram for chunk calculation
                
    #             # Calculate number of chunks
    #             total_frames = features.shape[2]
    #             n_chunks = total_frames // self.frames_per_chunk
                
    #             # Only add chunks if we have at least one complete chunk
    #             if n_chunks > 0:
    #                 for chunk_idx in range(n_chunks):
    #                     chunk_indices.append((idx, chunk_idx))
                        
    #         except Exception as e:
    #             print(f"  Warning: Could not process {file_path}: {e}")
    #             continue
        
    #     return chunk_indices

    # def _get_chunk(self, features, chunk_idx):
    #     """Extract a chunk from features."""
    #     start = chunk_idx * self.frames_per_chunk
    #     end = start + self.frames_per_chunk
    #     return features[:, :, start:end]

    # def __len__(self):
    #     return len(self.chunk_indices)

    # def __getitem__(self, idx):
    #     file_idx, chunk_idx = self.chunk_indices[idx]
        
    #     # Load original audio
    #     audio = self._load_audio(self.file_paths[file_idx])
        
    #     # Extract features
    #     if self.feature_type == 'combined':
    #         spec_features, mfcc_features = self._extract_features(audio)
    #         spec_chunk = self._get_chunk(spec_features, chunk_idx)
    #         mfcc_chunk = self._get_chunk(mfcc_features, chunk_idx)
            
    #         # Ensure chunks have correct size (pad if necessary)
    #         if spec_chunk.shape[2] < self.frames_per_chunk:
    #             pad_size = self.frames_per_chunk - spec_chunk.shape[2]
    #             spec_chunk = torch.nn.functional.pad(spec_chunk, (0, pad_size))
    #             mfcc_chunk = torch.nn.functional.pad(mfcc_chunk, (0, pad_size))
                
    #         return spec_chunk, mfcc_chunk, self.labels[file_idx]
    #     else:
    #         features = self._extract_features(audio)
    #         chunk = self._get_chunk(features, chunk_idx)
            
    #         # Pad if necessary
    #         if chunk.shape[2] < self.frames_per_chunk:
    #             pad_size = self.frames_per_chunk - chunk.shape[2]
    #             chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
    #         return chunk, self.labels[file_idx]
        
def get_dataset(dataset, data_path, feature_type='melspectrogram', batch_size=256, chunk_duration=3.0):
    
    if dataset == 'AUDIO_MNIST':
        channel = 1
        im_size =  MNIST_MEL_SPEC if feature_type in ['melspectrogram', 'log-melspectrogram'] else MNIST_MFCC
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
        freq_dim = 128 if feature_type != 'chromagram' else 12
        im_size = (freq_dim, 128)  
        # im_size = URBANSOUND_MEL_SPEC if feature_type in ['melspectrogram', 'log-melspectrogram'] else URBANSOUND_MFCC
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

    elif dataset == 'TinySOL':
        channel = 1 
        freq_dim = 128 if feature_type != 'chromagram' else 12
        im_size = (freq_dim, 126)  
        num_classes = 12
        mean = [0.0]
        std = [1.0]
        
        if feature_type == 'combined':
            dst_train = TinySOL(root_dir=os.path.join(data_path, 'TINY_SOL'),feature_type='combined',split='train',chunk_duration=chunk_duration)
            dst_test = TinySOL(root_dir=os.path.join(data_path, 'TINY_SOL'),feature_type='melspectrogram',split='val',chunk_duration=chunk_duration)
        else:
            dst_train = TinySOL(root_dir=os.path.join(data_path, 'TINY_SOL'),feature_type=feature_type,split='train',chunk_duration=chunk_duration)
            dst_test = TinySOL(root_dir=os.path.join(data_path, 'TINY_SOL'),feature_type=feature_type, split='val',chunk_duration=chunk_duration)
        class_names = dst_train.classes

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

if __name__ == "__main__":

    train_dataset = UrbanSound8KDataset(
        root_dir="data/URBANSOUND8K",
        split='train',
        feature_type='melspectrogram',
        sample_rate=16000
    )

    print(f"Train dataset length: {len(train_dataset)}")
    for i in range(5):
        features, label = train_dataset[i]
        print(f"Sample {i}: features shape {features.shape}, label {label}")

    # Create validation dataset (remaining songs)
    val_dataset = UrbanSound8KDataset(
        root_dir="data/URBANSOUND8K",
        split='train',
        feature_type='log-melspectrogram',
        sample_rate=16000
    )

    # Display one mel spectrogram from validation set
    import matplotlib.pyplot as plt
    features, label = train_dataset[0]
    plt.imshow(features[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Mel-Spectrogram - Label: {label}")
    plt.colorbar()
    plt.show()

    # Display one log-mel spectrogram from validation set
    features, label = val_dataset[0]
    plt.imshow(features[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Log Mel-Spectrogram - Label: {label}")
    plt.colorbar()
    plt.show()