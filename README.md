# DisTrack 🎧📢
Data distillation compresses the knowledge of large datasets into smaller, synthetic datasets that retain essential information to train models able to carry out downstream tasks. While extensively studied in the image domain, this technique remains underexplored for audio. In this work, we demonstrate that dataset distillation via distribution matching applied to image-like audio representations (e.g., mel-spectrograms, chromagrams), is capable of preserving key acoustic features such as phonetic structure, timbre, and pitch. Furthermore, we show that distillation can generate not only image-like representations but also raw audio tracks (waveform samples). We also investigate temporal condensation, revealing a trade-off between the number of synthetic tracks per class and their duration: models trained on more brief synthetic tracks (e.g., 1-second tracks) can match the performance of those trained on fewer but longer samples (e.g., 3-second tracks), suggesting that increasing the quantity of shorter tracks can achieve performance comparable to fewer longer ones. By enabling models to learn from synthetic audio representations, this work expands the applicability of data distillation in audio AI. This field holds unique potential to address challenges like data privacy by decoupling model training from proprietary or sensitive source material.

# Tasks and Datasets

## **Spoken Digit Classification** 1️⃣🗣️
- **Dataset**: [AudioMNIST](https://github.com/soerenab/AudioMNIST)  
- **Description**:  
  This task evaluates the preservation of phonetic content and basic speech patterns in distilled samples. The dataset consists of 30,000 audio samples of spoken digits (0-9) recorded by 60 speakers.  
- **Key Features**:  
  - Phonetic structure  
  - Speaker-independent digit recognition  

## **Urban Sound Recognition** 🔊🚨
- **Dataset**: [UrbanSound8K](https://urbansounddataset.weebly.com/)  
- **Description**:  
  This task assesses the retention of timbral qualities and broad acoustic signatures. The dataset includes 8,732 annotated samples across 10 urban sound categories (e.g., sirens, drilling, street music).  
- **Key Features**:  
  - Timbral texture  
  - Environmental sound classification  

## **Musical Instrument Identification** 🎸🎺
- **Dataset**: [Medley-solos-DB](https://zenodo.org/record/1344103)  
- **Description**:  
  This task focuses on identifying harmonic structures and attack-decay characteristics of musical instruments. The dataset contains 21,571 audio clips from 8 instruments (e.g., flute, violin, piano).  
- **Key Features**:  
  - Harmonic content  
  - Instrument-specific timbre  

## **Pitch Detection** 🎼🎤
- **Dataset**: [TinySOL](https://zenodo.org/record/3685367)  
- **Description**:  
  This task evaluates the accuracy of fundamental frequency (F0) estimation in distilled samples. The dataset comprises 2,913 single-note excerpts from 14 orchestral instruments.  
- **Key Features**:  
  - Pitch class invariance  
  - Frequency resolution  



# Getting Started 
```bash
git clone https://github.com/yourusername/DisTrackt.git
cd DisTrack
pip install -r requirements.txt
```

## Dataset Distillation in Spectral Domain
To run experiments to generate synthetic data in Spectral Domain run
```bash
python DistributionMatching.py --dataset AUDIO_MNIST --feature_type melspectrogram --ipc 10
```

## Dataset Distillation in Waveform Domain
To run experiments to generate synthetic data in Spectral Domain run
```bash
python DistributionMatching_wav.py --dataset AUDIO_MNIST --feature_type melspectrogram --ipc 10
```
# Results

## Distribution Matching

### Training random CNN
|  |AUDIO MNIST | UrbanSound8K | GTZAN | 
 :-: | :-: | :-: | :-: |
| 1 samples/cls  |0.4|0.25 | 0.19|
| 10 samples/cls |0.6 | 0.31 | 0.4| 
### Prototype networks using random CNN as latent space
|  |AUDIO MNIST | UrbanSound8K | GTZAN | 
 :-: | :-: | :-: | :-: |
| Mean of Clusters |0.59| ||
| 1 samples/cls  |0.59| | |
| 10 samples/cls |0.48|  | | 


# Sample Sounds 🎧📢

## Audio MNIST  🔢
https://github.com/user-attachments/assets/2b35c000-6090-490c-b80f-8a1df760953f
