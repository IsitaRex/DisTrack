# DisTrackted 🎧📢
This repository explores Dataset Distillation via Distribution Matching across diverse audio contexts, including music, speech, and sound events, making it easier to train models with reduced computational costs while maintaining performance.

## Use Cases
* **🎵 Music** – Distill large music datasets for genre classification or synthesis.
* **🗣 Speech** – Create compact versions of speech datasets (e.g., for keyword spotting).
* **🔊 Sound Events** – Improve efficiency in environmental sound recognition tasks.


## Getting Started 
```bash
git clone https://github.com/yourusername/DisTrackted.git
cd DisTrackted
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
<audio controls>
  <source src="synthetic_sounds/sample_5_0.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

## UrbanSound8K
<audio controls>
  <source src="path/to/urbansound8k_sample.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

## GTZAN
<audio controls>
  <source src="path/to/gtzan_sample.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>