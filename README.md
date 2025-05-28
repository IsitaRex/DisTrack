# DisTrackted 🎧📢
This repository explores distillation techniques across diverse audio contexts, including music, speech, and sound events, making it easier to train models with reduced computational costs while maintaining performance.

## Use Cases
* **🎵 Music** – Distill large music datasets for genre classification or synthesis.
* **🗣 Speech** – Create compact versions of speech datasets (e.g., for keyword spotting).
* **🔊 Sound Events** – Improve efficiency in environmental sound recognition tasks.

## Techniques
* **Distribution Matching:** Aligns synthetic and real data distributions in feature space for broad representational fidelity.
* **Gradient Matching:** Optimizes synthetic samples to mimic the training dynamics (gradients) of real data, improving learning efficiency.
* **Trajectory Matching:**  Matches the long-term optimization path of models trained on real vs. distilled data, enhancing generalization.

## Getting Started 
```bash
git clone https://github.com/yourusername/DisTrackted.git
cd DisTrackted
pip install -r requirements.txt
```

# Results

## Distribution Matching
|  |AUDIO MNIST | Bird Detection | GTZAN | 
 :-: | :-: | :-: | :-: |
| 1 samples/cls  |0.4|  | |
| 10 samples/cls |0.6 |  | | 
| 50 samples/cls |  |  | | 

