# prototype_analysis.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

class PrototypeAnalyzer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def get_embeddings(self, data):
        """Extract embeddings from model"""
        with torch.no_grad():
            embeddings = self.model.embed(data.to(self.device))
        return embeddings.detach().cpu().numpy()
    
    def plot_embeddings(self, real_embs, real_labels, synth_embs=None, synth_labels=None, 
                        label_names=None, title="t-SNE visualization", figsize=(8, 8)):
        """Plot 2D embeddings with t-SNE"""
        all_embs = np.concatenate((real_embs, synth_embs), axis=0) if synth_embs is not None else real_embs
        all_embs = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
        
        tsne = TSNE(n_components=2)
        all_embs_2d = tsne.fit_transform(all_embs)
        
        if synth_embs is not None:
            synth_embs_2d = all_embs_2d[len(real_embs):]
            real_embs_2d = all_embs_2d[:len(real_embs)]
        else:
            real_embs_2d = all_embs_2d
            
        fig, ax = plt.subplots(figsize=figsize)
        unique_labels = np.unique(real_labels)
        cmap = plt.cm.get_cmap('viridis', len(unique_labels))
        
        # Plot real data
        for i, label in enumerate(unique_labels):
            mask = real_labels == label
            ax.scatter(real_embs_2d[mask, 0], real_embs_2d[mask, 1], 
                      color=cmap(i), alpha=0.6, label=label_names[i] if label_names else label, marker='o')
            
            # Plot synthetic data if available
            if synth_embs is not None and synth_labels is not None:
                synth_mask = synth_labels == label
                ax.scatter(synth_embs_2d[synth_mask, 0], synth_embs_2d[synth_mask, 1], 
                          color=cmap(i), alpha=0.6, marker='*', s=300, edgecolors='black',
                          label=f"Synth {label_names[i]}" if label_names else f"Synth {label}")
                
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig, ax
    
    def plot_confusion_matrix(self, test_labels, predicted_labels, label_names=None, 
                             title="Confusion Matrix", cmap=plt.cm.Blues, figsize=(8, 6)):
        """Plot confusion matrix"""
        cm = confusion_matrix(test_labels, predicted_labels, labels=np.unique(test_labels))
        display_labels = label_names if label_names is not None else np.unique(test_labels)
        
        plt.figure(figsize=figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=cmap, ax=plt.gca())
        plt.title(title)
        plt.show()
        
    def evaluate_prototypes(self, prototypes, prototype_labels, test_embs, test_labels):
        """Evaluate prototype classifier"""
        classifier = PrototypeClassifier(prototypes, prototype_labels)
        predicted_labels = classifier.predict_batch(test_embs)
        accuracy = accuracy_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        return accuracy, f1, predicted_labels


class PrototypeClassifier:
    def __init__(self, prototypes, prototype_labels):
        self.prototypes = prototypes
        self.prototype_labels = prototype_labels
        
    def predict(self, embedding):
        """Predict label for single embedding"""
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        cosine_similarities = F.cosine_similarity(embedding.unsqueeze(0), self.prototypes)
        softmax_scores = F.softmax(cosine_similarities, dim=0)
        most_similar_index = torch.argmax(softmax_scores)
        return self.prototype_labels[most_similar_index], softmax_scores
    
    def predict_batch(self, embeddings):
        """Predict labels for multiple embeddings"""
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
        cosine_similarities = F.cosine_similarity(embeddings.unsqueeze(1), self.prototypes.unsqueeze(0), dim=2)
        softmax_scores = F.softmax(cosine_similarities, dim=1)
        most_similar_indices = torch.argmax(softmax_scores, dim=1)
        return self.prototype_labels[most_similar_indices]