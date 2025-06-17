import torch

class DistillationLosses:
    """
    Implements Vanilla DM, Joint Matching, and Modality Gap Matching losses.
    """
    @staticmethod
    def vanilla_loss(real_features, synthetic_features):
        """Vanilla unimodal distribution matching loss."""
        mean_real_features = torch.mean(real_features, dim=0)
        mean_synthetic_features = torch.mean(synthetic_features, dim=0)
        return torch.sum((mean_real_features - mean_synthetic_features)**2)

    @staticmethod
    def joint_matching_loss(real_audio, real_text, synthetic_audio, synthetic_text):
        """Joint Matching Loss."""
        real_joint = torch.mean(real_audio, dim=0) + torch.mean(real_text, dim=0)
        synthetic_joint = torch.mean(synthetic_audio, dim=0) + torch.mean(synthetic_text, dim=0)
        return torch.norm(real_joint - synthetic_joint, p=2)
    @staticmethod
    def modality_gap_loss(real_audio, real_text, synthetic_audio, synthetic_text):
        """Modality Gap Matching Loss."""
        Dat = torch.mean(real_audio, dim = 0) + torch.mean(synthetic_text, dim = 0)
        Dta = torch.mean(real_text, dim = 0) + torch.mean(synthetic_audio, dim = 0)
        return torch.norm(Dat - Dta, p=2)
    

if __name__ == "__main__":
    real_features = torch.randn(100, 512)
    synthetic_features = torch.randn(10, 512)

    loss = DistillationLosses.vanilla_loss(real_features, synthetic_features)
    print(f"Vanilla DM Loss: {loss.item()}")

    real_audio = torch.randn(100, 512)
    real_text = torch.randn(100, 512)
    synthetic_audio = torch.randn(10, 512)
    synthetic_text = torch.randn(10, 512)

    loss = DistillationLosses.joint_matching_loss(real_audio, real_text, synthetic_audio, synthetic_text)
    print(f"Joint Matching Loss: {loss.item()}")

    loss = DistillationLosses.modality_gap_loss(real_audio, real_text, synthetic_audio, synthetic_text)
    print(f"Modality Gap Matching Loss: {loss.item()}")