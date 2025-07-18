import torch
from torch import nn

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization Layer.
    The output of this layer is calculated as:
    y = scale * (x - mean) / std + bias
    where scale and bias are learned from the style vector.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, style):
        """
        Forward pass for AdaIN.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).
            style (torch.Tensor): Style tensor of shape (B, 2*C).
        """
        B, C, L = x.shape
        style_reshaped = style.view(B, C, 2)
        scale = style_reshaped[:, :, 0].unsqueeze(-1)
        bias = style_reshaped[:, :, 1].unsqueeze(-1)

        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True) + 1e-6
        
        return scale * (x - mu) / sigma + bias

class MetadataMLP(nn.Module):
    """
    MLP to process metadata and generate style vectors for AdaIN layers.
    """
    def __init__(self, input_features, output_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_features)
        )
    
    def forward(self, x):
        return self.net(x)

class AttenuationBaselineRemoval(nn.Module):
    """
    CNN-based model to estimate and remove the baseline from an attenuation signal,
    using a StyleGAN-like approach for metadata fusion via AdaIN.
    """
    def __init__(self, metadata_features, style_vector_dim):
        super().__init__()
        
        # Total channels for all AdaIN layers
        # C_out from Conv1, Conv2, Conv3, Conv4, Conv5, Conv6
        total_channels = 32 + 64 + 64 + 128 + 128 + 64
        self.metadata_mlp = MetadataMLP(metadata_features, total_channels * 2) # *2 for scale and bias
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding='same')
        self.adain1 = AdaIN()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.adain2 = AdaIN()
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=15, padding='same')
        self.adain3 = AdaIN()
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=100, padding='same')
        self.adain4 = AdaIN()
        
        self.conv5 = nn.Conv1d(128, 128, kernel_size=900, padding='same')
        self.adain5 = AdaIN()
        
        self.conv6 = nn.Conv1d(128, 64, kernel_size=1, padding='same')
        self.adain6 = AdaIN()
        
        self.conv7 = nn.Conv1d(64, 1, kernel_size=1, padding='same')
        
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, metadata):
        style_vectors = self.metadata_mlp(metadata)
        
        # Split style vectors for each AdaIN layer
        s1, s2, s3, s4, s5, s6 = torch.split(style_vectors, [32*2, 64*2, 64*2, 128*2, 128*2, 64*2], dim=1)

        x = self.conv1(x)
        x = self.adain1(x, s1)
        
        x = self.conv2(x)
        x = self.adain2(x, s2)
        
        x = self.conv3(x)
        x = self.adain3(x, s3)
        
        x = self.conv4(x)
        x = self.adain4(x, s4)
        
        x = self.conv5(x)
        x = self.adain5(x, s5)
        
        x = self.conv6(x)
        x = self.adain6(x, s6)
        
        x = self.conv7(x)
        
        output = self.relu(x)
        
        return output

