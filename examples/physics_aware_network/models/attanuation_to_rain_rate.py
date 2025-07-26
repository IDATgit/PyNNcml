import torch
from torch import nn

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, style):
        B, C = x.shape
        style_reshaped = style.view(B, C, 2)
        scale = style_reshaped[:, :, 0]
        bias = style_reshaped[:, :, 1]

        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True) + 1e-6
        
        return scale * (x - mu) / sigma + bias

class MetadataMLP(nn.Module):
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

class AttenuationToRainRate(nn.Module):
    def __init__(self, metadata_features):
        super().__init__()
        
        # C_out from FC1, FC2, FC3, FC4 - now using 8 dimensions instead of 128
        total_channels = 8 + 8 + 8 + 8
        self.metadata_mlp = MetadataMLP(metadata_features, total_channels * 2)

        self.fc1 = nn.Linear(1, 8)
        self.adain1 = AdaIN()
        
        self.fc2 = nn.Linear(8, 8)
        self.adain2 = AdaIN()
        
        self.fc3 = nn.Linear(8, 8)
        self.adain3 = AdaIN()
        
        self.fc4 = nn.Linear(8, 8)
        self.adain4 = AdaIN()
        
        self.fc5 = nn.Linear(8, 1)

        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, metadata):
        # Input shapes:
        # x: (batch_size, 1, time_samples)
        # metadata: (batch_size, metadata_features)
        
        batch_size, channels, time_samples = x.shape
        
        # Generate style vectors from metadata
        style_vectors = self.metadata_mlp(metadata)  # (batch_size, total_channels * 2)
        
        # Split style vectors for each AdaIN layer - now 8*2=16 each
        s1, s2, s3, s4 = torch.split(style_vectors, [8*2, 8*2, 8*2, 8*2], dim=1)
        
        # Reshape x to process all time samples at once
        # (batch_size, time_samples, 1) -> (batch_size * time_samples, 1)
        x = x.permute(0, 2, 1) # (batch_size, 1, time_samples) -> (batch_size, time_samples, 1)
        x = x.reshape(batch_size * time_samples, channels)
        
        # Apply first layer
        x = self.fc1(x)  # (batch_size * time_samples, 8)
        
        # For AdaIN, we need to repeat style vectors for each time sample
        # s1: (batch_size, 16) -> (batch_size * time_samples, 16)
        s1_expanded = s1.unsqueeze(1).expand(-1, time_samples, -1).reshape(batch_size * time_samples, -1)
        x = self.adain1(x, s1_expanded)
        x = self.relu(x)

        # Repeat for all layers
        x = self.fc2(x)  # (batch_size * time_samples, 8)
        s2_expanded = s2.unsqueeze(1).expand(-1, time_samples, -1).reshape(batch_size * time_samples, -1)
        x = self.adain2(x, s2_expanded)
        x = self.relu(x)

        x = self.fc3(x)  # (batch_size * time_samples, 8)
        s3_expanded = s3.unsqueeze(1).expand(-1, time_samples, -1).reshape(batch_size * time_samples, -1)
        x = self.adain3(x, s3_expanded)
        x = self.relu(x)

        x = self.fc4(x)  # (batch_size * time_samples, 8)
        s4_expanded = s4.unsqueeze(1).expand(-1, time_samples, -1).reshape(batch_size * time_samples, -1)
        x = self.adain4(x, s4_expanded)
        x = self.relu(x)

        x = self.fc5(x)  # (batch_size * time_samples, 1)
        
        output = self.relu(x)
        
        # Reshape back to original time series format
        # (batch_size * time_samples, 1) -> (batch_size, time_samples, 1)
        output = output.reshape(batch_size, time_samples, 1)
        output = output.permute(0, 2, 1) # (batch_size, time_samples, 1) -> (batch_size, 1, time_samples)
        
        return output

