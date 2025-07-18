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
    def __init__(self, metadata_features, style_vector_dim):
        super().__init__()
        
        # C_out from FC1, FC2, FC3, FC4
        total_channels = 128 + 128 + 128 + 128
        self.metadata_mlp = MetadataMLP(metadata_features, total_channels * 2)

        self.fc1 = nn.Linear(1, 128)
        self.adain1 = AdaIN()
        
        self.fc2 = nn.Linear(128, 128)
        self.adain2 = AdaIN()
        
        self.fc3 = nn.Linear(128, 128)
        self.adain3 = AdaIN()
        
        self.fc4 = nn.Linear(128, 128)
        self.adain4 = AdaIN()
        
        self.fc5 = nn.Linear(128, 1)

        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, metadata):
        style_vectors = self.metadata_mlp(metadata)
        
        # Split style vectors for each AdaIN layer
        s1, s2, s3, s4 = torch.split(style_vectors, [128*2, 128*2, 128*2, 128*2], dim=1)
        x = self.fc1(x)
        x = self.adain1(x, s1)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.adain2(x, s2)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.adain3(x, s3)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.adain4(x, s4)
        x = self.relu(x)

        x = self.fc5(x)
        
        output = self.relu(x)
        
        return output

