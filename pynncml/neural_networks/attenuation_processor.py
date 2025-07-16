import torch
import torch.nn as nn
from pynncml import neural_networks
from pynncml.neural_networks.normalization import InputNormalization


class AttenuationProcessor(nn.Module):
    """
    Dynamic non-rain-attenuation removal network with GAN-style metadata fusion.
    
    This fully-connected network takes a 1D attenuation signal and metadata as input,
    and outputs a processed 1D signal of the same shape with non-rain attenuation removed.
    
    :param normalization_cfg: InputNormalizationConfig which holds the normalization parameters.
    :param n_samples: int that represents the number of samples in the input signal.
    :param hidden_size: int that represents the hidden layer size.
    :param metadata_input_size: int that represents the metadata input size.
    :param metadata_n_features: int that represents the metadata feature size.
    :param n_layers: int that represents the number of hidden layers.
    :param dropout_rate: float that represents the dropout rate.
    """

    def __init__(self, 
                 normalization_cfg: neural_networks.InputNormalizationConfig,
                 n_samples: int,
                 hidden_size: int = 256,
                 metadata_input_size: int = 2,
                 metadata_n_features: int = 64,
                 n_layers: int = 3,
                 dropout_rate: float = 0.1):

        super(AttenuationProcessor, self).__init__()
        
        self.n_samples = n_samples
        self.hidden_size = hidden_size
        self.metadata_n_features = metadata_n_features
        self.n_layers = n_layers
        
        # Normalization layer
        self.normalization = InputNormalization(normalization_cfg)
        
        # Metadata processing (GAN-style)
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_input_size, metadata_n_features),
            nn.ReLU(),
            nn.Linear(metadata_n_features, metadata_n_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Main signal processing network
        # Input layer: from n_samples to hidden_size
        self.input_layer = nn.Sequential(
            nn.Linear(n_samples, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers
        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.extend([
                nn.Linear(hidden_size + metadata_n_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        # Output layer: back to n_samples
        self.output_layer = nn.Linear(hidden_size + metadata_n_features, n_samples)
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, attenuation: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attenuation processor.
        
        :param attenuation: A tensor of shape [batch_size, n_samples] containing the attenuation signal.
        :param metadata: A tensor of shape [batch_size, metadata_features] containing metadata.
        :return: A tensor of shape [batch_size, n_samples] containing processed attenuation.
        """
        batch_size = attenuation.shape[0]
        
        # Expand attenuation to match normalization expectations: [batch_size, n_samples, 1]
        attenuation_expanded = attenuation.unsqueeze(-1)
        
        # Normalize inputs
        input_tensor, input_meta_tensor = self.normalization(attenuation_expanded, metadata)
        
        # Squeeze back to [batch_size, n_samples]
        input_tensor = input_tensor.squeeze(-1)
        
        # Process metadata (GAN-style)
        meta_features = self.metadata_processor(input_meta_tensor)  # [batch_size, metadata_n_features]
        
        # Process main signal
        x = self.input_layer(input_tensor)  # [batch_size, hidden_size]
        
        # Apply hidden layers with metadata fusion
        for i in range(0, len(self.hidden_layers), 3):  # Every 3 modules (Linear, ReLU, Dropout)
            # GAN-style metadata fusion: concatenate metadata to features
            x_with_meta = torch.cat([x, meta_features], dim=-1)  # [batch_size, hidden_size + metadata_n_features]
            
            # Apply linear layer
            x = self.hidden_layers[i](x_with_meta)  # [batch_size, hidden_size]
            
            # Apply ReLU
            if i + 1 < len(self.hidden_layers):
                x = self.hidden_layers[i + 1](x)
            
            # Apply Dropout
            if i + 2 < len(self.hidden_layers):
                x = self.hidden_layers[i + 2](x)
        
        # Final output with metadata fusion
        x_with_meta = torch.cat([x, meta_features], dim=-1)
        output = self.output_layer(x_with_meta)  # [batch_size, n_samples]
        
        # Residual connection: processed = original + learned_correction
        processed_attenuation = input_tensor + self.residual_weight * output
        
        return processed_attenuation


class SimpleAttenuationProcessor(nn.Module):
    """
    Simplified version of the attenuation processor with direct element-wise processing.
    
    This network processes each time sample independently with metadata conditioning,
    making it more efficient for long sequences.
    
    :param normalization_cfg: InputNormalizationConfig which holds the normalization parameters.
    :param hidden_size: int that represents the hidden layer size.
    :param metadata_input_size: int that represents the metadata input size.
    :param metadata_n_features: int that represents the metadata feature size.
    :param n_layers: int that represents the number of hidden layers.
    :param dropout_rate: float that represents the dropout rate.
    """

    def __init__(self, 
                 normalization_cfg: neural_networks.InputNormalizationConfig,
                 hidden_size: int = 128,
                 metadata_input_size: int = 2,
                 metadata_n_features: int = 32,
                 n_layers: int = 2,
                 dropout_rate: float = 0.1):

        super(SimpleAttenuationProcessor, self).__init__()
        
        self.hidden_size = hidden_size
        self.metadata_n_features = metadata_n_features
        self.n_layers = n_layers
        
        # Normalization layer
        self.normalization = InputNormalization(normalization_cfg)
        
        # Metadata processing (GAN-style)
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_input_size, metadata_n_features),
            nn.ReLU(),
            nn.Linear(metadata_n_features, metadata_n_features),
            nn.ReLU()
        )
        
        # Point-wise processing network (processes each sample independently)
        layers = []
        
        # Input layer: 1 (attenuation value) + metadata_n_features -> hidden_size
        layers.append(nn.Linear(1 + metadata_n_features, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer: hidden_size -> 1 (correction value)
        layers.append(nn.Linear(hidden_size, 1))
        
        self.processor = nn.Sequential(*layers)
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, attenuation: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the simple attenuation processor.
        
        :param attenuation: A tensor of shape [batch_size, n_samples] containing the attenuation signal.
        :param metadata: A tensor of shape [batch_size, metadata_features] containing metadata.
        :return: A tensor of shape [batch_size, n_samples] containing processed attenuation.
        """
        batch_size, n_samples = attenuation.shape
        
        # Expand attenuation to match normalization expectations: [batch_size, n_samples, 1]
        attenuation_expanded = attenuation.unsqueeze(-1)
        
        # Normalize inputs
        input_tensor, input_meta_tensor = self.normalization(attenuation_expanded, metadata)
        
        # Process metadata (GAN-style)
        meta_features = self.metadata_processor(input_meta_tensor)  # [batch_size, metadata_n_features]
        
        # Expand metadata to match time dimension
        meta_expanded = meta_features.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, metadata_n_features]
        
        # Concatenate attenuation with metadata for each time step
        combined_input = torch.cat([input_tensor, meta_expanded], dim=-1)  # [batch_size, n_samples, 1 + metadata_n_features]
        
        # Process each time step with the same network
        # Reshape to process all time steps in parallel
        combined_flat = combined_input.view(-1, 1 + self.metadata_n_features)  # [batch_size * n_samples, 1 + metadata_n_features]
        
        # Apply processing network
        corrections_flat = self.processor(combined_flat)  # [batch_size * n_samples, 1]
        
        # Reshape back
        corrections = corrections_flat.view(batch_size, n_samples)  # [batch_size, n_samples]
        
        # Apply residual connection: processed = original + learned_correction
        input_tensor_squeezed = input_tensor.squeeze(-1)  # [batch_size, n_samples]
        processed_attenuation = input_tensor_squeezed + self.residual_weight * corrections
        
        return processed_attenuation 