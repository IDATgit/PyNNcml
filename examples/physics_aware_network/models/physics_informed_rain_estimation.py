import torch
from torch import nn
from .attanuation_baseline_removline import AttenuationBaselineRemoval
from .attanuation_to_rain_rate import AttenuationToRainRate

class PhysicsInformedRainEstimation(nn.Module):
    """
    A wrapper model that combines the two-stage process:
    1. Attenuation baseline removal.
    2. Conversion of the compensated attenuation to rain rate.
    """
    def __init__(self, metadata_features_baseline, style_vector_dim_baseline,
                 metadata_features_rain_rate, style_vector_dim_rain_rate):
        super().__init__()
        
        self.baseline_remover = AttenuationBaselineRemoval(
            metadata_features=metadata_features_baseline,
            style_vector_dim=style_vector_dim_baseline
        )
        
        self.rain_rate_estimator = AttenuationToRainRate(
            metadata_features=metadata_features_rain_rate,
            style_vector_dim=style_vector_dim_rain_rate
        )

    def forward(self, attenuation_signal, metadata):
        """
        Forward pass for the combined model.
        
        Args:
            attenuation_signal (torch.Tensor): The input attenuation signal of shape [B, 1, L].
            metadata (torch.Tensor): The corresponding metadata for the links.
        
        Returns:
            torch.Tensor: The final estimated rain rate.
            torch.Tensor: The intermediate rain-related attenuation signal.
        """
        # Step 1: Estimate and remove the baseline attenuation
        baseline_attenuation = self.baseline_remover(attenuation_signal, metadata)
        rain_related_attenuation = attenuation_signal - baseline_attenuation
        
        # Ensure the compensated attenuation is non-negative
        rain_related_attenuation = torch.relu(rain_related_attenuation)
        
        # Step 2: Convert the compensated attenuation to rain rate
        # The rain rate model expects a single value per sample, but our signal is a time series.
        # We process each time step individually by reshaping the batch.
        
        B, C, L = rain_related_attenuation.shape
        # Reshape for processing: (B, L, C) -> (B * L, C)
        rain_related_attenuation_reshaped = rain_related_attenuation.permute(0, 2, 1).reshape(B * L, C)
        
        # Expand metadata to match the new batch dimension: (B, F) -> (B, L, F) -> (B * L, F)
        metadata_features = metadata.shape[1]
        metadata_expanded = metadata.unsqueeze(1).expand(-1, L, -1).reshape(B * L, metadata_features)

        # Estimate rain rate on the reshaped tensor
        estimated_rain_rate_reshaped = self.rain_rate_estimator(rain_related_attenuation_reshaped, metadata_expanded)

        # Reshape back to sequence format: (B * L, C) -> (B, L, C) -> (B, C, L)
        estimated_rain_rate = estimated_rain_rate_reshaped.reshape(B, L, C).permute(0, 2, 1)
        
        return estimated_rain_rate, rain_related_attenuation 