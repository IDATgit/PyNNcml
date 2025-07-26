import torch
from torch import nn
from .simple_baseline_removal import SimpleBaselineRemoval
from .attanuation_to_rain_rate import AttenuationToRainRate

class PhysicsInformedRainEstimation(nn.Module):
    """
    A wrapper model that combines the two-stage process:
    1. Attenuation baseline removal.
    2. Conversion of the compensated attenuation to rain rate.
    """
    def __init__(self, metadata_features, window_hours=24):
        super().__init__()
        
        self.baseline_remover = SimpleBaselineRemoval(
            metadata_features=metadata_features,
            window_hours=window_hours
        )
        
        self.rain_rate_estimator = AttenuationToRainRate(
            metadata_features=metadata_features
        )

    def forward(self, attenuation_signal, metadata):
        """
        Forward pass for the combined model.
        
        Args:
            attenuation_signal (torch.Tensor): The input attenuation signal of shape [B, 1, L].
            metadata (torch.Tensor): The corresponding metadata for the links.shape is [B, metadata_features]
        
        Returns:
            rain_related_attenuation (torch.Tensor): The intermediate rain-related attenuation signal. shape is [B, 1, L]
            estimated_rain_rate (torch.Tensor): The final estimated rain rate. shape is [B, 1, L]
        """
        # Step 1: Estimate and remove the baseline attenuation
        baseline_attenuation = self.baseline_remover(attenuation_signal, metadata)
        rain_related_attenuation = attenuation_signal - baseline_attenuation
        
        # Step 2: Convert the compensated attenuation to rain rate
        # The rain rate model expects a single value per sample, but our signal is a time series.
        # We process each time step individually by reshaping the batch.
        
        B, C, L = rain_related_attenuation.shape
        
        # Estimate rain rate on the reshaped tensor
        estimated_rain_rate = self.rain_rate_estimator(rain_related_attenuation, metadata)

        # Apply ReLU during evaluation to ensure physically meaningful outputs
        if not self.training:
            # Compensated attenuation should be non-negative (no negative rain-related attenuation)
            rain_related_attenuation = torch.relu(rain_related_attenuation)
            # Rain rate should be non-negative (no negative rainfall)
            estimated_rain_rate = torch.relu(estimated_rain_rate)
        
        return rain_related_attenuation, estimated_rain_rate