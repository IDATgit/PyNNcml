import torch
from torch import nn
import torch.nn.functional as F

class SimpleBaselineRemoval(nn.Module):
    """
    Simple baseline removal model inspired by REBS (Robust Extraction of Baseline Signal).
    
    Key principles:
    1. Long-term baseline estimation using robust temporal aggregation
    2. Small metadata-based corrections
    3. Robust to outliers (rain events)
    4. Much simpler than CNN+AdaIN approaches
    
    Input shapes:
    - attenuation: (batch_size, 1, time_samples)
    - metadata: (batch_size, metadata_features)
    
    Output shape:
    - baseline: (batch_size, 1, time_samples)
    """
    
    def __init__(self, metadata_features, window_hours=24):
        super().__init__()
        
        self.window_hours = window_hours
        self.samples_per_minute = 6  # 10-second samples
        self.window_samples = window_hours * 60 * self.samples_per_minute
        
        # Learnable weights for temporal aggregation (replaces hard median)
        # Initialize with declining weights (recent samples get slightly more weight)
        initial_weights = torch.exp(-torch.arange(self.window_samples, dtype=torch.float32) / (self.window_samples / 4))
        self.temporal_weights = nn.Parameter(initial_weights)
        
        # Small correction network based on metadata
        self.correction_net = nn.Sequential(
            nn.Linear(metadata_features + 1, 32),  # +1 for baseline magnitude
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Output between -1 and 1 for correction factor
        )
        
        # Scale factor for corrections (start small)
        self.correction_scale = nn.Parameter(torch.tensor(0.05))  # Max 5% correction initially
        
    def robust_temporal_baseline(self, attenuation):
        """
        Vectorized computation of robust baseline using learnable temporal weights.
        Uses 1D convolution for efficient sliding window operation.
        """
        batch_size, channels, time_samples = attenuation.shape
        
        # Normalize weights to sum to 1
        weights = F.softmax(self.temporal_weights, dim=0)
        
        # Prepare weights for convolution (flip for proper convolution vs correlation)
        conv_weights = weights.flip(0).view(1, 1, -1)  # (out_channels=1, in_channels=1, kernel_size)
        
        # Pad the input to handle boundaries properly
        padding = self.window_samples // 2
        padded_input = F.pad(attenuation, (padding, padding), mode='replicate')
        
        # Apply 1D convolution - this is the vectorized sliding window operation
        baseline = F.conv1d(padded_input, conv_weights, padding=0)
        
        # Ensure output has the same shape as input
        if baseline.shape[-1] != time_samples:
            # Trim or pad to match exact input size
            if baseline.shape[-1] > time_samples:
                excess = baseline.shape[-1] - time_samples
                start_trim = excess // 2
                baseline = baseline[:, :, start_trim:start_trim + time_samples]
            else:
                # Should not happen with correct padding, but safety check
                baseline = F.pad(baseline, (0, time_samples - baseline.shape[-1]), mode='replicate')
        
        return baseline
    
    def forward(self, attenuation, metadata):
        """
        Forward pass: compute baseline = robust_temporal_baseline + small_corrections
        """
        batch_size, _, time_samples = attenuation.shape
        
        # 1. Compute robust long-term baseline
        baseline_long = self.robust_temporal_baseline(attenuation)
        
        # 2. Compute metadata-based corrections
        # Use mean baseline magnitude as additional input
        baseline_magnitude = baseline_long.mean(dim=-1)  # (batch_size, 1)
        
        # Prepare input for correction network
        correction_input = torch.cat([
            baseline_magnitude,  # (batch_size, 1)
            metadata  # (batch_size, metadata_features)
        ], dim=1)
        
        # Get correction factor for each sample in batch
        correction_factor = self.correction_net(correction_input)  # (batch_size, 1)
        
        # Apply correction (broadcast across time dimension)
        correction = self.correction_scale * correction_factor.unsqueeze(-1)  # (batch_size, 1, 1)
        
        # 3. Final baseline with small corrections
        baseline_final = baseline_long * (1 + correction)
        
        return baseline_final

class AdaptiveMedianBaseline(nn.Module):
    """
    Alternative simpler approach: Adaptive median with learnable window sizes
    """
    
    def __init__(self, metadata_features):
        super().__init__()
        
        # Learnable window size predictor
        self.window_predictor = nn.Sequential(
            nn.Linear(metadata_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.min_window_hours = 4
        self.max_window_hours = 48
        
    def forward(self, attenuation, metadata):
        """
        Compute baseline using adaptive median windows based on metadata.
        Vectorized implementation using unfold for efficiency.
        """
        batch_size, channels, time_samples = attenuation.shape
        
        # Predict window size for each sample
        window_factor = self.window_predictor(metadata)  # (batch_size, 1)
        window_hours = self.min_window_hours + window_factor.squeeze(-1) * (self.max_window_hours - self.min_window_hours)
        window_samples = (window_hours * 60 * 6).long()  # Convert to sample count
        
        # For simplicity, use the maximum window size for all samples
        # This allows for vectorization while still being adaptive per sample
        max_window = window_samples.max().item()
        max_window = min(max_window, time_samples)  # Don't exceed sequence length
        
        # Pad input for boundary handling
        padding = max_window // 2
        padded_input = F.pad(attenuation, (padding, padding), mode='replicate')
        
        # Create sliding windows using unfold
        # unfold(dimension, size, step) creates overlapping windows
        windows = padded_input.unfold(dimension=2, size=max_window, step=1)
        # Shape: (batch_size, channels, time_samples, max_window)
        
        # Compute median for each window (this is vectorized across all windows)
        baseline = torch.median(windows, dim=-1)[0]  # (batch_size, channels, time_samples)
        
        return baseline


def day_median_baseline_removal_fixed(attenuation):
    """
    Fixed version of the day_median_baseline_removal function with proper error handling
    """
    batch_size, _, time_samples = attenuation.shape
    attenuation_flat = attenuation.view(batch_size, -1)  # (batch_size, time_samples)
    
    samples_per_day = 60 * 24 * 6  # 10-second samples per day
    
    if time_samples <= samples_per_day:
        # If we have less than a day of data, use global median
        baseline = torch.median(attenuation_flat, dim=-1, keepdim=True)[0]
        baseline = baseline.unsqueeze(-1).expand(-1, 1, time_samples)
    else:
        # Original logic for longer sequences
        num_full_days = time_samples // samples_per_day
        residual_samples = time_samples % samples_per_day
        
        baseline = torch.zeros_like(attenuation)
        
        # Process full days
        if num_full_days > 0:
            full_days_data = attenuation_flat[:, :num_full_days * samples_per_day]
            full_days_reshaped = full_days_data.reshape(batch_size, num_full_days, samples_per_day)
            daily_medians = torch.median(full_days_reshaped, dim=-1)[0]  # (batch_size, num_full_days)
            
            # Expand daily medians to full time series
            for day in range(num_full_days):
                start_idx = day * samples_per_day
                end_idx = (day + 1) * samples_per_day
                baseline[:, :, start_idx:end_idx] = daily_medians[:, day:day+1].unsqueeze(-1)
        
        # Process residual samples
        if residual_samples > 0:
            residual_data = attenuation_flat[:, num_full_days * samples_per_day:]
            residual_median = torch.median(residual_data, dim=-1)[0]  # (batch_size,)
            baseline[:, :, num_full_days * samples_per_day:] = residual_median.unsqueeze(-1).unsqueeze(-1)
    
    return baseline 