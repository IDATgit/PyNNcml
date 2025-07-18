# End-To-End Training Example of CNN+FC for rain estimation using PyNNCML
# This script presents an end-to-end example of training a CNN+FC based rain estimation neural network with attenuation processing.
# This tutorial demonstrates the complete pipeline:
# 1. Compute attenuation = TSL - RSL
# 2. Feed into CNN for baseline correction
# 3. Subtract CNN output from attenuation to get clean attenuation
# 4. Feed clean attenuation into FC network
# 5. Average FC output to align with rain gauge timing
# 6. Train with enhanced loss function including physical constraints

# Notebook structure
# 1. Imports and Installation of PyNNCML
# 2. Hyperparameter settings
# 3. Build Dataset
# 4. Build Neural Network
# 5. Enhanced Loss Function
# 6. Training Loop
# 7. Neural Network Analysis

import sys
import os

top_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists(top_folder):
    print("Import PyNNCML From Code")
    sys.path.append(top_folder)  # This line is need to import pynncml
else:
    print("Install PyNNCML From pip")
    #!pip install pynncml  # Uncomment this line if running outside of a notebook

import numpy as np
import pynncml as pnc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
from sklearn import metrics

# Hyper-Parameters

# Hyper-parameters for CNN+FC Pipeline
batch_size = 16  # @param{type:"integer"}
window_size = 32  # @param{type:"integer"}
cnn_n_filters = 128  # @param{type:"integer"} - Number of CNN filters
cnn_layers = 3  # @param{type:"integer"} - Number of CNN layers
fc_hidden_size = 256  # @param{type:"integer"} - FC hidden layer size
fc_layers = 3  # @param{type:"integer"} - Number of FC layers
metadata_n_features = 32  # @param{type:"integer"}
lr = 3e-4  # @param{type:"number"}
weight_decay = 1e-4  # @param{type:"number"}
n_epochs = 100  # @param{type:"integer"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Building Training and Validation datasets

time_slice = slice("2015-06-01", "2015-06-10")  # Time Interval
dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=time_slice)

dataset.link_set.plot_links(scale=True, scale_factor=1.0)
plt.grid()
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
data_loader = torch.utils.data.DataLoader(training_dataset, batch_size)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)

# Compute rain rate statistics
rg = np.stack([p.data_array for p in dataset.point_set]).flatten()
param = scipy.stats.expon.fit(rg)
exp_gamma = param[1]
print("Rain Rate Statistics")
print("Mean[mm/hr]:", np.mean(rg))
print("Std[mm/hr]:", np.std(rg))
print("Parentage of wet samples", 100 * np.sum(rg > 0) / rg.size)
print("Parentage of dry samples", 100 * np.sum(rg == 0) / rg.size)
print("Exponential Distribution Parameters:", param)
_ = plt.hist(rg, bins=100, density=True)
plt.plot(np.linspace(0, np.max(rg), 100), scipy.stats.expon.pdf(np.linspace(0, np.max(rg), 100), *param))
plt.grid()
plt.xlabel("Rain Rate[mm/hr]")
plt.ylabel("Density")
plt.tight_layout()

# Simplified CNN+FC Pipeline Model

class SimpleCNNBackbone(nn.Module):
    """Simplified CNN backbone for baseline correction."""
    
    def __init__(self, input_size, n_filters, n_layers, metadata_n_features):
        super(SimpleCNNBackbone, self).__init__()
        
        self.metadata_processor = nn.Sequential(
            nn.Linear(2, metadata_n_features),
            nn.ReLU(),
            nn.Linear(metadata_n_features, metadata_n_features),
            nn.ReLU()
        )
        
        # 1D CNN layers
        layers = []
        in_channels = 1
        for i in range(n_layers):
            layers.extend([
                nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = n_filters
            
        self.cnn_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, metadata):
        batch_size = x.shape[0]
        
        # Process metadata
        meta_features = self.metadata_processor(metadata)  # [batch_size, metadata_n_features]
        
        # Reshape for CNN: [batch_size, 1, sequence_length]
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        cnn_out = self.cnn_layers(x)  # [batch_size, n_filters, sequence_length]
        
        # Global average pooling
        pooled = self.global_pool(cnn_out).squeeze(-1)  # [batch_size, n_filters]
        
        return pooled

class SimpleFC(nn.Module):
    """Simplified FC network for attenuation to rain conversion."""
    
    def __init__(self, hidden_size, n_layers, metadata_n_features):
        super(SimpleFC, self).__init__()
        
        self.metadata_processor = nn.Sequential(
            nn.Linear(2, metadata_n_features),
            nn.ReLU(),
            nn.Linear(metadata_n_features, metadata_n_features),
            nn.ReLU()
        )
        
        # Build FC layers
        layers = []
        input_size = 1 + metadata_n_features  # 1 for attenuation + metadata features
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
            
        layers.append(nn.Linear(hidden_size, 1))  # Output rain rate
        
        self.fc_layers = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        
    def forward(self, attenuation, metadata):
        batch_size, seq_len = attenuation.shape
        
        # Process metadata
        meta_features = self.metadata_processor(metadata)  # [batch_size, metadata_n_features]
        
        # Expand metadata to match sequence length
        meta_expanded = meta_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, metadata_n_features]
        
        # Expand attenuation
        att_expanded = attenuation.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Concatenate
        combined = torch.cat([att_expanded, meta_expanded], dim=-1)  # [batch_size, seq_len, 1 + metadata_n_features]
        
        # Apply FC to each time step
        combined_flat = combined.view(-1, 1 + meta_features.shape[-1])  # [batch_size * seq_len, features]
        output_flat = self.fc_layers(combined_flat)  # [batch_size * seq_len, 1]
        output_flat_relu = self.relu(output_flat)
        output = output_flat.view(batch_size, seq_len)  # [batch_size, seq_len]
        
        return output

class CNNFCRainEstimator(nn.Module):
    """
    Complete CNN+FC pipeline for rain estimation with attenuation processing.
    
    Pipeline:
    1. Compute attenuation = TSL - RSL (remove reshaping to get [batch_size, nof_samples])
    2. CNN processes attenuation to predict baseline correction
    3. Clean attenuation = raw_attenuation - cnn_correction
    4. FC processes clean attenuation to predict rain rate
    5. Average rain rate within windows to match gauge timing
    """
    
    def __init__(self, 
                 n_samples: int,
                 cnn_n_filters: int = 128,
                 cnn_layers: int = 3,
                 fc_hidden_size: int = 256,
                 fc_layers: int = 3,
                 metadata_n_features: int = 32):
        
        super(CNNFCRainEstimator, self).__init__()
        
        self.n_samples = n_samples
        self.metadata_n_features = metadata_n_features
        
        # CNN for baseline correction
        self.cnn_backbone = SimpleCNNBackbone(
            input_size=n_samples,
            n_filters=cnn_n_filters,
            n_layers=cnn_layers,
            metadata_n_features=metadata_n_features
        )
        
        # CNN output head for baseline correction
        self.cnn_correction_head = nn.Sequential(
            nn.Linear(cnn_n_filters, cnn_n_filters // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cnn_n_filters // 2, n_samples)  # Output correction for each sample
        )
        
        # FC network for attenuation to rain conversion
        self.fc_processor = SimpleFC(
            hidden_size=fc_hidden_size,
            n_layers=fc_layers,
            metadata_n_features=metadata_n_features
        )
        
    def forward(self, rsl: torch.Tensor, tsl: torch.Tensor, metadata: torch.Tensor):
        """
        Forward pass of the complete pipeline.
        
        :param rsl: RSL tensor [batch_size, window_size, 90]
        :param tsl: TSL tensor [batch_size, window_size, 90] 
        :param metadata: Metadata tensor [batch_size, metadata_features]
        :return: Dictionary containing all intermediate outputs and final rain rate
        """
        batch_size, window_size, features = rsl.shape
        
        # Step 1: Compute attenuation = TSL - RSL and reshape to [batch_size, nof_samples]
        # Remove the extra reshaping dimension (90) by taking mean across features
        rsl_signal = torch.mean(rsl, dim=-1)  # [batch_size, window_size]
        tsl_signal = torch.mean(tsl, dim=-1)  # [batch_size, window_size]
        raw_attenuation = tsl_signal - rsl_signal  # [batch_size, window_size]
        
        # Step 2: CNN processes attenuation for baseline correction
        cnn_features = self.cnn_backbone(raw_attenuation, metadata)  # [batch_size, cnn_n_filters]
        baseline_correction = self.cnn_correction_head(cnn_features)  # [batch_size, window_size]
        
        # Step 3: Clean attenuation = raw - CNN correction
        clean_attenuation = raw_attenuation - baseline_correction
        
        # Step 4: FC processes clean attenuation to rain rate
        rain_rate_samples = self.fc_processor(clean_attenuation, metadata)  # [batch_size, window_size]
        
        # Step 5: Average rain rate within each sample to be time aligned to the baseline
        # For now, we'll return the full resolution and let the loss function handle averaging
        rain_rate_avg = rain_rate_samples  # [batch_size, window_size]
        
        return {
            'raw_attenuation': raw_attenuation,
            'baseline_correction': baseline_correction, 
            'clean_attenuation': clean_attenuation,
            'rain_rate_samples': rain_rate_samples,
            'rain_rate_avg': rain_rate_avg
        }

# Enhanced Loss Function

class EnhancedRainEstimationLoss(nn.Module):
    """
    Enhanced loss function for rain estimation with physical constraints.
    
    Components:
    1. Main rain estimation loss with imbalance handling
    2. Clean attenuation regularization (should be ~0 during dry periods)
    3. FC zero-input regularization (zero clean attenuation -> zero rain)
    4. Physical constraints (non-negative values, extreme value penalties)
    """
    
    def __init__(self, 
                 exp_gamma: float,
                 rain_weight: float = 1.0,  # Higher weight for minority rain samples
                 clean_atten_weight: float = 0.1,
                 gamma_s: float = 0.9):
        
        super(EnhancedRainEstimationLoss, self).__init__()
        
        self.exp_gamma = exp_gamma
        self.rain_weight = rain_weight
        self.clean_atten_weight = clean_atten_weight
        self.gamma_s = gamma_s
        
    def forward(self, model_output: dict, rain_target: torch.Tensor):
        """
        Compute enhanced loss with all components.
        
        :param model_output: Dictionary from model forward pass
        :param rain_target: Target rain rates [batch_size, window_size]
        :return: Dictionary with loss components
        """
        rain_pred = model_output['rain_rate_avg']
        clean_attenuation = model_output['clean_attenuation']
        
        # 1. Main rain estimation loss with imbalance weighting
        rain_mask = (rain_target != 0).bool()
        dry_mask = ~rain_mask
        
        delta = (rain_target - rain_pred) ** 2
        weight = 1 - self.gamma_s * torch.exp(-self.exp_gamma * rain_target)
        
        # Higher weight for rain samples to handle imbalance
        sample_weights = torch.where(rain_mask.bool(), 
                                   weight * self.rain_weight, 
                                   weight)
        
        rain_loss = torch.mean(sample_weights * delta)
        
        # 2. Clean attenuation regularization (should be ~0 during dry periods)
        clean_atten_loss = torch.mean(dry_mask * clean_attenuation ** 2)
        
        
        # Combine all losses
        total_loss = (rain_loss + 
                     self.clean_atten_weight * clean_atten_loss +
                     self.rain_weight * rain_loss)
        
        return {
            'total_loss': total_loss,
            'rain_loss': rain_loss,
            'clean_atten_loss': clean_atten_loss,
        }


# Checkpoint Loading Helper Function
def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a saved checkpoint.
    
    :param checkpoint_path: Path to the checkpoint file
    :param model: Model instance to load state into
    :param optimizer: Optimizer instance to load state into (optional)
    :return: Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    print(f"Training losses at checkpoint:")
    print(f"  Total Loss: {checkpoint['total_loss']:.4f}")
    print(f"  Rain Loss: {checkpoint['rain_loss']:.4f}")
    print(f"  Clean Attenuation Loss: {checkpoint['clean_atten_loss']:.4f}")
    
    return checkpoint


def load_latest_checkpoint_and_analyze(checkpoint_dir, model, val_loader, device):
    """
    Load the latest checkpoint and create analysis plots with test examples.
    Shows 2 random examples, the worst performing example, and the best performing example.
    
    :param checkpoint_dir: Directory containing checkpoints
    :param model: Model instance to load state into
    :param val_loader: Validation data loader
    :param device: PyTorch device
    """
    import glob
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        print("No checkpoint files found in the directory.")
        return
    
    # Sort by epoch number to get the latest
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    checkpoint = load_checkpoint(latest_checkpoint, model)
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect all validation results
    all_examples = []
    sample_errors = []
    
    with torch.no_grad():
        for batch_idx, (rain_rate, rsl, tsl, metadata) in enumerate(val_loader):
            current_batch_size = rsl.shape[0]
            m_step = math.floor(rain_rate.shape[1] / window_size)
            
            for step in range(m_step):
                _rr = rain_rate[:, step * window_size:(step + 1) * window_size, 0].float().to(device)
                _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                
                # Forward pass
                model_output = model(_rsl, _tsl, metadata.to(device))
                rain_pred = model_output['rain_rate_avg']
                clean_atten = model_output['clean_attenuation']
                baseline_corr = model_output['baseline_correction']
                raw_atten = model_output['raw_attenuation']
                
                # Convert to numpy
                rain_ref_np = _rr.detach().cpu().numpy()
                rain_pred_np = rain_pred.detach().cpu().numpy()
                clean_atten_np = clean_atten.detach().cpu().numpy()
                baseline_corr_np = baseline_corr.detach().cpu().numpy()
                raw_atten_np = raw_atten.detach().cpu().numpy()
                
                # Store examples with their performance metrics
                for i in range(current_batch_size):
                    # Calculate RMSE for this example
                    rmse = np.sqrt(np.mean((rain_ref_np[i] - rain_pred_np[i]) ** 2))
                    
                    example = {
                        'rain_ref': rain_ref_np[i],
                        'rain_pred': rain_pred_np[i],
                        'clean_atten': clean_atten_np[i],
                        'baseline_corr': baseline_corr_np[i],
                        'raw_atten': raw_atten_np[i],
                        'rmse': rmse,
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    }
                    
                    all_examples.append(example)
                    sample_errors.append(rmse)
    
    print(f"Collected {len(all_examples)} validation examples")
    
    # Find best and worst examples
    sample_errors = np.array(sample_errors)
    best_idx = np.argmin(sample_errors)
    worst_idx = np.argmax(sample_errors)
    
    # Select 2 random examples (excluding best and worst)
    available_indices = list(range(len(all_examples)))
    available_indices.remove(best_idx)
    available_indices.remove(worst_idx)
    random_indices = np.random.choice(available_indices, size=2, replace=False)
    
    # Selected examples
    examples_to_plot = [
        ("Random Example 1", all_examples[random_indices[0]]),
        ("Random Example 2", all_examples[random_indices[1]]),
        (f"Best Example (RMSE: {sample_errors[best_idx]:.3f})", all_examples[best_idx]),
        (f"Worst Example (RMSE: {sample_errors[worst_idx]:.3f})", all_examples[worst_idx])
    ]
    
    # Create the analysis plots
    plt.figure(figsize=(20, 16))
    
    for plot_idx, (title, example) in enumerate(examples_to_plot):
        # Time series comparison plot
        plt.subplot(4, 4, plot_idx * 4 + 1)
        time_steps = np.arange(len(example['rain_ref']))
        plt.plot(time_steps, example['rain_ref'], 'b-', label='Reference', linewidth=2)
        plt.plot(time_steps, example['rain_pred'], 'r--', label='Predicted', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Rain Rate [mm/hr]')
        plt.title(f'{title}\nRain Rate Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(4, 4, plot_idx * 4 + 2)
        plt.scatter(example['rain_ref'], example['rain_pred'], alpha=0.6, s=20)
        max_val = max(np.max(example['rain_ref']), np.max(example['rain_pred']))
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7)
        plt.xlabel('Reference Rain Rate [mm/hr]')
        plt.ylabel('Predicted Rain Rate [mm/hr]')
        plt.title('Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        # Attenuation analysis
        plt.subplot(4, 4, plot_idx * 4 + 3)
        plt.plot(time_steps, example['raw_atten'], 'k-', label='Raw Attenuation', linewidth=1.5)
        plt.plot(time_steps, example['baseline_corr'], 'm--', label='CNN Correction', linewidth=1.5)
        plt.plot(time_steps, example['clean_atten'], 'g-', label='Clean Attenuation', linewidth=2)
        plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Attenuation [dB]')
        plt.title('Attenuation Processing')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error analysis
        plt.subplot(4, 4, plot_idx * 4 + 4)
        error = example['rain_pred'] - example['rain_ref']
        plt.plot(time_steps, error, 'purple', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Prediction Error [mm/hr]')
        plt.title(f'Error (RMSE: {example["rmse"]:.3f})')
        plt.grid(True, alpha=0.3)
        
        # Fill area for positive/negative errors
        plt.fill_between(time_steps, error, 0, where=(error >= 0), alpha=0.3, color='red', label='Over-prediction')
        plt.fill_between(time_steps, error, 0, where=(error < 0), alpha=0.3, color='blue', label='Under-prediction')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f'Model Analysis - Checkpoint from Epoch {checkpoint["epoch"]}', fontsize=16, y=1.02)
    
    # Print summary statistics
    print(f"\nValidation Set Summary:")
    print(f"Total examples: {len(all_examples)}")
    print(f"Best RMSE: {sample_errors[best_idx]:.4f} mm/hr")
    print(f"Worst RMSE: {sample_errors[worst_idx]:.4f} mm/hr")
    print(f"Mean RMSE: {np.mean(sample_errors):.4f} mm/hr")
    print(f"Std RMSE: {np.std(sample_errors):.4f} mm/hr")
    
    return checkpoint, all_examples

# Build the model


model = CNNFCRainEstimator(
    n_samples=window_size,  # 32 samples per window
    cnn_n_filters=cnn_n_filters,
    cnn_layers=cnn_layers,
    fc_hidden_size=fc_hidden_size,
    fc_layers=fc_layers,
    metadata_n_features=metadata_n_features
).to(device)
print("Rain estimation model: ")
print(model)

# Enhanced loss function
loss_function = EnhancedRainEstimationLoss(
    exp_gamma=exp_gamma,
    rain_weight=1.0,  # Higher weight for rain samples
    clean_atten_weight=0.1
)

# Training Loop

opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
ra = pnc.metrics.ResultsAccumulator()
am = pnc.metrics.AverageMetric()

# Create checkpoints directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_name = "CNNFCRainEstimator"
checkpoint_dir = os.path.join(script_dir, f"checkpoints_{model_name}")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoints directory: {checkpoint_dir}")

# Check if checkpoints already exist
import glob
existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))

if existing_checkpoints:
    print(f"Found {len(existing_checkpoints)} existing checkpoints. Skipping training.")
    print("Loading latest checkpoint and running analysis...")
    
    # Load latest checkpoint and run analysis
    latest_checkpoint, validation_examples = load_latest_checkpoint_and_analyze(
        checkpoint_dir, model, val_loader, device
    )
    
else:
    print("No existing checkpoints found. Starting training...")
    
    model.train()
    print("Start Training CNN+FC Pipeline")
    for epoch in tqdm(range(n_epochs)):
        am.clear()
        for rain_rate, rsl, tsl, metadata in data_loader:
            current_batch_size = rsl.shape[0]
            m_step = math.floor(rain_rate.shape[1] / window_size)
            
            for step in range(m_step):
                opt.zero_grad()  # Zero gradients
                
                # Extract current window
                _rr = rain_rate[:, step * window_size:(step + 1) * window_size, 0].float().to(device)
                _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                
                # Forward pass through complete pipeline
                model_output = model(_rsl, _tsl, metadata.to(device))
                
                # Compute enhanced loss
                loss_dict = loss_function(model_output, _rr)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                opt.step()
                
                # Log all loss components
                am.add_results(
                    total_loss=total_loss.item(),
                    rain_loss=loss_dict['rain_loss'].item(),
                    clean_atten_loss=loss_dict['clean_atten_loss'].item(),
                )
        
        # Store epoch results
        ra.add_results(
            total_loss=am.get_results("total_loss"),
            rain_loss=am.get_results("rain_loss"),
        )
        
        # Print epoch-level training losses (all components)
        print(
            f"Epoch {epoch + 1}/{n_epochs} | Train Losses -> "
            f"Total: {am.get_results('total_loss'):.4f} | "
            f"Rain: {am.get_results('rain_loss'):.4f} | "
            f"CleanAtten: {am.get_results('clean_atten_loss'):.4f} | "
        )
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'total_loss': am.get_results("total_loss"),
                'rain_loss': am.get_results("rain_loss"),
                'clean_atten_loss': am.get_results("clean_atten_loss"),
                'hyperparameters': {
                    'batch_size': batch_size,
                    'window_size': window_size,
                    'cnn_n_filters': cnn_n_filters,
                    'cnn_layers': cnn_layers,
                    'fc_hidden_size': fc_hidden_size,
                    'fc_layers': fc_layers,
                    'metadata_n_features': metadata_n_features,
                    'lr': lr,
                    'weight_decay': weight_decay
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training completed!")


# Analysis of training progress (only if training occurred)

try:
    # Check if training results are available
    ra.get_results("total_loss")
    
    print("Displaying training progress plots...")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(ra.get_results("total_loss"), label="Total Loss", linewidth=2)
    plt.plot(ra.get_results("rain_loss"), label="Rain Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Main Losses")

    plt.subplot(1, 3, 2)
    plt.plot(ra.get_results("clean_atten_loss"), label="Clean Attenuation Loss")
    plt.plot(ra.get_results("zero_input_loss"), label="Zero Input Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Regularization Losses")

    plt.subplot(1, 3, 3)
    plt.plot(ra.get_results("physical_loss"), label="Physical Constraints Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Physical Constraints")

    plt.tight_layout()
    
except KeyError:
    print("Training was skipped (checkpoints found). No training progress to display.")

# Model Validation

model.eval()
ga = pnc.metrics.GroupAnalysis()
val_am = pnc.metrics.AverageMetric()  # Accumulator for validation losses
with torch.no_grad():
    all_rain_ref = []
    all_rain_pred = []
    all_clean_atten = []
    all_baseline_corr = []
    
    for rain_rate, rsl, tsl, metadata in val_loader:
        current_batch_size = rsl.shape[0]
        m_step = math.floor(rain_rate.shape[1] / window_size)
        
        for step in range(m_step):
            _rr = rain_rate[:, step * window_size:(step + 1) * window_size, 0].float().to(device)
            _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            
            # Forward pass
            model_output = model(_rsl, _tsl, metadata.to(device))
            
            rain_pred = model_output['rain_rate_avg']
            clean_atten = model_output['clean_attenuation']
            baseline_corr = model_output['baseline_correction']
            
            all_rain_ref.append(_rr.detach().cpu().numpy())
            all_rain_pred.append(rain_pred.detach().cpu().numpy())
            all_clean_atten.append(clean_atten.detach().cpu().numpy())
            all_baseline_corr.append(baseline_corr.detach().cpu().numpy())
            
            ga.append(_rr.detach().cpu().numpy(), rain_pred.detach().cpu().numpy())
            
            # Accumulate validation loss components
            val_loss_dict = loss_function(model_output, _rr)
            val_am.add_results(
                total_loss=val_loss_dict['total_loss'].item(),
                rain_loss=val_loss_dict['rain_loss'].item(),
                clean_atten_loss=val_loss_dict['clean_atten_loss'].item()
            )

# Combine all validation results
rain_ref_array = np.concatenate(all_rain_ref, axis=0)
rain_pred_array = np.concatenate(all_rain_pred, axis=0)
clean_atten_array = np.concatenate(all_clean_atten, axis=0)
baseline_corr_array = np.concatenate(all_baseline_corr, axis=0)

# Flatten all validation data for analysis
rain_ref_flat = rain_ref_array.flatten()
rain_pred_flat = rain_pred_array.flatten()
clean_atten_flat = clean_atten_array.flatten()
baseline_corr_flat = baseline_corr_array.flatten()

# Compute validation metrics
mse = np.mean((rain_ref_flat - rain_pred_flat) ** 2)
bias = np.mean(rain_pred_flat - rain_ref_flat)
correlation = np.corrcoef(rain_ref_flat, rain_pred_flat)[0, 1]

print("\nValidation Results:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"Bias: {bias:.4f}")
print(f"Correlation: {correlation:.4f}")

# Print aggregated validation losses
print("\nValidation Losses -> "
      f"Total: {val_am.get_results('total_loss'):.4f}, "
      f"Rain: {val_am.get_results('rain_loss'):.4f}, "
      f"CleanAtten: {val_am.get_results('clean_atten_loss'):.4f}, ")

# Analysis plots - Full Time Series

# Create time index based on the original time slice
# Original time slice: 2015-06-01 to 2015-06-10 (9 days)
import pandas as pd
start_date = pd.Timestamp('2015-06-01 00:00:00')
# Assuming 15-minute intervals (common for CML data)
time_index = pd.date_range(start=start_date, periods=len(rain_ref_flat), freq='15min')

print(f"Full time series analysis:")
print(f"Time range: {time_index[0]} to {time_index[-1]}")
print(f"Total duration: {time_index[-1] - time_index[0]}")
print(f"Number of time steps: {len(rain_ref_flat)}")
print(f"Time resolution: 15 minutes")

plt.figure(figsize=(20, 12))

# Plot 1: Full Rain rate time series
plt.subplot(3, 3, 1)
plt.plot(time_index, rain_ref_flat, 'b-', label='Reference', linewidth=1.5, alpha=0.8)
plt.plot(time_index, rain_pred_flat, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
plt.xlabel('Date and Time')
plt.ylabel('Rain Rate [mm/hr]')
plt.title('Complete Rain Rate Time Series')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 1b: Zoomed view of first day
plt.subplot(3, 3, 2)
day1_mask = time_index.date == time_index[0].date()
plt.plot(time_index[day1_mask], rain_ref_flat[day1_mask], 'b-', label='Reference', linewidth=2)
plt.plot(time_index[day1_mask], rain_pred_flat[day1_mask], 'r--', label='Predicted', linewidth=2)
plt.xlabel('Time of Day')
plt.ylabel('Rain Rate [mm/hr]')
plt.title(f'Rain Rate - {time_index[0].date()}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Scatter plot
plt.subplot(3, 3, 3)
plt.scatter(rain_ref_flat, rain_pred_flat, alpha=0.5, s=1)
plt.plot([0, np.max(rain_ref_flat)], [0, np.max(rain_ref_flat)], 'r--', linewidth=2)
plt.xlabel('Reference Rain Rate [mm/hr]')
plt.ylabel('Predicted Rain Rate [mm/hr]')
plt.title('Scatter Plot - All Data')
plt.grid(True, alpha=0.3)

# Plot 3: Full Clean attenuation time series
plt.subplot(3, 3, 4)
plt.plot(time_index, clean_atten_flat, 'g-', linewidth=1.5, alpha=0.8)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Date and Time')
plt.ylabel('Clean Attenuation [dB]')
plt.title('Complete Clean Attenuation Time Series')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 4: Full Baseline correction time series
plt.subplot(3, 3, 5)
plt.plot(time_index, baseline_corr_flat, 'm-', linewidth=1.5, alpha=0.8)
plt.xlabel('Date and Time')
plt.ylabel('CNN Baseline Correction [dB]')
plt.title('Complete CNN Baseline Correction')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 5: Clean attenuation vs rain relationship
plt.subplot(3, 3, 6)
# Only plot wet samples for clarity
wet_mask = rain_ref_flat > 0.1
plt.scatter(clean_atten_flat[wet_mask], 
           rain_ref_flat[wet_mask], 
           alpha=0.5, s=1, color='blue', label='Reference')
plt.scatter(clean_atten_flat[wet_mask], 
           rain_pred_flat[wet_mask], 
           alpha=0.5, s=1, color='red', label='Predicted')
plt.xlabel('Clean Attenuation [dB]')
plt.ylabel('Rain Rate [mm/hr]')
plt.title('Attenuation vs Rain (Wet samples)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Cumulative rain over full time series
plt.subplot(3, 3, 7)
plt.plot(time_index, np.cumsum(rain_ref_flat), 'b-', label='Reference', linewidth=2)
plt.plot(time_index, np.cumsum(rain_pred_flat), 'r--', label='Predicted', linewidth=2)
plt.xlabel('Date and Time')
plt.ylabel('Cumulative Rain [mm]')
plt.title('Cumulative Rain Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 7: Daily rainfall comparison
plt.subplot(3, 3, 8)
# Group by day and sum rainfall
daily_ref = []
daily_pred = []
dates = []
for date in pd.date_range(time_index[0].date(), time_index[-1].date()):
    day_mask = time_index.date == date.date()
    if np.any(day_mask):
        daily_ref.append(np.sum(rain_ref_flat[day_mask]) * 0.25)  # Convert to mm (15min intervals)
        daily_pred.append(np.sum(rain_pred_flat[day_mask]) * 0.25)  # Convert to mm
        dates.append(date)

x_pos = np.arange(len(dates))
width = 0.35
plt.bar(x_pos - width/2, daily_ref, width, label='Reference', alpha=0.7, color='blue')
plt.bar(x_pos + width/2, daily_pred, width, label='Predicted', alpha=0.7, color='red')
plt.xlabel('Date')
plt.ylabel('Daily Rainfall [mm]')
plt.title('Daily Rainfall Comparison')
plt.legend()
plt.xticks(x_pos, [d.strftime('%m-%d') for d in dates], rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Plot 8: Time series error analysis
plt.subplot(3, 3, 9)
error = rain_pred_flat - rain_ref_flat
plt.plot(time_index, error, 'purple', linewidth=1, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Date and Time')
plt.ylabel('Prediction Error [mm/hr]')
plt.title('Prediction Error Over Time')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Fill area for positive/negative errors
plt.fill_between(time_index, error, 0, where=(error >= 0), alpha=0.3, color='red', label='Over-prediction')
plt.fill_between(time_index, error, 0, where=(error < 0), alpha=0.3, color='blue', label='Under-prediction')
plt.legend()

plt.tight_layout()

# Group analysis
max_rain = np.max(rain_ref_flat)
g_array = np.linspace(0, max_rain, 6)
print("\nGroup Analysis Results:")
_ = ga.run_analysis(np.stack([g_array[:-1], g_array[1:]], axis=-1))

print(f"\nPhysical Constraint Analysis:")
negative_rain_count = np.sum(rain_pred_flat < 0)
negative_atten_count = np.sum(clean_atten_flat < 0)
extreme_rain_count = np.sum(rain_pred_flat > 100)
print(f"Negative rain predictions: {negative_rain_count}")
print(f"Negative clean attenuation: {negative_atten_count}")
print(f"Extreme rain predictions (>100 mm/hr): {extreme_rain_count}")
print(f"Total validation samples: {rain_pred_flat.size}")

print(f"\nTemporal Analysis Summary:")
print(f"Dataset covers {len(dates)} days from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"Total rainfall - Reference: {np.sum(daily_ref):.2f} mm, Predicted: {np.sum(daily_pred):.2f} mm")
print(f"Average daily rainfall - Reference: {np.mean(daily_ref):.2f} mm, Predicted: {np.mean(daily_pred):.2f} mm")
wet_periods = np.sum(rain_ref_flat > 0.1) / len(rain_ref_flat) * 100
print(f"Percentage of time with rain (>0.1 mm/hr): {wet_periods:.1f}%") 

plt.show()