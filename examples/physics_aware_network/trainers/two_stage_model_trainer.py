import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Path Correction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.physics_informed_rain_estimation import PhysicsInformedRainEstimation
import pynncml as pnc

# --- Configuration ---
METADATA_FEATURES = 2 # Frequency and Length
STYLE_VECTOR_DIM = 128 # Placeholder, model calculates internally
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3000
DRY_WET_RATIO = 10.0 # Weight for wet samples
REGULARIZATION_WEIGHT = 0.1
CHECKPOINT_DIR = os.path.join(script_dir, '..', 'results', 'checkpoints')
TENSORBOARD_DIR = os.path.join(script_dir, '..', 'results', 'tensorboard')
PLOT_DIR = os.path.join(script_dir, '..', 'results', 'training_plots')

def custom_loss_function(predicted_rain, target_rain, compensated_attenuation, dry_wet_ratio, regularization_weight):
    """
    Custom loss function.
    - Weighted L2 loss for rain rate (MSE).
    - Regularization term for compensated attenuation during dry periods.
    """
    batch_size, time_samples, n_samples = compensated_attenuation.shape
    # Identify dry/wet samples
    is_wet = target_rain == 0
    
    # Calculate weighted L2 loss for rain rate
    weights = torch.ones_like(target_rain)
    weights[is_wet] = dry_wet_ratio
    rain_loss = nn.functional.mse_loss(predicted_rain, target_rain, reduction='none')
    weighted_rain_loss = torch.mean(weights * rain_loss)
    
    # Calculate regularization loss for dry periods
    dry_attenuation = compensated_attenuation[~is_wet]
    regularization_loss = torch.mean(dry_attenuation**2)
    attenuation_flat = compensated_attenuation.view(batch_size, -1)  # (batch_size, time_samples * n_samples)
    # Reshape to (batch_size, num_24h_chunks, samples_per_24h)
    samples_per_24h = 24 * 60  # 24 hours * 60 minutes
    num_chunks = attenuation_flat.shape[1] // samples_per_24h
    if num_chunks > 0:
        # Reshape to handle complete 24h chunks
        chunks = attenuation_flat[:, :num_chunks*samples_per_24h].view(batch_size, num_chunks, samples_per_24h)
        # Calculate median per 24h chunk
        baseline = torch.median(chunks, dim=2, keepdim=True)[0]  # (batch_size, num_chunks, 1)
        # Repeat the baseline for each minute in the 24h period
        baseline = baseline.repeat(1, 1, samples_per_24h)
        # Reshape back to match original flat shape
        baseline = baseline.view(batch_size, num_chunks*samples_per_24h)
        
        # Handle remaining samples if any
        if attenuation_flat.shape[1] > num_chunks*samples_per_24h:
            remaining = attenuation_flat[:, num_chunks*samples_per_24h:]
            remaining_baseline = torch.median(remaining, dim=1, keepdim=True)[0]
            baseline = torch.cat([baseline, remaining_baseline.repeat(1, remaining.shape[1])], dim=1)
    else:
        # If less than 24h of data, use single median
        baseline = torch.median(attenuation_flat, dim=1, keepdim=True)[0]
        baseline = baseline.view(batch_size, time_samples, n_samples)
    
    # Combine losses
    total_loss = weighted_rain_loss + regularization_weight * regularization_loss
    return total_loss, weighted_rain_loss, regularization_loss

# --- Main Trainer Class ---
class TwoStageModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Models and Optimizer
        self.model = PhysicsInformedRainEstimation(
            metadata_features_baseline=METADATA_FEATURES,
            style_vector_dim_baseline=STYLE_VECTOR_DIM,
            metadata_features_rain_rate=METADATA_FEATURES,
            style_vector_dim_rain_rate=STYLE_VECTOR_DIM
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # 2. DataLoaders
        print("Loading datasets...")
        train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
        val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-08-01", "2015-08-31"))
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("Datasets loaded.")

        # 3. Logging and Checkpoints
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
        os.makedirs(PLOT_DIR, exist_ok=True)
        self.writer = SummaryWriter(TENSORBOARD_DIR)

        # 4. Fixed visualization batch
        self.vis_batch = next(iter(self.train_loader))


    def train(self):
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            total_train_loss, total_rain_loss, total_reg_loss = 0, 0, 0

            for i, batch in enumerate(self.train_loader):
                rain, rsl, tsl, metadata = batch
                rain, rsl, tsl, metadata = rain.to(self.device), rsl.to(self.device), tsl.to(self.device), metadata.to(self.device)
                attenuation = tsl - rsl

                # Forward pass
                predicted_rain, compensated_att = self.model(attenuation, metadata)
                
                # Calculate loss
                loss, rain_loss, reg_loss = custom_loss_function(
                    predicted_rain, rain_target, compensated_att, DRY_WET_RATIO, REGULARIZATION_WEIGHT
                )

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                total_rain_loss += rain_loss.item()
                total_reg_loss += reg_loss.item()
                
            # --- Logging ---
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Train_Rain', total_train_loss / len(self.train_loader), epoch)
            self.writer.add_scalar('Loss/Train_Regularization', total_reg_loss / len(self.train_loader), epoch)

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}")

            # --- Validation ---
            self.validate(epoch)

            # --- Save Checkpoint and Plot ---
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth"))
            
            if (epoch + 1) % 5 == 0:
                self.save_progress_plot(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                attenuation, metadata, rain_target = prepare_batch(batch, self.device)
                predicted_rain, compensated_att = self.model(attenuation, metadata)
                loss, _, _ = custom_loss_function(
                    predicted_rain, rain_target, compensated_att, DRY_WET_RATIO, REGULARIZATION_WEIGHT
                )
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

    def save_progress_plot(self, epoch):
        """Saves a plot of model performance on a fixed batch."""
        self.model.eval()
        with torch.no_grad():
            attenuation, metadata, rain_target = prepare_batch(self.vis_batch, self.device)
            predicted_rain, compensated_att = self.model(attenuation, metadata)

        # Detach and move to CPU for plotting
        original_att = attenuation.cpu().numpy()
        compensated_att = compensated_att.cpu().numpy()
        target_rain = rain_target.cpu().numpy()
        predicted_rain = predicted_rain.cpu().numpy()
        
        # Create plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Attenuation (first sample in batch)
        axs[0].plot(original_att[0, 0, :], label='Original Attenuation')
        axs[0].plot(compensated_att[0, 0, :], label='Compensated Attenuation', linestyle='--')
        axs[0].set_title(f'Epoch {epoch+1}: Attenuation Signals')
        axs[0].set_ylabel('Attenuation (dB)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Rain Rate (first sample in batch)
        axs[1].plot(target_rain[0, 0, :], label='Target Rain Rate')
        axs[1].plot(predicted_rain[0, 0, :], label='Predicted Rain Rate', linestyle='--')
        axs[1].set_title('Rain Rate Estimation')
        axs[1].set_ylabel('Rain Rate (mm/hr)')
        axs[1].set_xlabel('Time Step (15 min intervals)')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, f"progress_epoch_{epoch+1}.png")
        fig.savefig(save_path)
        plt.close(fig) # Close figure to free memory
        
        self.model.train() # Set model back to training mode

if __name__ == '__main__':
    trainer = TwoStageModelTrainer()
    trainer.train()
