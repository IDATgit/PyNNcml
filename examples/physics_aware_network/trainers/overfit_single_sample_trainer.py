import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
from examples.physics_aware_network.trainers.two_stage_model_trainer import prepare_batch, custom_loss_function

# --- Configuration ---
METADATA_FEATURES = 2
STYLE_VECTOR_DIM = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2000 # Increased epochs for overfitting
DRY_WET_RATIO = 10.0
REGULARIZATION_WEIGHT = 0.1
CHECKPOINT_DIR = os.path.join(script_dir, '..', 'results', 'checkpoints_overfit')
TENSORBOARD_DIR = os.path.join(script_dir, '..', 'results', 'tensorboard_overfit')
SAMPLE_TO_OVERFIT = 42 # Using a specific, potentially interesting sample

# --- Main Trainer Class ---
class OverfitSingleSampleTrainer:
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
        print("Loading single sample for overfitting test...")
        full_train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
        
        # Get a single sample
        rain, rsl, tsl, metadata = full_train_dataset[SAMPLE_TO_OVERFIT]
        
        # Convert numpy arrays to tensors and unsqueeze to create a batch of 1
        rain = torch.from_numpy(rain).unsqueeze(0)
        rsl = torch.from_numpy(rsl).unsqueeze(0)
        tsl = torch.from_numpy(tsl).unsqueeze(0)
        metadata = torch.from_numpy(metadata).unsqueeze(0)

        # Create a simple dataloader that will yield this same batch forever
        single_sample_dataset = TensorDataset(rain, rsl, tsl, metadata)
        self.train_loader = DataLoader(single_sample_dataset, batch_size=1, shuffle=False)
        print(f"Single sample at index {SAMPLE_TO_OVERFIT} loaded.")

        # 3. Logging and Checkpoints
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
        self.writer = SummaryWriter(TENSORBOARD_DIR)

        # 4. Live Plotting Setup (DISABLED FOR STABILITY)
        # plt.ion()
        # self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # self.first_batch_visualized = False

    def train(self):
        print("Starting overfitting process...")
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            
            # The loader only has one item, so this loop runs once per epoch
            for i, batch in enumerate(self.train_loader):
                attenuation, metadata, rain_target = prepare_batch(batch, self.device)
                
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
                
                # --- Logging ---
                self.writer.add_scalar('Loss/Overfit', loss.item(), epoch)
                self.writer.add_scalar('Loss/Overfit_Rain', rain_loss.item(), epoch)
                self.writer.add_scalar('Loss/Overfit_Regularization', reg_loss.item(), epoch)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}")

                # Update live plot every few epochs (DISABLED)
                # if (epoch + 1) % 5 == 0:
                #     self.update_live_plot(
                #         attenuation.cpu().detach().numpy(),
                #         compensated_att.cpu().detach().numpy(),
                #         rain_target.cpu().detach().numpy(),
                #         predicted_rain.cpu().detach().numpy(),
                #         epoch
                #     )
            
        # --- Save Final Model ---
        final_model_path = os.path.join(CHECKPOINT_DIR, "overfit_final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Overfitting complete. Final model saved to {final_model_path}")
        
        # plt.ioff()
        # plt.show()

    def update_live_plot(self, original_att, compensated_att, target_rain, predicted_rain, epoch):
        """Updates the live plot with a single sample from the batch."""
        self.axs[0].clear()
        self.axs[1].clear()

        # Plot 1: Attenuation
        self.axs[0].plot(original_att[0, 0, :], label='Original Attenuation')
        self.axs[0].plot(compensated_att[0, 0, :], label='Compensated Attenuation', linestyle='--')
        self.axs[0].set_title(f'Epoch {epoch+1}: Attenuation Signals (Overfitting)')
        self.axs[0].set_ylabel('Attenuation (dB)')
        self.axs[0].legend()
        self.axs[0].grid(True)
        
        # Plot 2: Rain Rate
        self.axs[1].plot(target_rain[0, 0, :], label='Target Rain Rate', linewidth=2)
        self.axs[1].plot(predicted_rain[0, 0, :], label='Predicted Rain Rate', linestyle='--')
        self.axs[1].set_title('Rain Rate Estimation (Overfitting)')
        self.axs[1].set_ylabel('Rain Rate (mm/hr)')
        self.axs[1].set_xlabel('Time Step (15 min intervals)')
        self.axs[1].legend()
        self.axs[1].grid(True)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    trainer = OverfitSingleSampleTrainer()
    trainer.train() 