import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import random

# --- Path Correction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.physics_informed_rain_estimation import PhysicsInformedRainEstimation
from examples.physics_aware_network.trainers.two_stage_model_trainer import prepare_batch, custom_loss_function
import pynncml as pnc

# --- Configuration ---
METADATA_FEATURES = 2
STYLE_VECTOR_DIM = 128
# Using the final model from the real training run
MODEL_PATH = os.path.join(script_dir, 'results', 'checkpoints', 'model_epoch_100.pth')
SAVE_DIR = os.path.join(script_dir, 'results', 'final_analysis')

def get_worst_sample(model, dataset, device):
    """Finds the sample with the highest loss in a dataset."""
    max_loss = -1
    worst_sample_idx = -1
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get sample and create a batch of 1
            rain, rsl, tsl, metadata = dataset[i]
            batch = (
                torch.from_numpy(rain).unsqueeze(0),
                torch.from_numpy(rsl).unsqueeze(0),
                torch.from_numpy(tsl).unsqueeze(0),
                torch.from_numpy(metadata).unsqueeze(0)
            )
            
            # Prepare data and run inference
            attenuation, metadata_prepared, rain_target = prepare_batch(batch, device)
            predicted_rain, compensated_att = model(attenuation, metadata_prepared)
            
            # Calculate loss (using dummy values for loss components we don't need here)
            loss, _, _ = custom_loss_function(predicted_rain, rain_target, compensated_att, 1.0, 0.0)
            
            if loss.item() > max_loss:
                max_loss = loss.item()
                worst_sample_idx = i
                
    print(f"Worst sample found at index {worst_sample_idx} with loss {max_loss:.4f}")
    return worst_sample_idx

def plot_sample(model, dataset, sample_idx, plot_title, save_path, device):
    """Generates and saves a plot for a given sample."""
    model.eval()
    
    # Get sample and create a batch of 1
    rain, rsl, tsl, metadata = dataset[sample_idx]
    batch = (
        torch.from_numpy(rain).unsqueeze(0),
        torch.from_numpy(rsl).unsqueeze(0),
        torch.from_numpy(tsl).unsqueeze(0),
        torch.from_numpy(metadata).unsqueeze(0)
    )

    with torch.no_grad():
        attenuation, metadata_prepared, rain_target = prepare_batch(batch, device)
        predicted_rain, compensated_att = model(attenuation, metadata_prepared)

    # Convert to numpy for plotting
    attenuation_np = attenuation.cpu().numpy()
    compensated_att_np = compensated_att.cpu().numpy()
    rain_target_np = rain_target.cpu().numpy()
    predicted_rain_np = predicted_rain.cpu().numpy()

    # Create and save the plot
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(plot_title, fontsize=16)
    
    # Plot 1: Attenuation
    axs[0].plot(attenuation_np[0, 0, :], label='Original Attenuation', color='blue')
    axs[0].plot(compensated_att_np[0, 0, :], label='Compensated Attenuation', linestyle='--', color='cyan')
    axs[0].set_title('Attenuation Signals')
    axs[0].set_ylabel('Attenuation (dB)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Rain Rate
    axs[1].plot(rain_target_np[0, 0, :], label='Target Rain Rate', linewidth=2, color='green')
    axs[1].plot(predicted_rain_np[0, 0, :], label='Predicted Rain Rate', linestyle='--', color='orange')
    axs[1].set_title('Rain Rate Estimation')
    axs[1].set_ylabel('Rain Rate (mm/hr)')
    axs[1].set_xlabel('Time Step (15 min intervals)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")
    plt.close(fig)

def main_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Datasets
    print("Loading datasets...")
    train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
    val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-08-01", "2015-08-31"))

    # 2. Load Model
    print(f"Loading trained model from {MODEL_PATH}")
    model = PhysicsInformedRainEstimation(
        metadata_features_baseline=METADATA_FEATURES,
        style_vector_dim_baseline=STYLE_VECTOR_DIM,
        metadata_features_rain_rate=METADATA_FEATURES,
        style_vector_dim_rain_rate=STYLE_VECTOR_DIM
    ).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 3. Perform Analysis
    # Random train sample
    random_train_idx = random.randint(0, len(train_dataset) - 1)
    plot_sample(model, train_dataset, random_train_idx, 'Random Training Sample', os.path.join(SAVE_DIR, 'random_train_sample.png'), device)

    # Random test sample
    random_test_idx = random.randint(0, len(val_dataset) - 1)
    plot_sample(model, val_dataset, random_test_idx, 'Random Test Sample', os.path.join(SAVE_DIR, 'random_test_sample.png'), device)

    # Worst train sample
    worst_train_idx = get_worst_sample(model, train_dataset, device)
    plot_sample(model, train_dataset, worst_train_idx, 'Worst Training Sample', os.path.join(SAVE_DIR, 'worst_train_sample.png'), device)

    # Worst test sample
    worst_test_idx = get_worst_sample(model, val_dataset, device)
    plot_sample(model, val_dataset, worst_test_idx, 'Worst Test Sample', os.path.join(SAVE_DIR, 'worst_test_sample.png'), device)

if __name__ == '__main__':
    main_analysis()
