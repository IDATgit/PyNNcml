import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# --- Path Correction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.physics_informed_rain_estimation import PhysicsInformedRainEstimation
from examples.physics_aware_network.trainers.two_stage_model_trainer import prepare_batch
import pynncml as pnc

# --- Configuration ---
METADATA_FEATURES = 2
STYLE_VECTOR_DIM = 128
SAMPLE_TO_ANALYZE = 42 # Same sample as was used for overfitting
MODEL_PATH = os.path.join(script_dir, 'results', 'checkpoints_overfit', 'overfit_final_model.pth')
SAVE_PLOT_PATH = os.path.join(script_dir, 'results', 'overfit_analysis_plot.png')

def analyze_overfit():
    """
    Loads the overfitted model, runs it on the single sample, and plots the result.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the single data sample
    print(f"Loading sample {SAMPLE_TO_ANALYZE}...")
    full_train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
    rain, rsl, tsl, metadata = full_train_dataset[SAMPLE_TO_ANALYZE]

    # Convert to tensors and create a batch of 1
    batch = (
        torch.from_numpy(rain).unsqueeze(0),
        torch.from_numpy(rsl).unsqueeze(0),
        torch.from_numpy(tsl).unsqueeze(0),
        torch.from_numpy(metadata).unsqueeze(0)
    )

    # 2. Initialize and load the model
    print(f"Loading overfitted model from {MODEL_PATH}")
    model = PhysicsInformedRainEstimation(
        metadata_features_baseline=METADATA_FEATURES,
        style_vector_dim_baseline=STYLE_VECTOR_DIM,
        metadata_features_rain_rate=METADATA_FEATURES,
        style_vector_dim_rain_rate=STYLE_VECTOR_DIM
    ).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please run the overfit_single_sample_trainer.py script first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Prepare data and run inference
    with torch.no_grad():
        attenuation, metadata_prepared, rain_target = prepare_batch(batch, device)
        predicted_rain, compensated_att = model(attenuation, metadata_prepared)

    # Convert to numpy for plotting
    attenuation_np = attenuation.cpu().numpy()
    compensated_att_np = compensated_att.cpu().numpy()
    rain_target_np = rain_target.cpu().numpy()
    predicted_rain_np = predicted_rain.cpu().numpy()

    # 4. Create and save the plot
    print("Generating plot...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Attenuation
    axs[0].plot(attenuation_np[0, 0, :], label='Original Attenuation', color='blue')
    axs[0].plot(compensated_att_np[0, 0, :], label='Compensated Attenuation', linestyle='--', color='cyan')
    axs[0].set_title('Overfit Result: Attenuation Signals')
    axs[0].set_ylabel('Attenuation (dB)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Rain Rate
    axs[1].plot(rain_target_np[0, 0, :], label='Target Rain Rate', linewidth=2, color='green')
    axs[1].plot(predicted_rain_np[0, 0, :], label='Predicted Rain Rate', linestyle='--', color='orange')
    axs[1].set_title('Overfit Result: Rain Rate Estimation')
    axs[1].set_ylabel('Rain Rate (mm/hr)')
    axs[1].set_xlabel('Time Step (15 min intervals)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PATH)
    print(f"Plot saved successfully to {SAVE_PLOT_PATH}")

if __name__ == '__main__':
    analyze_overfit() 