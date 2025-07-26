import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader

# --- Path Correction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.physics_informed_rain_estimation import PhysicsInformedRainEstimation
import pynncml as pnc

# --- Configuration ---
METADATA_FEATURES = 3  # Updated to match phase_1_iterate_training.py
WINDOW_HOURS = 24      # Added to match phase_1_iterate_training.py
L2_PENALTY_WEIGHT = 0.1  # Match training script

# Using the final model from the iterative training
MODEL_PATH = os.path.join(
    script_dir, 'trainers', 'trained_models', 'iterative_training',
    'final_iterative_model.pth'  # Use final model instead of specific epoch
)
SAVE_DIR = os.path.join(script_dir, 'results', 'final_analysis')

def average_to_15min_resolution(tensor_10sec):
    """
    Average every 90 consecutive samples (10sec → 15min resolution)
    Input: (batch_size, 1, time_samples_10sec)
    Output: (batch_size, time_samples_15min, 1) - matches rain format
    """
    batch_size, channels, time_samples = tensor_10sec.shape
    
    # Calculate how many complete 15-min periods we have
    samples_per_15min = 90
    num_15min_periods = time_samples // samples_per_15min
    
    if num_15min_periods == 0:
        # Handle case where sequence is shorter than 15 minutes
        return tensor_10sec.mean(dim=-1, keepdim=True).permute(0, 2, 1)
    
    # Reshape to group every 90 samples and average
    # Take only complete 15-min periods
    complete_samples = num_15min_periods * samples_per_15min
    truncated = tensor_10sec[:, :, :complete_samples]
    
    # Reshape: (batch, 1, num_periods*90) → (batch, 1, num_periods, 90)
    reshaped = truncated.view(batch_size, channels, num_15min_periods, samples_per_15min)
    
    # Average over the 90 samples: (batch, 1, num_periods, 90) → (batch, 1, num_periods)
    averaged = reshaped.mean(dim=-1)
    
    # Permute to match rain format: (batch, 1, num_periods) → (batch, num_periods, 1)
    return averaged.permute(0, 2, 1)

def compute_loss(predicted_rain_15min, actual_rain_15min, compensated_attenuation_10sec):
    """
    Compute combined loss: Weighted MSE on rain + L2 penalty on compensated attenuation when rain=0
    (Copied from phase_1_iterate_training.py for compatibility)
    """
    # Create class-balanced weights for rain MSE
    rain_weights = torch.where(actual_rain_15min > 0, 1.0, 0.1)
    
    # Weighted MSE loss (15-min resolution)
    squared_errors = torch.square(predicted_rain_15min - actual_rain_15min)
    weighted_squared_errors = squared_errors * rain_weights
    rain_mse = torch.mean(weighted_squared_errors)
    
    # Also compute unweighted MSE for monitoring
    unweighted_rain_mse = torch.nn.functional.mse_loss(predicted_rain_15min, actual_rain_15min)
    
    # L2 penalty on compensated attenuation when rain = 0 (10-sec resolution)
    batch_size, time_15min, _ = actual_rain_15min.shape
    _, _, time_10sec = compensated_attenuation_10sec.shape
    
    # Create mask for zero rain periods
    zero_rain_mask_15min = (actual_rain_15min == 0)  # (batch, time_15min, 1)
    
    # Expand mask to 10-sec resolution
    samples_per_15min = 90
    num_complete_periods = time_10sec // samples_per_15min
    
    if num_complete_periods > 0:
        # Take only the periods we have rain data for
        compensated_truncated = compensated_attenuation_10sec[:, :, :num_complete_periods * samples_per_15min]
        zero_rain_mask_truncated = zero_rain_mask_15min[:, :num_complete_periods, :]
        
        # Expand 15-min mask to 10-sec: (batch, periods, 1) → (batch, 1, periods*90)
        zero_rain_mask_10sec = zero_rain_mask_truncated.permute(0, 2, 1)  # (batch, 1, periods)
        zero_rain_mask_10sec = zero_rain_mask_10sec.repeat_interleave(samples_per_15min, dim=-1)  # (batch, 1, periods*90)
        
        # Apply L2 penalty only where rain = 0
        l2_penalty = torch.mean(torch.square(compensated_truncated) * zero_rain_mask_10sec.float())
    else:
        l2_penalty = torch.tensor(0.0, device=compensated_attenuation_10sec.device)
    
    total_loss = rain_mse + L2_PENALTY_WEIGHT * l2_penalty
    
    return total_loss, rain_mse, l2_penalty, unweighted_rain_mse

def truncate_rain_to_match_attenuation(rain, attenuation_time_samples):
    """
    Truncate rain data to match the number of complete 15-min periods in attenuation
    """
    samples_per_15min = 90
    num_complete_periods = attenuation_time_samples // samples_per_15min
    
    if num_complete_periods > 0:
        return rain[:, :num_complete_periods, :]
    else:
        # If attenuation is shorter than 15 min, take first rain sample
        return rain[:, :1, :]

def prepare_data_batch(batch, device):
    """
    Prepare data batch in the same format as phase_1_iterate_training.py
    """
    rain, rsl, tsl, metadata = batch
    rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
    
    # Calculate attenuation
    attenuation = tsl - rsl  # (batch_size, n_samples_low_res, time_samples_within_low_res_time)
    
    # Reformat dimensions to match training format
    attenuation = attenuation.view(attenuation.shape[0], -1)  # (batch_size, time_samples)
    attenuation = attenuation.unsqueeze(1)  # (batch_size, 1, time_samples)
    
    # Truncate rain to match complete 15-min periods
    rain_truncated = truncate_rain_to_match_attenuation(rain, attenuation.shape[-1])
    
    return attenuation, metadata, rain_truncated

def get_worst_sample(model, dataset, device):
    """Finds the sample with the highest loss in a dataset."""
    max_loss = -1
    worst_sample_idx = -1
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get sample and create a batch of 1
            sample = dataset[i]
            batch = tuple(torch.from_numpy(x).unsqueeze(0) for x in sample)
            
            # Prepare data using same format as training
            attenuation, metadata, rain_target = prepare_data_batch(batch, device)
            
            # Forward pass through model
            rain_related_attenuation, predicted_rain_rate = model(attenuation, metadata)
            
            # Convert to 15-min resolution
            predicted_rain_15min = average_to_15min_resolution(predicted_rain_rate)
            
            # Calculate loss
            loss, _, _, _ = compute_loss(predicted_rain_15min, rain_target, rain_related_attenuation)
            
            if loss.item() > max_loss:
                max_loss = loss.item()
                worst_sample_idx = i
                
    print(f"Worst sample found at index {worst_sample_idx} with loss {max_loss:.4f}")
    return worst_sample_idx

def plot_sample(model, dataset, sample_idx, plot_title, save_path, device):
    """Generates and saves a plot for a given sample."""
    model.eval()
    
    # Get sample and create a batch of 1
    sample = dataset[sample_idx]
    batch = tuple(torch.from_numpy(x).unsqueeze(0) for x in sample)

    with torch.no_grad():
        # Prepare data using same format as training
        attenuation, metadata, rain_target = prepare_data_batch(batch, device)
        
        # Forward pass through model
        rain_related_attenuation, predicted_rain_rate = model(attenuation, metadata)
        
        # Convert to 15-min resolution
        predicted_rain_15min = average_to_15min_resolution(predicted_rain_rate)
        
        # Calculate baseline (original - compensated)
        baseline_attenuation = attenuation - rain_related_attenuation
        
        # Average original attenuation to 15-min resolution to match rain plot
        attenuation_15min = average_to_15min_resolution(attenuation)

    # Convert to numpy for plotting
    attenuation_np = attenuation.cpu().numpy()
    rain_related_attenuation_np = rain_related_attenuation.cpu().numpy()
    baseline_attenuation_np = baseline_attenuation.cpu().numpy()
    attenuation_15min_np = attenuation_15min.cpu().numpy()
    rain_target_np = rain_target.cpu().numpy()
    predicted_rain_15min_np = predicted_rain_15min.cpu().numpy()

    # Create and save the plot with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 16))
    fig.suptitle(plot_title, fontsize=16)
    
    # Plot 1: Attenuation signals (10-sec resolution)
    axs[0].plot(attenuation_np[0, 0, :], label='Original Attenuation', color='blue', alpha=0.7)
    axs[0].plot(rain_related_attenuation_np[0, 0, :], label='Rain-Related Attenuation (Compensated)', linestyle='--', color='cyan', linewidth=2)
    axs[0].plot(baseline_attenuation_np[0, 0, :], label='Baseline Attenuation', linestyle=':', color='red', linewidth=2)
    axs[0].set_title('Attenuation Signals (10-second resolution)')
    axs[0].set_ylabel('Attenuation (dB)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Averaged attenuation (15-min resolution) - matches rain resolution
    axs[1].plot(attenuation_15min_np[0, :, 0], label='Original Attenuation (15-min avg)', color='blue', linewidth=2)
    axs[1].plot(predicted_rain_15min_np[0, :, 0], label='Predicted Rain Rate', linestyle='--', color='orange', linewidth=2)
    axs[1].set_title('Averaged Attenuation vs Rain Rate (15-minute resolution)')
    axs[1].set_ylabel('Attenuation (dB) / Rain Rate (mm/hr)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: Rain Rate comparison
    axs[2].plot(rain_target_np[0, :, 0], label='Target Rain Rate', linewidth=2, color='green')
    axs[2].plot(predicted_rain_15min_np[0, :, 0], label='Predicted Rain Rate', linestyle='--', color='orange', linewidth=2)
    axs[2].set_title('Rain Rate Estimation')
    axs[2].set_ylabel('Rain Rate (mm/hr)')
    axs[2].set_xlabel('Time Step (15 min intervals)')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")
    plt.show()  # Display plot interactively
    plt.close(fig)

def create_performance_tables(model, dataset, device, dataset_name="Dataset"):
    """
    Create performance tables with RMSE and bias for different rain rate bins.
    Adapted from analyze.py to work with the physics-informed model.
    """
    print(f"\nGenerating performance tables for {dataset_name}...")
    
    # Define rain rate bins (same as analyze.py)
    rain_rates_bins_low = np.array([0, 0.1] + [0.1 + 3.5 * i for i in range(1, 9)])
    rain_rates_bins_high = np.array([0.1] + [0.1 + 3.5 * i for i in range(1, 10)])
    
    # Initialize accumulators
    rmse_by_rain_rate = np.zeros(len(rain_rates_bins_low))
    bias_by_rain_rate = np.zeros(len(rain_rates_bins_low))
    rmse_by_rain_rate_nof_samples = np.zeros(len(rain_rates_bins_low))
    
    model.eval()
    
    # Create DataLoader for batched processing
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in data_loader:
            # Prepare data using same format as training
            attenuation, metadata, rain_target = prepare_data_batch(batch, device)
            
            # Forward pass through model
            rain_related_attenuation, predicted_rain_rate = model(attenuation, metadata)
            
            # Convert to 15-min resolution
            predicted_rain_15min = average_to_15min_resolution(predicted_rain_rate)
            
            # Flatten for bin analysis (convert to CPU and flatten)
            targets = rain_target.cpu().flatten()  # (batch * time_steps,)
            preds = predicted_rain_15min.cpu().flatten()  # (batch * time_steps,)
            
            # Process each rain rate bin
            for i, (rain_rate_low, rain_rate_high) in enumerate(zip(rain_rates_bins_low, rain_rates_bins_high)):
                mask = (targets >= rain_rate_low) & (targets < rain_rate_high)
                if mask.sum() > 0:
                    # Compute squared error sum and accumulate
                    rmse_by_rain_rate[i] += ((preds[mask] - targets[mask])**2).sum().item()
                    # Compute bias (mean error)
                    bias_by_rain_rate[i] += (preds[mask] - targets[mask]).sum().item()
                    rmse_by_rain_rate_nof_samples[i] += mask.sum().item()
    
    # Calculate final metrics (avoid division by zero)
    valid_bins = rmse_by_rain_rate_nof_samples > 0
    rmse_by_rain_rate[valid_bins] /= rmse_by_rain_rate_nof_samples[valid_bins]
    rmse_by_rain_rate[valid_bins] = np.sqrt(rmse_by_rain_rate[valid_bins])
    bias_by_rain_rate[valid_bins] /= rmse_by_rain_rate_nof_samples[valid_bins]
    
    # Set invalid bins to NaN
    rmse_by_rain_rate[~valid_bins] = np.nan
    bias_by_rain_rate[~valid_bins] = np.nan
    
    # Create results dictionary
    rain_rate_rmse_table = {
        "rain_rate_bins_low": rain_rates_bins_low,
        "rain_rate_bins_high": rain_rates_bins_high,
        "rmse_by_rain_rate": rmse_by_rain_rate,
        "bias_by_rain_rate": bias_by_rain_rate,
        "rmse_by_rain_rate_nof_samples": rmse_by_rain_rate_nof_samples.astype(int),
    }
    
    # Create and print DataFrame
    bin_labels = [f"{low:.1f} - {high:.1f}" for low, high in zip(rain_rates_bins_low, rain_rates_bins_high)]
    rain_rate_rmse_df = pd.DataFrame({
        "Bin (mm/hr)": bin_labels,
        "RMSE": rain_rate_rmse_table["rmse_by_rain_rate"],
        "Bias": rain_rate_rmse_table["bias_by_rain_rate"],
        "#Samples": rain_rate_rmse_table["rmse_by_rain_rate_nof_samples"]
    })
    
    print(f"\n{dataset_name} Rain Rate Performance Table:")
    print("=" * 60)
    print(rain_rate_rmse_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"))
    
    return rain_rate_rmse_table

def main_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Datasets
    print("Loading datasets...")
    train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
    val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-08-01", "2015-08-31"))
    # train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-06-02"))
    # val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-03", "2015-06-04"))

    # 2. Load Model (Updated to match phase_1_iterate_training.py constructor)
    model_path = MODEL_PATH  # Use local variable to avoid UnboundLocalError
    print(f"Loading trained model from {model_path}")
    model = PhysicsInformedRainEstimation(
        metadata_features=METADATA_FEATURES,  # Updated: 3 instead of separate baseline/rain_rate
        window_hours=WINDOW_HOURS             # Updated: added window_hours parameter
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        # Try alternative path with specific best checkpoint
        alt_path = os.path.join(script_dir, 'trainers', 'trained_models', 'iterative_training', 'best_model_iter_5_epoch_20.pth')
        if os.path.exists(alt_path):
            print(f"Trying alternative path: {alt_path}")
            model_path = alt_path
        else:
            print("No trained model found. Please run phase_1_iterate_training.py first.")
            return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")

    # 3. Perform Analysis
    print("Analyzing model performance...")
    
    # Random train sample
    random_train_idx = random.randint(0, len(train_dataset) - 1)
    plot_sample(model, train_dataset, random_train_idx, 'Random Training Sample', 
                os.path.join(SAVE_DIR, 'random_train_sample.png'), device)

    # Random test sample
    random_test_idx = random.randint(0, len(val_dataset) - 1)
    plot_sample(model, val_dataset, random_test_idx, 'Random Test Sample', 
                os.path.join(SAVE_DIR, 'random_test_sample.png'), device)

    # Worst test sample (skip train worst to save time)
    print("Finding worst performing sample...")
    worst_test_idx = get_worst_sample(model, val_dataset, device)
    plot_sample(model, val_dataset, worst_test_idx, 'Worst Test Sample', 
                os.path.join(SAVE_DIR, 'worst_test_sample.png'), device)

    # 4. Generate Performance Tables
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE STATISTICS")
    print("="*60)
    
    # Compute performance tables for both datasets
    train_performance = create_performance_tables(model, train_dataset, device, "Training Dataset")
    val_performance = create_performance_tables(model, val_dataset, device, "Validation Dataset")
    
    # Save performance tables to CSV
    def save_performance_table(performance_dict, dataset_name):
        bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in 
                     zip(performance_dict["rain_rate_bins_low"], performance_dict["rain_rate_bins_high"])]
        df = pd.DataFrame({
            "Bin_mm_hr": bin_labels,
            "RMSE": performance_dict["rmse_by_rain_rate"],
            "Bias": performance_dict["bias_by_rain_rate"],
            "Samples": performance_dict["rmse_by_rain_rate_nof_samples"]
        })
        csv_path = os.path.join(SAVE_DIR, f'{dataset_name.lower().replace(" ", "_")}_performance.csv')
        df.to_csv(csv_path, index=False)
        print(f"Performance table saved to: {csv_path}")
    
    save_performance_table(train_performance, "Training Dataset")
    save_performance_table(val_performance, "Validation Dataset")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"Visual results saved to: {SAVE_DIR}")
    print(f"Performance tables saved to CSV files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main_analysis()
