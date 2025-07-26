import os
import sys
import torch
from torch.utils.data import DataLoader

# --- Path Correction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from examples.physics_aware_network.models.physics_informed_rain_estimation import PhysicsInformedRainEstimation
from examples.physics_aware_network.models.simple_baseline_removal import SimpleBaselineRemoval
import pynncml as pnc

# Global parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS_PER_ITERATION = 20
NUM_ITERATIONS = 5
L2_PENALTY_WEIGHT = 0.1

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
    
    Class imbalance handling:
    - Rain periods (rain > 0): weight = 1.0 (full weight)
    - Non-rain periods (rain = 0): weight = 0.1 (reduced weight to compensate for ~10x frequency)
    """
    # Create class-balanced weights for rain MSE
    # Non-rain periods are ~10x more common, so we reduce their weight to 0.1
    rain_weights = torch.where(actual_rain_15min > 0, 1.0, 0.1)  # (batch, time_15min, 1)
    
    # Weighted MSE loss (15-min resolution)
    squared_errors = torch.square(predicted_rain_15min - actual_rain_15min)
    weighted_squared_errors = squared_errors * rain_weights
    rain_mse = torch.mean(weighted_squared_errors)
    
    # Also compute unweighted MSE for monitoring
    unweighted_rain_mse = torch.nn.functional.mse_loss(predicted_rain_15min, actual_rain_15min)
    
    # L2 penalty on compensated attenuation when rain = 0 (10-sec resolution)
    # Find where rain = 0 (expand to 10-sec resolution for masking)
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
    
    # Return both weighted and unweighted MSE for monitoring
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

# Create new physics-informed model
full_model = PhysicsInformedRainEstimation(metadata_features=3, window_hours=24).to(device)

# Load pre-trained baseline remover from phase_0
print("Loading pre-trained baseline remover...")
baseline_checkpoint_path = os.path.join(os.path.dirname(__file__), "trained_models", "simple_baseline_removal", "simple_baseline_removal_only.pth")

if os.path.exists(baseline_checkpoint_path):
    baseline_state_dict = torch.load(baseline_checkpoint_path, map_location=device)
    full_model.baseline_remover.load_state_dict(baseline_state_dict)
    print(f"Loaded baseline remover from: {baseline_checkpoint_path}")
else:
    print(f"Warning: Could not find baseline checkpoint at {baseline_checkpoint_path}")
    print("Starting with randomly initialized baseline remover...")

# Setup datasets and dataloaders
print("Loading datasets...")
train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-08-01", "2015-08-31"))
# # Use a single day for both train and val for quick debugging
# train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-02", "2015-06-02"))
# val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-03", "2015-06-03"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Datasets loaded.")

print(f"Full model parameters: {sum(p.numel() for p in full_model.parameters()):,}")
print(f"Baseline remover parameters: {sum(p.numel() for p in full_model.baseline_remover.parameters()):,}")
print(f"Rain estimator parameters: {sum(p.numel() for p in full_model.rain_rate_estimator.parameters()):,}")

# Training variables
best_val_loss = float('inf')
save_dir = os.path.join(os.path.dirname(__file__), "trained_models", "iterative_training")
os.makedirs(save_dir, exist_ok=True)

# Iterative Training Loop
print(f"Starting iterative training: {NUM_ITERATIONS} iterations, {EPOCHS_PER_ITERATION} epochs each")
print("=" * 80)

for iteration in range(NUM_ITERATIONS):
    # Determine which component to train
    train_rain_rate = (iteration % 2 == 0)  # Odd iterations (0,2,4): train rain rate
    
    if train_rain_rate:
        print(f"\n🌧️  ITERATION {iteration + 1}: Training Rain Rate Estimator (Baseline Frozen)")
        # Freeze baseline remover, train rain rate estimator
        for param in full_model.baseline_remover.parameters():
            param.requires_grad = False
        for param in full_model.rain_rate_estimator.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(full_model.rain_rate_estimator.parameters(), lr=LEARNING_RATE)
    else:
        print(f"\n📊 ITERATION {iteration + 1}: Training Baseline Remover (Rain Rate Frozen)")
        # Freeze rain rate estimator, train baseline remover
        for param in full_model.rain_rate_estimator.parameters():
            param.requires_grad = False
        for param in full_model.baseline_remover.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(full_model.baseline_remover.parameters(), lr=LEARNING_RATE)
    
    # Training loop for this iteration
    for epoch in range(EPOCHS_PER_ITERATION):
        # Training phase
        full_model.train()
        total_train_loss = 0
        total_train_rain_mse = 0
        total_train_unweighted_mse = 0
        total_train_l2_penalty = 0
        num_train_batches = 0
        
        for batch in train_loader:
            rain, rsl, tsl, metadata = batch
            rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
            attenuation = tsl - rsl  # (batch_size, n_samples_low_res, time_samples_within_low_res_time)
            
            # Reformat dimensions
            attenuation = attenuation.view(attenuation.shape[0], -1)  # (batch_size, time_samples)
            attenuation = attenuation.unsqueeze(1)  # (batch_size, 1, time_samples)
            
            # Truncate rain to match complete 15-min periods
            rain_truncated = truncate_rain_to_match_attenuation(rain, attenuation.shape[-1])
            
            # Forward pass through full model
            rain_related_attenuation, predicted_rain_rate = full_model(attenuation, metadata)
            
            # Convert predicted rain rate to 15-min resolution
            predicted_rain_15min = average_to_15min_resolution(predicted_rain_rate)
            
            # Compute combined loss
            loss, rain_mse, l2_penalty, unweighted_rain_mse = compute_loss(
                predicted_rain_15min, 
                rain_truncated,
                rain_related_attenuation
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_rain_mse += rain_mse.item()
            total_train_unweighted_mse += unweighted_rain_mse.item()
            total_train_l2_penalty += l2_penalty.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_rain_mse = total_train_rain_mse / num_train_batches
        avg_train_unweighted_mse = total_train_unweighted_mse / num_train_batches
        avg_train_l2_penalty = total_train_l2_penalty / num_train_batches
        
        # Validation phase
        full_model.eval()
        total_val_loss = 0
        total_val_rain_mse = 0
        total_val_unweighted_mse = 0
        total_val_l2_penalty = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rain, rsl, tsl, metadata = batch
                rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
                attenuation = tsl - rsl
                
                # Reformat dimensions
                attenuation = attenuation.view(attenuation.shape[0], -1)
                attenuation = attenuation.unsqueeze(1)
                
                # Truncate rain to match complete 15-min periods
                rain_truncated = truncate_rain_to_match_attenuation(rain, attenuation.shape[-1])
                
                # Forward pass
                rain_related_attenuation, predicted_rain_rate = full_model(attenuation, metadata)
                
                # Convert to 15-min resolution
                predicted_rain_15min = average_to_15min_resolution(predicted_rain_rate)
                
                # Compute loss
                loss, rain_mse, l2_penalty, unweighted_rain_mse = compute_loss(
                    predicted_rain_15min,
                    rain_truncated,
                    rain_related_attenuation
                )
                
                total_val_loss += loss.item()
                total_val_rain_mse += rain_mse.item()
                total_val_unweighted_mse += unweighted_rain_mse.item()
                total_val_l2_penalty += l2_penalty.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_rain_mse = total_val_rain_mse / num_val_batches
        avg_val_unweighted_mse = total_val_unweighted_mse / num_val_batches
        avg_val_l2_penalty = total_val_l2_penalty / num_val_batches
        
        # Print detailed progress
        mode = "Rain Rate" if train_rain_rate else "Baseline"
        print(f"Iter {iteration+1} Epoch [{epoch+1}/{EPOCHS_PER_ITERATION}] ({mode})")
        print(f"  Train: Loss={avg_train_loss:.6f} | Weighted_MSE={avg_train_rain_mse:.6f} | Unweighted_MSE={avg_train_unweighted_mse:.6f} | L2_Penalty={avg_train_l2_penalty:.6f}")
        print(f"  Val:   Loss={avg_val_loss:.6f} | Weighted_MSE={avg_val_rain_mse:.6f} | Unweighted_MSE={avg_val_unweighted_mse:.6f} | L2_Penalty={avg_val_l2_penalty:.6f}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, f"best_model_iter_{iteration+1}_epoch_{epoch+1}.pth")
            torch.save(full_model.state_dict(), best_model_path)
            print(f"  💾 New best model saved! Val Loss: {best_val_loss:.6f}")

print("\n" + "=" * 80)
print("🎉 Iterative training completed!")

# Save final model
final_model_path = os.path.join(save_dir, "final_iterative_model.pth")
torch.save(full_model.state_dict(), final_model_path)

# Save training info
info_path = os.path.join(save_dir, "iterative_training_info.txt")
with open(info_path, 'w') as f:
    f.write(f"Iterative Training of PhysicsInformedRainEstimation\n")
    f.write(f"Iterations: {NUM_ITERATIONS}\n")
    f.write(f"Epochs per iteration: {EPOCHS_PER_ITERATION}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"L2 penalty weight: {L2_PENALTY_WEIGHT}\n")
    f.write(f"Training data: 2015-06-01 to 2015-07-31\n")
    f.write(f"Validation data: 2015-08-01 to 2015-08-31\n")
    f.write(f"Best validation loss: {best_val_loss:.6f}\n")
    f.write(f"Final model path: {final_model_path}\n")

print(f"Final model saved to: {final_model_path}")
print(f"Training info saved to: {info_path}")
print(f"Best validation loss achieved: {best_val_loss:.6f}")

# Final model test
print("\n=== Final Model Test ===")
with torch.no_grad():
    full_model.eval()
    sample_batch = next(iter(train_loader))
    rain, rsl, tsl, metadata = sample_batch
    rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
    attenuation = tsl - rsl
    attenuation = attenuation.view(attenuation.shape[0], -1).unsqueeze(1)
    
    # Test full pipeline
    rain_related_attenuation, estimated_rain_rate = full_model(attenuation, metadata)
    predicted_rain_15min = average_to_15min_resolution(estimated_rain_rate)
    rain_truncated = truncate_rain_to_match_attenuation(rain, attenuation.shape[-1])
    
    # Compute final loss
    final_loss, final_rain_mse, final_l2_penalty, final_unweighted_mse = compute_loss(
        predicted_rain_15min, rain_truncated, rain_related_attenuation
    )
    
    print(f"Input attenuation shape: {attenuation.shape}")
    print(f"Rain-related attenuation shape: {rain_related_attenuation.shape}")
    print(f"Predicted rain 15min shape: {predicted_rain_15min.shape}")
    print(f"Actual rain shape: {rain_truncated.shape}")
    print(f"Final test loss: {final_loss:.6f}")
    print(f"Final weighted MSE: {final_rain_mse:.6f}")
    print(f"Final unweighted MSE: {final_unweighted_mse:.6f}")
    print(f"Final L2 penalty: {final_l2_penalty:.6f}")




















