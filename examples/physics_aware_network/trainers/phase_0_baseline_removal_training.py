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
from examples.physics_aware_network.models.simple_baseline_removal import day_median_baseline_removal_fixed
import pynncml as pnc

# Global parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Use the integrated physics-informed model with our new simple baseline removal
full_model = PhysicsInformedRainEstimation(metadata_features=3, window_hours=24).to(device)

# Setup optimizer - only train the baseline remover for now
optimizer = torch.optim.Adam(full_model.baseline_remover.parameters(), lr=LEARNING_RATE)

# Setup datasets and dataloaders
print("Loading datasets...")
train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-01", "2015-07-31"))
val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-08-01", "2015-08-31"))
# train_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-02", "2015-06-03"))
# val_dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=slice("2015-06-04", "2015-06-05"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Datasets loaded.")

print(f"Full model parameters: {sum(p.numel() for p in full_model.parameters()):,}")
print(f"Baseline remover parameters: {sum(p.numel() for p in full_model.baseline_remover.parameters()):,}")
print(f"Rain estimator parameters: {sum(p.numel() for p in full_model.rain_rate_estimator.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in full_model.baseline_remover.parameters() if p.requires_grad):,}")

# Training loop
print(f"Starting training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    # Training phase
    full_model.baseline_remover.train()
    total_train_loss = 0
    num_train_batches = 0
    
    for batch in train_loader:
        rain, rsl, tsl, metadata = batch
        rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
        attenuation = tsl - rsl  # (batch_size, n_samples_low_res, time_samples_within_low_res_time)
        
        # reformat dimensions
        attenuation = attenuation.view(attenuation.shape[0], -1)  # (batch_size, time_samples)
        attenuation = attenuation.unsqueeze(1)  # (batch_size, 1, time_samples)
        
        # Forward pass
        output = full_model.baseline_remover.forward(attenuation, metadata)
        target = day_median_baseline_removal_fixed(attenuation)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_train_batches += 1
    
    avg_train_loss = total_train_loss / num_train_batches
    
    # Validation phase
    full_model.baseline_remover.eval()
    total_val_loss = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            rain, rsl, tsl, metadata = batch
            rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
            attenuation = tsl - rsl
            
            # reformat dimensions
            attenuation = attenuation.view(attenuation.shape[0], -1)
            attenuation = attenuation.unsqueeze(1)
            
            # Forward pass
            output = full_model.baseline_remover.forward(attenuation, metadata)
            target = day_median_baseline_removal_fixed(attenuation)
            loss = torch.nn.functional.mse_loss(output, target)
            
            total_val_loss += loss.item()
            num_val_batches += 1
    
    avg_val_loss = total_val_loss / num_val_batches
    
    # Print progress
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

# Save the final model
print("Saving final model...")

# Create directory structure for trained models
save_dir = os.path.join(os.path.dirname(__file__), "trained_models", "simple_baseline_removal")
os.makedirs(save_dir, exist_ok=True)

# Save model checkpoint
model_path = os.path.join(save_dir, "physics_informed_rain_model.pth")
torch.save(full_model.state_dict(), model_path)

# Also save just the baseline remover for easier loading
baseline_path = os.path.join(save_dir, "simple_baseline_removal_only.pth")
torch.save(full_model.baseline_remover.state_dict(), baseline_path)

# Save training info
info_path = os.path.join(save_dir, "training_info.txt")
with open(info_path, 'w') as f:
    f.write(f"Model: SimpleBaselineRemoval integrated in PhysicsInformedRainEstimation\n")
    f.write(f"Training epochs: {NUM_EPOCHS}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Window hours: 24\n")
    f.write(f"Metadata features: 3\n")
    f.write(f"Training data: 2015-06-01 to 2015-07-31\n")
    f.write(f"Validation data: 2015-08-01 to 2015-08-31\n")
    f.write(f"Total model parameters: {sum(p.numel() for p in full_model.parameters()):,}\n")
    f.write(f"Baseline remover parameters: {sum(p.numel() for p in full_model.baseline_remover.parameters()):,}\n")
    f.write(f"Rain estimator parameters: {sum(p.numel() for p in full_model.rain_rate_estimator.parameters()):,}\n")

print(f"Model saved to: {model_path}")
print(f"Baseline remover saved to: {baseline_path}")
print(f"Training info saved to: {info_path}")
print("Training completed!")

# Quick test to see output shapes and sample values
print("\n=== Model Test ===")
with torch.no_grad():
    full_model.eval()
    sample_batch = next(iter(train_loader))
    rain, rsl, tsl, metadata = sample_batch
    rain, rsl, tsl, metadata = rain.to(device), rsl.to(device), tsl.to(device), metadata.to(device)
    attenuation = tsl - rsl
    attenuation = attenuation.view(attenuation.shape[0], -1).unsqueeze(1)
    
    # Test baseline removal
    baseline_output = full_model.baseline_remover(attenuation, metadata)
    target_output = day_median_baseline_removal_fixed(attenuation)
    
    # Test full pipeline
    rain_related_attenuation, estimated_rain_rate = full_model(attenuation, metadata)
    
    print(f"Input attenuation shape: {attenuation.shape}")
    print(f"Baseline output shape: {baseline_output.shape}")
    print(f"Target baseline shape: {target_output.shape}")
    print(f"Rain-related attenuation shape: {rain_related_attenuation.shape}")
    print(f"Estimated rain rate shape: {estimated_rain_rate.shape}")
    print(f"Input range: [{attenuation.min():.3f}, {attenuation.max():.3f}]")
    print(f"Baseline output range: [{baseline_output.min():.3f}, {baseline_output.max():.3f}]")
    print(f"Target baseline range: [{target_output.min():.3f}, {target_output.max():.3f}]")
    print(f"Rain-related range: [{rain_related_attenuation.min():.3f}, {rain_related_attenuation.max():.3f}]")
    print(f"Rain rate range: [{estimated_rain_rate.min():.3f}, {estimated_rain_rate.max():.3f}]")
    print(f"Baseline MSE: {torch.nn.functional.mse_loss(baseline_output, target_output):.6f}")






















