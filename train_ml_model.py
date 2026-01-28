import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time

print("="*80)
print("DROOP-CONTROLLED POWER FLOW - MACHINE LEARNING")
print("="*80)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

data_dir = Path('./data_out_droop_publication/case24_ieee_rts/raw')

print(f"Loading from: {data_dir}")

bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')
scenarios_data = pd.read_parquet(data_dir / 'scenarios_agg_load_profile.parquet')

print(f"\n✓ Loaded bus_data: {len(bus_data)} records")
print(f"✓ Loaded scenarios: {len(scenarios_data)} records")

print(f"\nBus data columns: {list(bus_data.columns)}")
print(f"Scenarios columns: {list(scenarios_data.columns)}")

# Check unique scenarios
n_scenarios = bus_data['scenario'].nunique()
n_buses = bus_data['bus'].nunique()
print(f"\nDataset: {n_scenarios} scenarios × {n_buses} buses = {len(bus_data)} samples")

# ============================================================================
# STEP 2: PREPARE FEATURES AND TARGETS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PREPARING FEATURES AND TARGETS")
print("="*80)

# For each scenario, we need:
# INPUT: Load at all buses (Pd, Qd) - shape (n_scenarios, n_buses * 2)
# OUTPUT: Voltage at all buses (Vm, Va) - shape (n_scenarios, n_buses * 2)

# Pivot data to get one row per scenario
print("\nPivoting data to scenario-level format...")

# Get Pd and Qd for each scenario
pd_pivot = bus_data.pivot_table(index='scenario', columns='bus', values='Pd', aggfunc='first')
qd_pivot = bus_data.pivot_table(index='scenario', columns='bus', values='Qd', aggfunc='first')

# Get Vm and Va for each scenario (targets)
vm_pivot = bus_data.pivot_table(index='scenario', columns='bus', values='Vm', aggfunc='first')
va_pivot = bus_data.pivot_table(index='scenario', columns='bus', values='Va', aggfunc='first')

print(f"✓ Pivoted data shape: {pd_pivot.shape}")

# Combine into feature matrix and target matrix
X_pd = pd_pivot.values  # (n_scenarios, n_buses)
X_qd = qd_pivot.values  # (n_scenarios, n_buses)
X = np.concatenate([X_pd, X_qd], axis=1)  # (n_scenarios, n_buses * 2)

y_vm = vm_pivot.values  # (n_scenarios, n_buses)
y_va = va_pivot.values  # (n_scenarios, n_buses)
y = np.concatenate([y_vm, y_va], axis=1)  # (n_scenarios, n_buses * 2)

print(f"\n✓ Input features (X): {X.shape}")
print(f"✓ Output targets (y): {y.shape}")

print(f"\nFeature statistics:")
print(f"  Pd range: [{X_pd.min():.2f}, {X_pd.max():.2f}] MW")
print(f"  Qd range: [{X_qd.min():.2f}, {X_qd.max():.2f}] MVAr")
print(f"\nTarget statistics:")
print(f"  Vm range: [{y_vm.min():.4f}, {y_vm.max():.4f}] pu")
print(f"  Va range: [{y_va.min():.4f}, {y_va.max():.4f}] rad")

# ============================================================================
# STEP 3: TRAIN-VALIDATION-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TRAIN-VALIDATION-TEST SPLIT")
print("="*80)

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
)

print(f"Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: FEATURE SCALING")
print("="*80)

# Standardize features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print("✓ Features scaled (mean=0, std=1)")
print("✓ Targets scaled (mean=0, std=1)")

# Save scalers for later use
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("✓ Scalers saved to disk")

# ============================================================================
# STEP 5: CREATE PYTORCH DATASETS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING PYTORCH DATASETS")
print("="*80)

class PowerFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PowerFlowDataset(X_train_scaled, y_train_scaled)
val_dataset = PowerFlowDataset(X_val_scaled, y_val_scaled)
test_dataset = PowerFlowDataset(X_test_scaled, y_test_scaled)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✓ Created DataLoaders with batch_size={batch_size}")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# STEP 6: DEFINE NEURAL NETWORK
# ============================================================================
print("\n" + "="*80)
print("STEP 6: DEFINING NEURAL NETWORK")
print("="*80)

class DroopPowerFlowNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DroopPowerFlowNN, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 1
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]   # n_buses * 2 (Pd, Qd)
output_size = y_train.shape[1]  # n_buses * 2 (Vm, Va)

model = DroopPowerFlowNN(input_size, output_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✓ Model created:")
print(f"  Input size:  {input_size} (Pd + Qd for {n_buses} buses)")
print(f"  Output size: {output_size} (Vm + Va for {n_buses} buses)")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

print(f"\nModel architecture:")
print(model)

# ============================================================================
# STEP 7: TRAINING SETUP
# ============================================================================
print("\n" + "="*80)
print("STEP 7: TRAINING SETUP")
print("="*80)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print("✓ Loss function: MSE")
print("✓ Optimizer: Adam (lr=0.001)")
print("✓ LR Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")

# ============================================================================
# STEP 8: TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAINING")
print("="*80)

num_epochs = 100
best_val_loss = float('inf')
patience_counter = 0
max_patience = 15

train_losses = []
val_losses = []

print(f"\nStarting training for {num_epochs} epochs...")
print("-" * 80)

start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

total_time = time.time() - start_time
print("-" * 80)
print(f"\n✓ Training complete!")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"  Final epoch: {epoch+1}")

# ============================================================================
# STEP 9: LOAD BEST MODEL AND EVALUATE
# ============================================================================
print("\n" + "="*80)
print("STEP 9: EVALUATION ON TEST SET")
print("="*80)

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")

model.eval()
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

test_loss /= len(test_loader)
print(f"\nTest Loss (MSE): {test_loss:.6f}")

# Concatenate all predictions and targets
predictions_scaled = np.concatenate(all_predictions, axis=0)
targets_scaled = np.concatenate(all_targets, axis=0)

# Inverse transform to original scale
predictions = scaler_y.inverse_transform(predictions_scaled)
targets = scaler_y.inverse_transform(targets_scaled)

# ============================================================================
# STEP 10: DETAILED METRICS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: DETAILED PERFORMANCE METRICS")
print("="*80)

# Split predictions into Vm and Va
n_buses = output_size // 2
pred_vm = predictions[:, :n_buses]
pred_va = predictions[:, n_buses:]
true_vm = targets[:, :n_buses]
true_va = targets[:, n_buses:]

# Calculate metrics for voltage magnitude
mae_vm = np.mean(np.abs(pred_vm - true_vm))
rmse_vm = np.sqrt(np.mean((pred_vm - true_vm)**2))
max_error_vm = np.max(np.abs(pred_vm - true_vm))
mape_vm = np.mean(np.abs((pred_vm - true_vm) / true_vm)) * 100

# Calculate metrics for voltage angle
mae_va = np.mean(np.abs(pred_va - true_va))
rmse_va = np.sqrt(np.mean((pred_va - true_va)**2))
max_error_va = np.max(np.abs(pred_va - true_va))

print("\nVOLTAGE MAGNITUDE (Vm) PREDICTIONS:")
print(f"  MAE:        {mae_vm:.6f} pu ({mae_vm*100:.4f}%)")
print(f"  RMSE:       {rmse_vm:.6f} pu")
print(f"  Max Error:  {max_error_vm:.6f} pu")
print(f"  MAPE:       {mape_vm:.4f}%")

print("\nVOLTAGE ANGLE (Va) PREDICTIONS:")
print(f"  MAE:        {mae_va:.6f} rad ({np.degrees(mae_va):.4f}°)")
print(f"  RMSE:       {rmse_va:.6f} rad ({np.degrees(rmse_va):.4f}°)")
print(f"  Max Error:  {max_error_va:.6f} rad ({np.degrees(max_error_va):.4f}°)")

# R² score
from sklearn.metrics import r2_score
r2_vm = r2_score(true_vm.flatten(), pred_vm.flatten())
r2_va = r2_score(true_va.flatten(), pred_va.flatten())

print(f"\nR² SCORES:")
print(f"  Voltage Magnitude: {r2_vm:.6f}")
print(f"  Voltage Angle:     {r2_va:.6f}")

# ============================================================================
# STEP 11: SAVE RESULTS AND PLOTS
# ============================================================================
print("\n" + "="*80)
print("STEP 11: SAVING RESULTS")
print("="*80)

output_dir = Path('./ml_results')
output_dir.mkdir(exist_ok=True)

# Plot 1: Training history
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_losses, label='Training Loss', linewidth=2)
ax.plot(val_losses, label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('Training History - Droop Power Flow NN', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: training_history.png")

# Plot 2: Voltage magnitude predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(true_vm.flatten(), pred_vm.flatten(), s=1, alpha=0.5, color='steelblue')
ax.plot([true_vm.min(), true_vm.max()], [true_vm.min(), true_vm.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('True Voltage Magnitude (pu)', fontsize=12)
ax.set_ylabel('Predicted Voltage Magnitude (pu)', fontsize=12)
ax.set_title(f'Voltage Magnitude Predictions (MAE: {mae_vm:.6f} pu)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'vm_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: vm_predictions.png")

# Plot 3: Voltage angle predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(true_va.flatten(), pred_va.flatten(), s=1, alpha=0.5, color='darkorange')
ax.plot([true_va.min(), true_va.max()], [true_va.min(), true_va.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('True Voltage Angle (rad)', fontsize=12)
ax.set_ylabel('Predicted Voltage Angle (rad)', fontsize=12)
ax.set_title(f'Voltage Angle Predictions (MAE: {mae_va:.6f} rad)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'va_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: va_predictions.png")

# Plot 4: Error distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

errors_vm = (pred_vm - true_vm).flatten()
axes[0].hist(errors_vm * 1000, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Voltage Magnitude Error (×10⁻³ pu)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title(f'Vm Error Distribution (μ={np.mean(errors_vm)*1000:.3f}, σ={np.std(errors_vm)*1000:.3f})', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

errors_va = (pred_va - true_va).flatten()
axes[1].hist(np.degrees(errors_va), bins=50, color='darkorange', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Voltage Angle Error (degrees)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title(f'Va Error Distribution (μ={np.degrees(np.mean(errors_va)):.3f}°, σ={np.degrees(np.std(errors_va)):.3f}°)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: error_distributions.png")

# Save summary
summary = {
    'dataset': {
        'total_scenarios': int(n_scenarios),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'n_buses': int(n_buses)
    },
    'model': {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params)
    },
    'training': {
        'epochs': int(epoch + 1),
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_losses[-1]),
        'training_time_seconds': float(total_time)
    },
    'test_performance': {
        'vm_mae': float(mae_vm),
        'vm_rmse': float(rmse_vm),
        'vm_max_error': float(max_error_vm),
        'vm_mape': float(mape_vm),
        'vm_r2': float(r2_vm),
        'va_mae': float(mae_va),
        'va_rmse': float(rmse_va),
        'va_max_error': float(max_error_va),
        'va_r2': float(r2_va)
    }
}

import json
with open(output_dir / 'ml_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: ml_summary.json")

print("\n" + "="*80)
print("MACHINE LEARNING TRAINING COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {output_dir.absolute()}")
print("\nKey Results:")
print(f"  • Voltage Magnitude MAE: {mae_vm*100:.4f}% ({mae_vm:.6f} pu)")
print(f"  • Voltage Angle MAE: {np.degrees(mae_va):.4f}°")
print(f"  • R² Score (Vm): {r2_vm:.6f}")
print(f"  • Training Time: {total_time/60:.1f} minutes")
print(f"  • Model saved as: best_model.pth")
print("="*80)