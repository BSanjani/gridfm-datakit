"""
CONTROL-AWARE POWER FLOW ML MODEL
==================================
Trains a neural network that takes:
INPUTS:  Load (Pd, Qd) + Control parameters (mp_droop, mq_droop, K_I_secondary)
OUTPUTS: Voltage profiles (Vm, Va)

This model learns how droop control and AGC affect power flow solutions.
"""

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
print("CONTROL-AWARE POWER FLOW - MACHINE LEARNING")
print("Training with Droop + AGC Parameters")
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

data_dir = Path('./data_out_droop_secondary_varied_1500/case24_ieee_rts/raw')

print(f"Loading from: {data_dir}")

bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')
gen_data = pd.read_parquet(data_dir / 'gen_data.parquet')

print(f"\n✓ Loaded bus_data: {len(bus_data)} records")
print(f"✓ Loaded gen_data: {len(gen_data)} records")

print(f"\nBus data columns: {list(bus_data.columns)}")
print(f"Gen data columns: {list(gen_data.columns)}")

# Check unique scenarios
n_scenarios = bus_data['load_scenario_idx'].nunique()
n_buses = bus_data['bus'].nunique()
n_gens_per_scenario = gen_data.groupby('load_scenario_idx').size().iloc[0]

print(f"\nDataset: {n_scenarios} scenarios × {n_buses} buses")
print(f"Generators per scenario: {n_gens_per_scenario}")

# ============================================================================
# STEP 2: PREPARE FEATURES AND TARGETS WITH CONTROL PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PREPARING FEATURES WITH CONTROL PARAMETERS")
print("="*80)

# Pivot bus data to get one row per scenario
print("\nPivoting bus data...")
pd_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Pd', aggfunc='first')
qd_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Qd', aggfunc='first')
vm_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Vm', aggfunc='first')
va_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Va', aggfunc='first')

print(f"✓ Pivoted data shape: {pd_pivot.shape}")

# Extract control parameters (one value per scenario)
print("\nExtracting control parameters per scenario...")
control_params = gen_data.groupby('load_scenario_idx')[['mp_droop', 'mq_droop', 'K_I_secondary']].first()

# Align control params with pivoted data
control_params = control_params.loc[pd_pivot.index]

print(f"✓ Control parameters shape: {control_params.shape}")
print(f"\nControl parameter ranges:")
print(f"  mp_droop:      {control_params['mp_droop'].min():.4f} - {control_params['mp_droop'].max():.4f}")
print(f"  mq_droop:      {control_params['mq_droop'].min():.4f} - {control_params['mq_droop'].max():.4f}")
print(f"  K_I_secondary: {control_params['K_I_secondary'].min():.4f} - {control_params['K_I_secondary'].max():.4f}")

# Combine into feature matrix: [Pd, Qd, mp, mq, K_I]
X_pd = pd_pivot.values         # (n_scenarios, n_buses)
X_qd = qd_pivot.values         # (n_scenarios, n_buses)
X_mp = control_params['mp_droop'].values.reshape(-1, 1)          # (n_scenarios, 1)
X_mq = control_params['mq_droop'].values.reshape(-1, 1)          # (n_scenarios, 1)
X_K_I = control_params['K_I_secondary'].values.reshape(-1, 1)    # (n_scenarios, 1)

# Concatenate: [n_buses*2 load features + 3 control features]
X = np.concatenate([X_pd, X_qd, X_mp, X_mq, X_K_I], axis=1)

# Target matrix: [Vm, Va]
y_vm = vm_pivot.values  # (n_scenarios, n_buses)
y_va = va_pivot.values  # (n_scenarios, n_buses)
y = np.concatenate([y_vm, y_va], axis=1)  # (n_scenarios, n_buses * 2)

print(f"\n✓ Input features (X): {X.shape}")
print(f"  - Load features: {n_buses * 2} (Pd + Qd)")
print(f"  - Control features: 3 (mp_droop, mq_droop, K_I_secondary)")
print(f"  - Total: {X.shape[1]} features")
print(f"\n✓ Output targets (y): {y.shape}")
print(f"  - Voltage magnitude: {n_buses}")
print(f"  - Voltage angle: {n_buses}")

print(f"\nFeature statistics:")
print(f"  Pd range:      [{X_pd.min():.2f}, {X_pd.max():.2f}] MW")
print(f"  Qd range:      [{X_qd.min():.2f}, {X_qd.max():.2f}] MVAr")
print(f"\nTarget statistics:")
print(f"  Vm range:      [{y_vm.min():.4f}, {y_vm.max():.4f}] pu")
print(f"  Va range:      [{y_va.min():.4f}, {y_va.max():.4f}] deg")

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
with open('scaler_X_control_aware.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y_control_aware.pkl', 'wb') as f:
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

batch_size = 32  # Smaller batch size for small dataset
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
print("STEP 6: DEFINING CONTROL-AWARE NEURAL NETWORK")
print("="*80)

class ControlAwarePowerFlowNN(nn.Module):
    """
    Neural network that learns power flow with droop control and AGC.
    
    Architecture designed to learn the relationship between:
    - Load conditions (Pd, Qd)
    - Control parameters (mp_droop, mq_droop, K_I_secondary)
    - Voltage profiles (Vm, Va)
    """
    def __init__(self, input_size, output_size):
        super(ControlAwarePowerFlowNN, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer - process all features
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 1
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Hidden layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]   # n_buses * 2 + 3 control params
output_size = y_train.shape[1]  # n_buses * 2 (Vm, Va)

model = ControlAwarePowerFlowNN(input_size, output_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✓ Control-Aware Model created:")
print(f"  Input size:  {input_size} features")
print(f"    - Load features: {n_buses * 2} (Pd + Qd)")
print(f"    - Control features: 3 (mp, mq, K_I)")
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
    optimizer, mode='min', factor=0.5, patience=10
)

print("✓ Loss function: MSE")
print("✓ Optimizer: Adam (lr=0.001)")
print("✓ LR Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)")

# ============================================================================
# STEP 8: TRAINING LOOP WITH EPOCHS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAINING WITH EPOCHS")
print("="*80)

num_epochs = 200
best_val_loss = float('inf')
patience_counter = 0
max_patience = 20

train_losses = []
val_losses = []

print(f"\nStarting training for {num_epochs} epochs...")
print(f"Early stopping patience: {max_patience} epochs")
print("-" * 80)

start_time = time.time()

for epoch in range(num_epochs):
    # ========== TRAINING PHASE ==========
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
    
    # ========== VALIDATION PHASE ==========
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
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s")
    
    # Early stopping with model saving
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
            # 'scaler_X': scaler_X,
            # 'scaler_y': scaler_y,
        }, 'best_model_control_aware.pth')
        print(f"         ✓ Best model saved (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"  No improvement for {max_patience} epochs")
            break

total_time = time.time() - start_time
print("-" * 80)
print(f"\n✓ Training complete!")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"  Final epoch: {epoch+1}/{num_epochs}")
print(f"  Total epochs trained: {len(train_losses)}")

# ============================================================================
# STEP 9: LOAD BEST MODEL AND EVALUATE
# ============================================================================
print("\n" + "="*80)
print("STEP 9: EVALUATION ON TEST SET")
print("="*80)

# Load best model
checkpoint = torch.load('best_model_control_aware.pth', weights_only=False)
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
print(f"  MAE:        {mae_va:.6f} deg")
print(f"  RMSE:       {rmse_va:.6f} deg")
print(f"  Max Error:  {max_error_va:.6f} deg")

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

output_dir = Path('./ml_results_control_aware')
output_dir.mkdir(exist_ok=True)

# Plot 1: Training history
fig, ax = plt.subplots(figsize=(10, 6))
epochs_range = range(1, len(train_losses) + 1)
ax.plot(epochs_range, train_losses, label='Training Loss', linewidth=2, color='steelblue')
ax.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2, color='darkorange')
ax.axvline(checkpoint['epoch'] + 1, color='red', linestyle='--', 
           label=f'Best Model (epoch {checkpoint["epoch"]+1})', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
ax.set_title('Training History - Control-Aware Power Flow NN\n(with Droop + AGC Parameters)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: training_history.png")

# Plot 2: Voltage magnitude predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(true_vm.flatten(), pred_vm.flatten(), s=2, alpha=0.6, color='steelblue')
ax.plot([true_vm.min(), true_vm.max()], [true_vm.min(), true_vm.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('True Voltage Magnitude (pu)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Voltage Magnitude (pu)', fontsize=12, fontweight='bold')
ax.set_title(f'Voltage Magnitude Predictions\n(MAE: {mae_vm:.6f} pu, R²: {r2_vm:.4f})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'vm_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: vm_predictions.png")

# Plot 3: Error distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

errors_vm = (pred_vm - true_vm).flatten()
axes[0].hist(errors_vm * 1000, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Voltage Magnitude Error (×10⁻³ pu)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].set_title(f'Vm Error Distribution\n(μ={np.mean(errors_vm)*1000:.3f}, σ={np.std(errors_vm)*1000:.3f})', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

errors_va = (pred_va - true_va).flatten()
axes[1].hist(errors_va, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Voltage Angle Error (deg)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].set_title(f'Va Error Distribution\n(μ={np.mean(errors_va):.3f}°, σ={np.std(errors_va):.3f}°)', 
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
        'n_buses': int(n_buses),
        'input_features': int(input_size),
        'load_features': int(n_buses * 2),
        'control_features': 3
    },
    'control_parameters': {
        'mp_droop_range': [float(X_mp.min()), float(X_mp.max())],
        'mq_droop_range': [float(X_mq.min()), float(X_mq.max())],
        'K_I_secondary_range': [float(X_K_I.min()), float(X_K_I.max())]
    },
    'model': {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'architecture': 'Control-Aware Power Flow NN'
    },
    'training': {
        'epochs_trained': int(len(train_losses)),
        'best_epoch': int(checkpoint['epoch'] + 1),
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
with open(output_dir / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: training_summary.json")

print("\n" + "="*80)
print("CONTROL-AWARE ML TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 TRAINING SUMMARY:")
print("="*80)
print(f"\n1. DATASET:")
print(f"   • {n_scenarios} scenarios with varied droop and AGC")
print(f"   • Input features: {input_size} ({n_buses*2} load + 3 control)")
print(f"   • Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

print(f"\n2. TRAINING:")
print(f"   • Epochs trained: {len(train_losses)}")
print(f"   • Best epoch: {checkpoint['epoch']+1}")
print(f"   • Training time: {total_time/60:.1f} minutes")
print(f"   • Best val loss: {best_val_loss:.6f}")

print(f"\n3. TEST PERFORMANCE:")
print(f"   • Voltage MAE: {mae_vm*100:.4f}% ({mae_vm:.6f} pu)")
print(f"   • Voltage R²: {r2_vm:.6f}")
print(f"   • Max error: {max_error_vm:.6f} pu")

print(f"\n4. MODEL LEARNS:")
print(f"   ✓ Effect of mp_droop on power flow")
print(f"   ✓ Effect of mq_droop on voltage")
print(f"   ✓ Effect of K_I_secondary (AGC) on system response")

print(f"\n📁 RESULTS SAVED TO:")
print(f"   {output_dir.absolute()}")

print("\n" + "="*80)
print("✓ READY TO USE FOR INFERENCE!")
print("="*80)