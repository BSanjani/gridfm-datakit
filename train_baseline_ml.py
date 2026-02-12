"""
BASELINE ML MODEL TRAINING (No Droop Control)
==============================================
Trains neural network with ONLY load (Pd, Qd) as inputs
For comparison with droop-aware model
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import time

print("="*80)
print("BASELINE ML MODEL TRAINING")
print("Without Droop Control Parameters")
print("="*80)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============================================================================
# LOAD BASELINE DATA
# ============================================================================
print("\n[1/8] Loading baseline data (no droop)...")

data_dir = Path('./data_out_baseline_no_droop_1500/case24_ieee_rts/raw')

bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')

n_scenarios = bus_data['load_scenario_idx'].nunique()
n_buses = bus_data['bus'].nunique()

print(f"✓ Loaded {n_scenarios} scenarios, {n_buses} buses")

# ============================================================================
# PREPARE FEATURES (LOAD ONLY - NO CONTROL PARAMETERS)
# ============================================================================
print("\n[2/8] Preparing features (LOAD ONLY)...")

pd_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Pd', aggfunc='first')
qd_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Qd', aggfunc='first')
vm_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Vm', aggfunc='first')
va_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Va', aggfunc='first')

# Input: ONLY Pd and Qd (no droop parameters)
X = np.concatenate([pd_pivot.values, qd_pivot.values], axis=1)

# Output: Vm and Va
y = np.concatenate([vm_pivot.values, va_pivot.values], axis=1)

print(f"✓ Input features (X): {X.shape}")
print(f"  - Load features: {n_buses * 2} (Pd + Qd)")
print(f"  - NO control parameters")
print(f"✓ Output targets (y): {y.shape}")

# ============================================================================
# TRAIN-VAL-TEST SPLIT
# ============================================================================
print("\n[3/8] Train-Validation-Test split...")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

print(f"Training:   {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test:       {X_test.shape[0]} samples")

# ============================================================================
# FEATURE SCALING
# ============================================================================
print("\n[4/8] Feature scaling...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Save scalers
with open('scaler_X_baseline.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y_baseline.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("✓ Scalers saved")

# ============================================================================
# CREATE DATASETS
# ============================================================================
print("\n[5/8] Creating PyTorch datasets...")

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

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✓ DataLoaders created (batch_size={batch_size})")

# ============================================================================
# DEFINE BASELINE MODEL
# ============================================================================
print("\n[6/8] Defining baseline neural network...")

class BaselinePowerFlowNN(nn.Module):
    """Baseline model WITHOUT control parameters"""
    def __init__(self, input_size, output_size):
        super(BaselinePowerFlowNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]   # n_buses * 2 (Pd + Qd ONLY)
output_size = y_train.shape[1]  # n_buses * 2 (Vm + Va)

model = BaselinePowerFlowNN(input_size, output_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Baseline model created")
print(f"  Input:  {input_size} (load only)")
print(f"  Output: {output_size} (voltage)")
print(f"  Parameters: {total_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================
print("\n[7/8] Training baseline model...")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

num_epochs = 200
best_val_loss = float('inf')
patience_counter = 0
max_patience = 20

train_losses = []
val_losses = []

print(f"\nTraining for up to {num_epochs} epochs...")
print("-" * 80)

start_time = time.time()

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
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
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'best_model_baseline.pth')
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

total_time = time.time() - start_time
print("-" * 80)
print(f"\n✓ Training complete!")
print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Best val loss: {best_val_loss:.6f}")
print(f"  Epochs: {len(train_losses)}")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================
print("\n[8/8] Evaluating on test set...")

checkpoint = torch.load('best_model_baseline.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
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

predictions_scaled = np.concatenate(all_predictions, axis=0)
targets_scaled = np.concatenate(all_targets, axis=0)

predictions = scaler_y.inverse_transform(predictions_scaled)
targets = scaler_y.inverse_transform(targets_scaled)

pred_vm = predictions[:, :n_buses]
pred_va = predictions[:, n_buses:]
true_vm = targets[:, :n_buses]
true_va = targets[:, n_buses:]

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_vm = mean_absolute_error(true_vm.flatten(), pred_vm.flatten())
rmse_vm = np.sqrt(mean_squared_error(true_vm.flatten(), pred_vm.flatten()))
r2_vm = r2_score(true_vm.flatten(), pred_vm.flatten())

mae_va = mean_absolute_error(true_va.flatten(), pred_va.flatten())
r2_va = r2_score(true_va.flatten(), pred_va.flatten())

print(f"\nBASELINE MODEL PERFORMANCE:")
print("="*80)
print(f"\nVoltage Magnitude:")
print(f"  MAE:  {mae_vm:.6f} pu ({mae_vm*100:.4f}%)")
print(f"  RMSE: {rmse_vm:.6f} pu")
print(f"  R²:   {r2_vm:.6f}")

print(f"\nVoltage Angle:")
print(f"  MAE:  {mae_va:.6f} deg")
print(f"  R²:   {r2_va:.6f}")

# Save summary
summary = {
    'model_type': 'Baseline (No Droop)',
    'input_features': int(input_size),
    'output_features': int(output_size),
    'total_parameters': int(total_params),
    'training_epochs': int(len(train_losses)),
    'best_val_loss': float(best_val_loss),
    'test_performance': {
        'vm_mae_pu': float(mae_vm),
        'vm_mae_pct': float(mae_vm * 100),
        'vm_rmse': float(rmse_vm),
        'vm_r2': float(r2_vm),
        'va_mae': float(mae_va),
        'va_r2': float(r2_va)
    }
}

import json
with open('baseline_model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Summary saved to: baseline_model_summary.json")
print("\n" + "="*80)
print("BASELINE MODEL TRAINING COMPLETE!")
print("="*80)
print("\nNext: Run comparison script")
print("  python compare_droop_vs_baseline.py")
print("="*80)
