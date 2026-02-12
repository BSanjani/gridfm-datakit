import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the same model architecture
class PowerFlowNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super(PowerFlowNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Load data
print("Loading data...")
bus_data = pd.read_parquet('data_out/case24_ieee_rts/raw/bus_data.parquet')

n_scenarios = bus_data['scenario'].nunique()
n_buses = bus_data['bus'].nunique()

# Prepare data
features_list = []
targets_list = []

for scenario in range(n_scenarios):
    scenario_data = bus_data[bus_data['scenario'] == scenario].sort_values('bus')
    features = np.concatenate([scenario_data['Pd'].values, scenario_data['Qd'].values])
    targets = np.concatenate([scenario_data['Vm'].values, scenario_data['Va'].values])
    features_list.append(features)
    targets_list.append(targets)

features = np.array(features_list)
targets = np.array(targets_list)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
features_scaled = scaler_X.fit_transform(features)
targets_scaled = scaler_y.fit_transform(targets)

# Use last 10% for testing
test_size = int(0.1 * len(features_scaled))
X_test = features_scaled[-test_size:]
y_test_scaled = targets_scaled[-test_size:]
y_test = targets[-test_size:]

# Load model
model = PowerFlowNN(input_dim=48, output_dim=48)
model.load_state_dict(torch.load('best_power_flow_model.pth'))
model.eval()

# Make predictions
print("\nMaking predictions...")
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    predictions_scaled = model(X_test_tensor).numpy()

# Denormalize predictions
predictions = scaler_y.inverse_transform(predictions_scaled)

# Split into Vm and Va
vm_pred = predictions[:, :n_buses]
va_pred = predictions[:, n_buses:]
vm_true = y_test[:, :n_buses]
va_true = y_test[:, n_buses:]

# Calculate errors
vm_mae = np.mean(np.abs(vm_pred - vm_true))
va_mae = np.mean(np.abs(va_pred - va_true))
vm_rmse = np.sqrt(np.mean((vm_pred - vm_true)**2))
va_rmse = np.sqrt(np.mean((va_pred - va_true)**2))

print("\n" + "="*60)
print("MODEL PERFORMANCE ON TEST SET")
print("="*60)
print(f"Voltage Magnitude (Vm):")
print(f"  MAE:  {vm_mae:.6f} p.u.")
print(f"  RMSE: {vm_rmse:.6f} p.u.")
print(f"  Max Error: {np.max(np.abs(vm_pred - vm_true)):.6f} p.u.")
print(f"\nVoltage Angle (Va):")
print(f"  MAE:  {va_mae:.6f} degrees")
print(f"  RMSE: {va_rmse:.6f} degrees")
print(f"  Max Error: {np.max(np.abs(va_pred - va_true)):.6f} degrees")
print("="*60)

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Voltage magnitude
axes[0].scatter(vm_true.flatten(), vm_pred.flatten(), alpha=0.3, s=1)
axes[0].plot([vm_true.min(), vm_true.max()], [vm_true.min(), vm_true.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('True Vm (p.u.)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Vm (p.u.)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Voltage Magnitude\nMAE: {vm_mae:.6f} p.u.', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Voltage angle
axes[1].scatter(va_true.flatten(), va_pred.flatten(), alpha=0.3, s=1)
axes[1].plot([va_true.min(), va_true.max()], [va_true.min(), va_true.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('True Va (degrees)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Va (degrees)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Voltage Angle\nMAE: {va_mae:.6f} deg', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✅ Evaluation complete! Plot saved to model_evaluation.png")
plt.show()