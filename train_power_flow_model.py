"""
Power Flow Neural Network Trainer
Train a neural network to predict power flow solutions from load conditions
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class PowerFlowDataset(Dataset):
    """Dataset for Power Flow prediction"""
    
    def __init__(self, bus_data, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_and_prepare_data(data_path='data_out/case24_ieee_rts/raw'):
    """Load and prepare data for ML training"""
    
    print("Loading data...")
    bus_data = pd.read_parquet(f'{data_path}/bus_data.parquet')
    
    print(f"Loaded {len(bus_data):,} samples")
    print(f"Number of scenarios: {bus_data['scenario'].nunique():,}")
    print(f"Number of buses: {bus_data['bus'].nunique()}")
    
    # Prepare features and targets per scenario
    n_scenarios = bus_data['scenario'].nunique()
    n_buses = bus_data['bus'].nunique()
    
    # Features: Active and Reactive loads (Pd, Qd) for all buses
    # Targets: Voltage magnitude (Vm) and angle (Va) for all buses
    
    features_list = []
    targets_list = []
    
    for scenario in range(n_scenarios):
        scenario_data = bus_data[bus_data['scenario'] == scenario].sort_values('bus')
        
        # Features: Pd and Qd for all buses (flattened)
        features = np.concatenate([
            scenario_data['Pd'].values,
            scenario_data['Qd'].values
        ])
        
        # Targets: Vm and Va for all buses (flattened)
        targets = np.concatenate([
            scenario_data['Vm'].values,
            scenario_data['Va'].values
        ])
        
        features_list.append(features)
        targets_list.append(targets)
    
    features_array = np.array(features_list)
    targets_array = np.array(targets_list)
    
    print(f"\nFeature shape: {features_array.shape}")
    print(f"Target shape: {targets_array.shape}")
    
    return features_array, targets_array, n_buses


# ============================================================================
# 2. NEURAL NETWORK MODEL
# ============================================================================

class PowerFlowNN(nn.Module):
    """Neural Network for Power Flow prediction"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super(PowerFlowNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Complete training loop"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining on {device}...")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Best Val':<12}")
    print("-" * 50)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_power_flow_model.pth')
            print(f"✓ Epoch {epoch+1:<3} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | ⭐ New Best!")
        else:
            print(f"  Epoch {epoch+1:<3} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


# ============================================================================
# 4. EVALUATION AND VISUALIZATION
# ============================================================================

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('MSE Loss', fontsize=12, fontweight='bold')
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_predictions(model, test_loader, n_buses, device='cpu'):
    """Evaluate model predictions on test set"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Split into voltage magnitude and angle
    vm_pred = predictions[:, :n_buses]
    va_pred = predictions[:, n_buses:]
    vm_true = targets[:, :n_buses]
    va_true = targets[:, n_buses:]
    
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
    print(f"\nVoltage Angle (Va):")
    print(f"  MAE:  {va_mae:.6f} degrees")
    print(f"  RMSE: {va_rmse:.6f} degrees")
    print("="*60)
    
    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Voltage magnitude
    axes[0].scatter(vm_true.flatten(), vm_pred.flatten(), alpha=0.3, s=1)
    axes[0].plot([vm_true.min(), vm_true.max()], 
                 [vm_true.min(), vm_true.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Vm (p.u.)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Vm (p.u.)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Voltage Magnitude\nMAE: {vm_mae:.6f} p.u.', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Voltage angle
    axes[1].scatter(va_true.flatten(), va_pred.flatten(), alpha=0.3, s=1)
    axes[1].plot([va_true.min(), va_true.max()], 
                 [va_true.min(), va_true.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Va (degrees)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted Va (degrees)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Voltage Angle\nMAE: {va_mae:.6f} deg', 
                     fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return vm_mae, va_mae


# ============================================================================
# 5. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    features, targets, n_buses = load_and_prepare_data()
    
    # Normalize data
    print("\nNormalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    features_scaled = scaler_X.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, targets_scaled, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Training:   {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test:       {len(X_test):,} samples")
    
    # Create datasets and dataloaders
    train_dataset = PowerFlowDataset(None, X_train, y_train)
    val_dataset = PowerFlowDataset(None, X_val, y_val)
    test_dataset = PowerFlowDataset(None, X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Create model
    input_dim = features.shape[1]  # 2 * n_buses (Pd and Qd)
    output_dim = targets.shape[1]  # 2 * n_buses (Vm and Va)
    
    model = PowerFlowNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[512, 256, 128]
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=50, lr=0.001, device=device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_power_flow_model.pth'))
    vm_mae, va_mae = evaluate_predictions(model, test_loader, n_buses, device)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: best_power_flow_model.pth")
    print(f"📊 Plots saved to: training_history.png, prediction_accuracy.png")
    
    return model, scaler_X, scaler_y


if __name__ == "__main__":
    model, scaler_X, scaler_y = main()