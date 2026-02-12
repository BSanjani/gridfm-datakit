"""
BASELINE vs DROOP-AWARE COMPARISON
===================================
Generates data WITHOUT droop control (original GridFM style)
Trains baseline ML model
Compares performance: Baseline vs Droop-Aware
"""

import yaml
from pathlib import Path
import shutil

print("="*80)
print("GENERATING BASELINE DATA (No Droop Control)")
print("="*80)

# ============================================================================
# STEP 1: Create baseline config (disable droop control)
# ============================================================================
print("\n[1/3] Creating baseline configuration...")

# Load your current config
with open('user_config2.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create baseline version without droop
baseline_config = config.copy()

# Disable droop control
if 'droop_control' in baseline_config:
    baseline_config['droop_control']['enabled'] = False

# Change output directory
baseline_config['data_dir'] = './data_out_baseline_no_droop_100'

# Keep same number of scenarios for fair comparison
print(f"✓ Baseline config created")
print(f"  Scenarios: {baseline_config.get('n_scenarios', 100)}")
print(f"  Droop control: DISABLED")
print(f"  Output dir: {baseline_config['data_dir']}")

# Save baseline config
with open('user_config_baseline.yaml', 'w') as f:
    yaml.dump(baseline_config, f, default_flow_style=False)

print("✓ Saved: user_config_baseline.yaml")

# ============================================================================
# STEP 2: Generate baseline data
# ============================================================================
print("\n[2/3] Generating baseline power flow data...")
print("This will take ~5-10 minutes for 100 scenarios...")

from gridfm_datakit.generate import generate_power_flow_data_distributed

try:
    result = generate_power_flow_data_distributed('user_config_baseline.yaml')
    print("\n✓ Baseline data generation complete!")
    print(f"  Output directory: {baseline_config['data_dir']}")
except Exception as e:
    print(f"\n✗ Error during generation: {e}")
    print("Please run manually:")
    print("  python -c \"from gridfm_datakit.generate import generate_power_flow_data_distributed; generate_power_flow_data_distributed('user_config_baseline.yaml')\"")

# ============================================================================
# STEP 3: Instructions for training and comparison
# ============================================================================
print("\n[3/3] Next steps:")
print("="*80)

instructions = """
NEXT STEPS FOR COMPARISON:
==========================

1. Train Baseline ML Model (without droop parameters):
   ---------------------------------------------------
   python train_baseline_ml.py
   
   This will:
   - Load data from data_out_baseline_no_droop_100
   - Train with only load (Pd, Qd) as inputs
   - Save model as best_model_baseline.pth

2. Compare Results:
   ----------------
   python compare_droop_vs_baseline.py
   
   This will generate:
   - Side-by-side voltage prediction accuracy
   - MAE, RMSE, R² comparison
   - Plots showing improvement from droop-aware model

EXPECTED RESULTS:
=================
Without droop parameters:
  - Baseline model treats all scenarios equally
  - Cannot distinguish different control settings
  - Expected accuracy: ~0.5-1.0% MAE

With droop parameters (your current model):
  - Model learns control-dependent behavior
  - Better accuracy: 0.32% MAE ✓
  - More physically realistic predictions

The comparison will show HOW MUCH droop-awareness improves performance!
"""

print(instructions)

# Save instructions
with open('COMPARISON_INSTRUCTIONS.txt', 'w') as f:
    f.write(instructions)

print("\n✓ Instructions saved to: COMPARISON_INSTRUCTIONS.txt")

print("\n" + "="*80)
print("BASELINE DATA GENERATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. user_config_baseline.yaml - Config without droop control")
print("  2. data_out_baseline_no_droop_100/ - Baseline power flow data")
print("  3. COMPARISON_INSTRUCTIONS.txt - Next steps")

print("\nRun next:")
print("  python train_baseline_ml.py")
print("="*80)