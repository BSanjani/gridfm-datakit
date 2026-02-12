"""
Plot frequency vs scenarios for IEEE 24-bus with deadband
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your data
data_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithDeadband_10k\case24_ieee_rts\raw")

print("="*70)
print(" "*20 + "FREQUENCY PLOT - WITH DEADBAND")
print("="*70)

# Load runtime data
print("\nLoading data...")
runtime_data = pd.read_parquet(data_dir / "runtime_data.parquet")
print(f"✓ Loaded {len(runtime_data)} scenarios")
print(f"  Columns: {list(runtime_data.columns)}")

# Extract frequency deviation and calculate actual frequency
# Assuming base frequency = 60 Hz (or 1.0 p.u.)
df = runtime_data['frequency_deviation'].values
frequency = 60.0 + df * 60.0  # Convert p.u. deviation to Hz

scenarios = np.arange(len(frequency))

print(f"\nFrequency Statistics:")
print(f"  Mean: {frequency.mean():.4f} Hz")
print(f"  Min:  {frequency.min():.4f} Hz")
print(f"  Max:  {frequency.max():.4f} Hz")
print(f"  Std:  {frequency.std():.4f} Hz")

# Create plot
print("\nCreating plot...")

plt.figure(figsize=(16, 8))
plt.plot(scenarios, frequency, linewidth=0.5, alpha=0.7, color='#2E86AB')
plt.axhline(y=60.0, color='green', linestyle='--', linewidth=2, label='Nominal (60 Hz)')
plt.axhline(y=60.0 + 0.0006*60, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label='Deadband Upper (60.036 Hz)')
plt.axhline(y=60.0 - 0.0006*60, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label='Deadband Lower (59.964 Hz)')

plt.xlabel('Scenario Number', fontsize=14, fontweight='bold')
plt.ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
plt.title('System Frequency Across Scenarios - IEEE 24-Bus with Droop & Deadband\n'
          f'{len(scenarios):,} Scenarios', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
output_file = 'frequency_vs_scenarios.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_file}")

plt.show()

print("\n✅ Done!")