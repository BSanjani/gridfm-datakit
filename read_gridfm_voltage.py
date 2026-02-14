import pandas as pd
import os
import gridfm_datakit as gfd

# Load the IEEE 14-bus case
print("=" * 80)
print("IEEE 14-BUS TRANSFORMER DATA (GRIDFM)")
print("=" * 80)

case = gfd.network.load_net_from_pglib('case14_ieee')

# Branch data includes both lines and transformers
branches = case.branches

print(f"\nTotal branches (lines + transformers): {len(branches)}")

# In MATPOWER format, transformers have non-zero TAP ratio
# TAP column index
TAP_col = gfd.network.TAP
SHIFT_col = gfd.network.SHIFT
F_BUS_col = gfd.network.F_BUS
T_BUS_col = gfd.network.T_BUS
BR_R_col = gfd.network.BR_R
BR_X_col = gfd.network.BR_X

print("\n" + "=" * 80)
print("TRANSFORMER IDENTIFICATION")
print("=" * 80)
print("\nBranches with TAP ratio ≠ 0 or ≠ 1 (likely transformers):")
print("-" * 80)

# Create a dataframe for easier viewing
branch_data = []
for i, branch in enumerate(branches):
    f_bus = int(branch[F_BUS_col])
    t_bus = int(branch[T_BUS_col])
    tap = branch[TAP_col]
    shift = branch[SHIFT_col]
    r = branch[BR_R_col]
    x = branch[BR_X_col]
    
    # Transformers typically have tap != 0 and tap != 1, or non-zero shift
    is_transformer = (tap != 0.0 and abs(tap - 1.0) > 0.0001) or abs(shift) > 0.0001
    
    branch_data.append({
        'Branch': i + 1,
        'From Bus (GRIDFM)': f_bus,
        'To Bus (GRIDFM)': t_bus,
        'From Bus (PSCAD)': f_bus + 1,
        'To Bus (PSCAD)': t_bus + 1,
        'TAP Ratio': tap,
        'Phase Shift (deg)': shift,
        'R (pu)': r,
        'X (pu)': x,
        'Type': 'Transformer' if is_transformer else 'Line'
    })

df = pd.DataFrame(branch_data)

# Show transformers
transformers = df[df['Type'] == 'Transformer']
if len(transformers) > 0:
    print(transformers.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Found {len(transformers)} transformers in the network")
    print("=" * 80)
    
    # Key transformer connections
    print("\nKey Transformer Connections:")
    print("-" * 80)
    for _, row in transformers.iterrows():
        print(f"Branch {row['Branch']}: Bus {row['From Bus (PSCAD)']} → Bus {row['To Bus (PSCAD)']}")
        print(f"  TAP ratio: {row['TAP Ratio']:.6f}")
        print(f"  Phase shift: {row['Phase Shift (deg)']:.6f}°")
        print(f"  Impedance: {row['R (pu)']:.6f} + j{row['X (pu)']:.6f} pu")
        print()
else:
    print("⚠ No explicit transformers found (all TAP = 0 or 1)")
    print("\nNote: In some MATPOWER files, TAP=0 means TAP=1 (nominal)")
    print("Showing all branches:")
    print(df.to_string(index=False))

# Show all branch data for reference
print("\n" + "=" * 80)
print("ALL BRANCH DATA (for PSCAD comparison)")
print("=" * 80)
print(df.to_string(index=False))

# Save to CSV
output_file = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\gridfm_branch_data.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Branch data saved to: {output_file}")

# Also check from the parquet file
print("\n" + "=" * 80)
print("BRANCH DATA FROM RESULTS (if available)")
print("=" * 80)

results_dir = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee14_base_case\case14_ieee\raw"
branch_data_path = os.path.join(results_dir, "branch_data.parquet")

if os.path.exists(branch_data_path):
    df_branch_results = pd.read_parquet(branch_data_path)
    print(f"\nColumns in branch results: {list(df_branch_results.columns)}")
    print(f"\nFirst few rows:")
    print(df_branch_results.head(10))
else:
    print("Branch data parquet file not found")

print("\n" + "=" * 80)
print("INSTRUCTIONS FOR PSCAD COMPARISON:")
print("=" * 80)
print("1. In PSCAD, check each transformer component")
print("2. Verify TAP ratio matches the values above")
print("3. Check if PSCAD transformers have LTC (Load Tap Changing) enabled")
print("4. If LTC is enabled, DISABLE it for power flow comparison")
print("5. Verify impedance values (R and X) match")
print("=" * 80)