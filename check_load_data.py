import pandas as pd

# Path to GRIDFM results
results_dir = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee14_droop_withdeadband\case14_ieee\raw"

print("=" * 80)
print("GRIDFM BRANCH DATA - ALL LINES AND TRANSFORMERS")
print("=" * 80)

# Read branch data
df_branch = pd.read_parquet(results_dir + r"\branch_data.parquet")

# Create clean table
df_branch['From_PSCAD'] = df_branch['from_bus'] + 1
df_branch['To_PSCAD'] = df_branch['to_bus'] + 1

# Select relevant columns
branch_table = df_branch[[
    'idx', 'From_PSCAD', 'To_PSCAD', 'from_bus', 'to_bus',
    'r', 'x', 'b', 'tap', 'shift',
    'pf', 'qf', 'pt', 'qt',
    'br_status'
]].copy()

branch_table.columns = [
    'Branch', 'From(PSCAD)', 'To(PSCAD)', 'From(GRIDFM)', 'To(GRIDFM)',
    'R(pu)', 'X(pu)', 'B(pu)', 'TAP', 'Shift(deg)',
    'Pf(MW)', 'Qf(MVAr)', 'Pt(MW)', 'Qt(MVAr)',
    'Status'
]

# Identify transformers (non-zero tap)
branch_table['Type'] = branch_table.apply(
    lambda row: 'Transformer' if (row['TAP'] != 0 and abs(row['TAP'] - 1.0) > 0.0001) else 'Line',
    axis=1
)

print("\nALL 20 BRANCHES:")
print("=" * 80)
print(branch_table.to_string(index=False))

# Separate transformers
print("\n" + "=" * 80)
print("TRANSFORMERS ONLY:")
print("=" * 80)
transformers = branch_table[branch_table['Type'] == 'Transformer']
print(transformers.to_string(index=False))

# Separate lines
print("\n" + "=" * 80)
print("TRANSMISSION LINES ONLY:")
print("=" * 80)
lines = branch_table[branch_table['Type'] == 'Line']
print(lines[['Branch', 'From(PSCAD)', 'To(PSCAD)', 'R(pu)', 'X(pu)', 'B(pu)']].to_string(index=False))

# Power flow summary
print("\n" + "=" * 80)
print("POWER FLOW SUMMARY:")
print("=" * 80)

# Total generation
df_bus = pd.read_parquet(results_dir + r"\bus_data.parquet")
total_gen = df_bus['Pg'].sum()
total_load = df_bus['Pd'].sum()
total_qgen = df_bus['Qg'].sum()
total_qload = df_bus['Qd'].sum()

print(f"Total Generation (P): {total_gen:.2f} MW")
print(f"Total Load (P):       {total_load:.2f} MW")
print(f"Losses (P):           {total_gen - total_load:.2f} MW")
print(f"Loss %:               {(total_gen - total_load)/total_gen * 100:.2f}%")

print(f"\nTotal Generation (Q): {total_qgen:.2f} MVAr")
print(f"Total Load (Q):       {total_qload:.2f} MVAr")
print(f"Shunt/Losses (Q):     {total_qgen - total_qload:.2f} MVAr")

# Line losses
print("\n" + "=" * 80)
print("BRANCH LOSSES:")
print("=" * 80)

df_branch['P_loss'] = df_branch['pf'] + df_branch['pt']
df_branch['Q_loss'] = df_branch['qf'] + df_branch['qt']

loss_table = df_branch[['idx', 'from_bus', 'to_bus', 'P_loss', 'Q_loss']].copy()
loss_table['From_PSCAD'] = loss_table['from_bus'] + 1
loss_table['To_PSCAD'] = loss_table['to_bus'] + 1
loss_table = loss_table[['idx', 'From_PSCAD', 'To_PSCAD', 'P_loss', 'Q_loss']]
loss_table.columns = ['Branch', 'From', 'To', 'P Loss(MW)', 'Q Loss(MVAr)']

print(loss_table.to_string(index=False))

total_p_loss = df_branch['P_loss'].sum()
total_q_loss = df_branch['Q_loss'].sum()

print(f"\nTotal P Losses: {total_p_loss:.2f} MW")
print(f"Total Q Losses: {total_q_loss:.2f} MVAr")

# Export to CSV
output_file = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\gridfm_branch_data_detailed.csv"
branch_table.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

print("\n" + "=" * 80)
print("FOR PSCAD COMPARISON:")
print("=" * 80)
print("1. Verify all R, X, B values match PSCAD branch impedances")
print("2. Verify transformer TAPs match (0.978, 0.969, 0.932)")
print("3. Total losses should be ~13-15 MW for IEEE 14-bus")
print("4. If PSCAD losses are different, check branch parameters")
print("=" * 80)