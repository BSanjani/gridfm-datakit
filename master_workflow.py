import pandas as pd
import subprocess
import os

# --- CONFIGURATION ---
PSSE_PYTHON_EXE = r"C:\Users\Bestu\anaconda3\envs\gridfm-psse\python.exe" # Verify this path!
PSSE_SCRIPT = "run_psse_logic.py"
BASE_RAW_FILE = r"data_out/case24_ieee_rts/raw/case24_ieee_rts.raw" # Path to your ORIGINAL raw file
EXCHANGE_CSV = os.path.abspath("temp_setpoints.csv")

# --- 1. PREPARE DATA (GRIDFM SIDE) ---
print(f"[GridFM] Reading Droop Results...")

# Load the data (Just like in your visualization script)
df_droop = pd.read_parquet('data_out/case24_ieee_rts/droop_control/gen_data_droop_response_sample.parquet')

# PICK A SCENARIO TO TEST (e.g., the first one found)
test_scenario = df_droop['scenario'].unique()[0]
print(f"[GridFM] Extracting setpoints for Scenario: {test_scenario}")

# Filter data for this scenario
scenario_data = df_droop[df_droop['scenario'] == test_scenario].copy()

# Prepare a simple CSV for PSS/E: Bus Number, Generator ID, New P, New Q
# Assuming your parquet has columns: 'bus_id', 'ckt_id', 'p_mw_droop', 'q_mvar_droop'
# You might need to adjust column names to match your parquet exactly
exchange_df = scenario_data[['bus_id', 'ckt_id', 'p_mw_droop', 'q_mvar_droop']]
exchange_df.to_csv(EXCHANGE_CSV, index=False)

print(f"[GridFM] Setpoints saved to {EXCHANGE_CSV}")

# --- 2. CALL PSS/E (SUBPROCESS) ---
print(f"[Master] Handoff to PSS/E environment...")

try:
    # Pass BOTH the Base RAW file and the CSV of changes
    subprocess.check_call([PSSE_PYTHON_EXE, PSSE_SCRIPT, BASE_RAW_FILE, EXCHANGE_CSV])
    print("[Master] PSS/E simulation complete.")
except subprocess.CalledProcessError:
    print("[Master] PSS/E crashed or failed.")