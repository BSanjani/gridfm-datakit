import pandas as pd
import numpy as np
import os

# --- INPUTS ---
PARQUET_FILE = r"data_out/case24_ieee_rts/droop_control/gen_data_droop_response_sample.parquet"
OUTPUT_CSV = os.path.abspath("psse_inputs.csv")

def prepare_data():
    print("--- STEP 1.5: FIXING IDs ---")
    
    # 1. Load Data
    df = pd.read_parquet(PARQUET_FILE)
    target_scenario = df['scenario'].unique()[0]
    df_scenario = df[df['scenario'] == target_scenario].copy()

    # 2. FIX: Create PSS/E Bus Numbers (Index + 1)
    df_scenario['bus_number'] = df_scenario['bus'].astype(int) + 1
    
    # 3. FIX: Create Unique Generator IDs
    # Group by Bus and assign sequential IDs (1, 2, 3...) for each gen on that bus
    print("Generating sequential IDs for multiple generators on the same bus...")
    df_scenario['gen_id'] = df_scenario.groupby('bus_number').cumcount() + 1
    df_scenario['gen_id'] = df_scenario['gen_id'].astype(str)

    # 4. Export
    output_df = pd.DataFrame()
    output_df['bus'] = df_scenario['bus_number']
    output_df['id'] = df_scenario['gen_id']
    output_df['p_mw'] = df_scenario['p_mw_droop']
    output_df['q_mvar'] = df_scenario['q_mvar_droop']

    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n--- NEW DATA PREVIEW ---")
    print(output_df.head(5))
    print("-" * 30)

if __name__ == "__main__":
    prepare_data()