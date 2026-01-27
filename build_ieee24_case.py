"""
Build IEEE 24-Bus case directly in PSS/E from GridFM data
Uses PSS/E API instead of RAW files
"""
import psspy
import pandas as pd
import os

print("="*60)
print("Building IEEE 24-Bus Case in PSS/E from GridFM Data")
print("="*60)

# Initialize PSS/E
psspy.psseinit(50000)

# Load GridFM data
data_path = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out\case24_ieee_rts\raw'
print("\nLoading GridFM data...")
bus_df = pd.read_parquet(os.path.join(data_path, 'bus_data.parquet'))
gen_df = pd.read_parquet(os.path.join(data_path, 'gen_data.parquet'))
branch_df = pd.read_parquet(os.path.join(data_path, 'branch_data.parquet'))

# Filter to scenario 0
bus_data = bus_df[bus_df['scenario'] == 0].reset_index(drop=True)
gen_data = gen_df[gen_df['scenario'] == 0].reset_index(drop=True)
branch_data = branch_df[branch_df['scenario'] == 0].reset_index(drop=True)

print(f"Loaded {len(bus_data)} buses, {len(gen_data)} generators, {len(branch_data)} branches")

# Create new case
print("\nCreating new case in PSS/E...")
psspy.case('dummy')  # Start with empty case

# Add buses
print("\nAdding buses...")
for idx, bus in bus_data.iterrows():
    bus_num = int(bus['bus']) + 1
    bus_name = f'BUS{bus_num}'
    base_kv = float(bus['vn_kv'])
    
    # Determine bus type
    if bus['REF'] == 1:
        ide = 3  # Slack
    elif bus['PV'] == 1:
        ide = 2  # PV
    else:
        ide = 1  # PQ
    
    ierr = psspy.bus_data_3(
        i=bus_num,
        intgar1=ide,  # Bus type
        realar1=base_kv,  # Base kV
        name=bus_name
    )
    if ierr != 0:
        print(f"  Warning: Bus {bus_num} add error {ierr}")

ierr, nbuses = psspy.abuscount()
print(f"Added {nbuses} buses")

# Add branches
print("\nAdding branches...")
for idx, branch in branch_data.iterrows():
    from_bus = int(branch['from_bus']) + 1
    to_bus = int(branch['to_bus']) + 1
    ckt = str(int(branch['idx']) % 10 + 1)
    r = float(branch['r'])
    x = float(branch['x'])
    b = float(branch['b'])
    rate = float(branch.get('rate_a', 999.0))
    
    ierr = psspy.branch_data(
        i=from_bus,
        j=to_bus,
        ckt=ckt,
        intgar1=1,  # Status (1=in-service)
        realar=[r, x, b, rate, rate, rate]
    )
    if ierr != 0 and ierr != 1:  # 1 means already exists, which is OK
        print(f"  Warning: Branch {from_bus}-{to_bus} add error {ierr}")

ierr, nbranch = psspy.abrncount()
print(f"Added {nbranch} branches")

# Add loads
print("\nAdding loads...")
for idx, bus in bus_data.iterrows():
    bus_num = int(bus['bus']) + 1
    pd = float(bus['Pd'])
    qd = float(bus['Qd'])
    
    if abs(pd) > 0.001 or abs(qd) > 0.001:
        ierr = psspy.load_data_4(
            i=bus_num,
            id='1',
            intgar1=1,  # Status
            realar=[pd, qd]
        )
        if ierr != 0:
            print(f"  Warning: Load at bus {bus_num} add error {ierr}")

# Add generators
print("\nAdding generators...")
for idx, gen in gen_data.iterrows():
    bus_num = int(gen['bus']) + 1
    gen_id = str(int(gen['idx']) % 100 + 1)
    pg = float(gen['p_mw'])
    qg = float(gen['q_mvar'])
    qmax = float(gen.get('max_q_mvar', 999.0))
    qmin = float(gen.get('min_q_mvar', -999.0))
    
    # Get voltage setpoint from bus
    bus_idx = int(gen['bus'])
    vs = float(bus_data[bus_data['bus'] == bus_idx]['Vm'].iloc[0])
    
    pmax = float(gen.get('max_p_mw', 999.0))
    pmin = float(gen.get('min_p_mw', 0.0))
    status = int(gen.get('in_service', 1))
    
    ierr = psspy.machine_data_2(
        i=bus_num,
        id=gen_id,
        intgar1=status,
        realar=[pg, qg, qmax, qmin, vs, 0.0, pmax, pmin]
    )
    if ierr != 0:
        print(f"  Warning: Generator at bus {bus_num} id {gen_id} add error {ierr}")

ierr, ngen = psspy.amachcount()
print(f"Added {ngen} generators")

# Set solution parameters
print("\nSolving power flow...")
psspy.fnsl([0,0,0,1,1,0,99,0])

print("\n" + "="*60)
print("Case built successfully!")
print("="*60)

# Save the case
save_file = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit\ieee24_gridfm.sav'
ierr = psspy.save(save_file)
if ierr == 0:
    print(f"\nCase saved to: {save_file}")
else:
    print(f"\nWarning: Save error {ierr}")

# Verify
ierr, nbuses = psspy.abuscount()
ierr, ngen = psspy.amachcount()
ierr, nbranch = psspy.abrncount()
print(f"\nFinal case statistics:")
print(f"  Buses: {nbuses}")
print(f"  Generators: {ngen}")
print(f"  Branches: {nbranch}")