"""
Build IEEE 24-Bus case in PSS/E from pickled data
No pandas dependency - works in PSS/E Python
"""
import psspy
import pickle
import os

print("="*60)
print("Building IEEE 24-Bus Case in PSS/E")
print("="*60)

# Initialize PSS/E
psspy.psseinit(50000)

# Load the pickled data
os.chdir(r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit')
with open('ieee24_data.pkl', 'rb') as f:
    data = pickle.load(f)

bus_data = data['bus']
gen_data = data['gen']
branch_data = data['branch']

print(f"\nLoaded data:")
print(f"  Buses: {len(bus_data)}")
print(f"  Generators: {len(gen_data)}")
print(f"  Branches: {len(branch_data)}")

# Create new case
print("\nInitializing new case...")

# Add buses
print("\nAdding buses...")
for idx in range(len(bus_data)):
    bus = bus_data.iloc[idx]
    bus_num = int(bus['bus']) + 1
    bus_name = f'BUS{bus_num}'
    base_kv = float(bus['vn_kv'])
    
    if bus['REF'] == 1:
        ide = 3
    elif bus['PV'] == 1:
        ide = 2
    else:
        ide = 1
    
    ierr = psspy.bus_data_3(i=bus_num, intgar1=ide, realar1=base_kv, name=bus_name)

ierr, nbuses = psspy.abuscount()
print(f"  Added {nbuses} buses")

# Add branches
print("\nAdding branches...")
for idx in range(len(branch_data)):
    branch = branch_data.iloc[idx]
    from_bus = int(branch['from_bus']) + 1
    to_bus = int(branch['to_bus']) + 1
    ckt = str(int(branch['idx']) % 10 + 1)
    r = float(branch['r'])
    x = float(branch['x'])
    b = float(branch['b'])
    rate = float(branch['rate_a'])
    
    ierr = psspy.branch_data(i=from_bus, j=to_bus, ckt=ckt, intgar1=1,
                             realar=[r, x, b, rate, rate, rate])

ierr, nbranch = psspy.abrncount()
print(f"  Added {nbranch} branches")

# Add loads
print("\nAdding loads...")
for idx in range(len(bus_data)):
    bus = bus_data.iloc[idx]
    bus_num = int(bus['bus']) + 1
    pd = float(bus['Pd'])
    qd = float(bus['Qd'])
    
    if abs(pd) > 0.001 or abs(qd) > 0.001:
        ierr = psspy.load_data_4(i=bus_num, id='1', intgar1=1, realar=[pd, qd])

# Add generators
print("\nAdding generators...")
for idx in range(len(gen_data)):
    gen = gen_data.iloc[idx]
    bus_num = int(gen['bus']) + 1
    gen_id = str(int(gen['idx']) % 100 + 1)
    pg = float(gen['p_mw'])
    qg = float(gen['q_mvar'])
    qmax = float(gen['max_q_mvar'])
    qmin = float(gen['min_q_mvar'])
    
    bus_idx = int(gen['bus'])
    vs = float(bus_data.iloc[bus_idx]['Vm'])
    
    pmax = float(gen['max_p_mw'])
    pmin = float(gen['min_p_mw'])
    status = int(gen['in_service'])
    
    ierr = psspy.machine_data_2(i=bus_num, id=gen_id, intgar1=status,
                                realar=[pg, qg, qmax, qmin, vs, 0.0, pmax, pmin])

ierr, ngen = psspy.amachcount()
print(f"  Added {ngen} generators")

# Solve power flow
print("\nSolving power flow...")
ierr = psspy.fnsl([0,0,0,1,1,0,99,0])
print(f"  Power flow result: {ierr}")

# Save
save_file = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit\ieee24_gridfm.sav'
ierr = psspy.save(save_file)
print(f"\nCase saved to: {save_file}")
print("\n" + "="*60)
print("COMPLETE!")
print("="*60)