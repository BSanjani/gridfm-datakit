import pandas as pd
import numpy as np
import os

class GridFMtoPSSE:
    """Convert GridFM-datakit output to PSS/E format for dynamic studies"""
    
    def __init__(self, data_path):
        """
        Initialize converter with path to gridfm output data
        
        Args:
            data_path: Path to directory containing parquet files
        """
        self.data_path = data_path
        self.bus_data = None
        self.branch_data = None
        self.gen_data = None
        
    def load_gridfm_data(self, scenario=0):
        """
        Load parquet files and filter to single scenario
        
        Args:
            scenario: Scenario number to extract (default: 0)
        """
        print("Loading GridFM data...")
        
        # Load full datasets
        bus_full = pd.read_parquet(os.path.join(self.data_path, 'bus_data.parquet'))
        branch_full = pd.read_parquet(os.path.join(self.data_path, 'branch_data.parquet'))
        gen_full = pd.read_parquet(os.path.join(self.data_path, 'gen_data.parquet'))
        
        print(f"Total rows before filtering: {len(bus_full)} bus records")
        print(f"Unique scenarios: {bus_full['scenario'].nunique()}")
        
        # Filter to selected scenario
        self.bus_data = bus_full[bus_full['scenario'] == scenario].reset_index(drop=True)
        self.branch_data = branch_full[branch_full['scenario'] == scenario].reset_index(drop=True)
        self.gen_data = gen_full[gen_full['scenario'] == scenario].reset_index(drop=True)
        
        print(f"\nFiltered to scenario {scenario}:")
        print(f"  {len(self.bus_data)} buses")
        print(f"  {len(self.branch_data)} branches")
        print(f"  {len(self.gen_data)} generators")
        
    def write_psse_raw(self, output_file='ieee24_psse.raw', version=33):
        """
        Write PSS/E RAW file format
        
        Args:
            output_file: Output filename for RAW file
            version: PSS/E version (33 or 35)
        """
        print(f"\nWriting PSS/E RAW file (v{version}): {output_file}")
        
        with open(output_file, 'w') as f:
            # Header (Case Identification Data) - Version 33 format
            if version == 33:
                f.write(f"0, 100.00, 33, 0, 0, 60.00  / PSS(R)E-33 RAW created by PSSE\n")
            else:
                f.write(f" 0,   100.00, {version},  0,  1,  60.00     / PSS/E-{version} RAW\n")
            
            f.write("IEEE 24-Bus Reliability Test System from GridFM\n")
            f.write("Converted from gridfm-datakit output\n")
            
            # Bus Data
            f.write("0 / END OF SYSTEM-WIDE DATA, BEGIN BUS DATA\n")
            for idx, bus in self.bus_data.iterrows():
                bus_num = int(bus['bus']) + 1  # PSS/E buses usually start at 1
                bus_name = f"BUS{bus_num}"
                base_kv = float(bus.get('vn_kv', 138.0))
                
                # Determine bus type from flags
                if bus['REF'] == 1:
                    bus_type = 3  # Slack bus
                elif bus['PV'] == 1:
                    bus_type = 2  # PV bus
                else:
                    bus_type = 1  # PQ bus
                
                v_mag = float(bus['Vm'])
                v_ang = float(bus['Va'])
                area = 1
                zone = 1
                owner = 1
                
                # Version 33 format - simpler
                f.write(f"{bus_num}, '{bus_name}', {base_kv:.2f}, {bus_type}, "
                       f"{area}, {zone}, {owner}, {v_mag:.5f}, {v_ang:.5f}\n")
            
            # Load Data
            f.write("0 / END OF BUS DATA, BEGIN LOAD DATA\n")
            for idx, bus in self.bus_data.iterrows():
                bus_num = int(bus['bus']) + 1
                pd = float(bus['Pd'])
                qd = float(bus['Qd'])
                
                # Only write load record if there's actually load on the bus
                if abs(pd) > 0.001 or abs(qd) > 0.001:
                    load_id = '1'
                    status = 1
                    area = 1
                    zone = 1
                    
                    f.write(f"{bus_num}, '{load_id}', {status}, {area}, {zone}, "
                           f"{pd:.2f}, {qd:.2f}, 0.0, 0.0, 0.0, 0.0, 1\n")
            
            # Generator Data
            f.write("0 / END OF LOAD DATA, BEGIN GENERATOR DATA\n")
            for idx, gen in self.gen_data.iterrows():
                bus_num = int(gen['bus']) + 1  # Match bus numbering
                gen_id = str(int(gen['idx']) % 100 + 1)
                pg = float(gen['p_mw'])
                qg = float(gen['q_mvar'])
                qmax = float(gen.get('max_q_mvar', 999.0))
                qmin = float(gen.get('min_q_mvar', -999.0))
                # Get voltage setpoint from bus data
                bus_idx = int(gen['bus'])
                vs = float(self.bus_data[self.bus_data['bus'] == bus_idx]['Vm'].iloc[0])
                mbase = 100.0  # Default MVA base
                status = int(gen.get('in_service', 1))
                pmax = float(gen.get('max_p_mw', 999.0))
                pmin = float(gen.get('min_p_mw', 0.0))
                
                f.write(f"{bus_num}, '{gen_id}', {pg:.2f}, {qg:.2f}, {qmax:.2f}, {qmin:.2f}, "
                       f"{vs:.5f}, {mbase:.1f}, {status}, {pmax:.2f}, {pmin:.2f}, "
                       f"1.0, 0, 1.0, 0, 1.0, 1.0, 0, 1.0\n")
            for idx, branch in self.branch_data.iterrows():
                from_bus = int(branch['from_bus']) + 1  # Match bus numbering
                to_bus = int(branch['to_bus']) + 1
                ckt = f"'{int(branch['idx']) % 10 + 1:1d}'"
                r = float(branch['r'])
                x = float(branch['x'])
                b = float(branch.get('b', 0.0))
                rate_a = float(branch.get('rate_a', 999.0))
                status = int(branch.get('br_status', 1))
                
                f.write(f"{from_bus:6d},{to_bus:6d},{ckt:3s},{r:12.6f},{x:12.6f},"
                       f"{b:12.6f},{rate_a:10.2f},{status:2d}\n")
            
            # Transformer Data (if any)
            f.write("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA\n")
            f.write("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\n")
            f.write("0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA\n")
            f.write("0 / END OF TWO-TERMINAL DC DATA, BEGIN SWITCHED SHUNT DATA\n")
            f.write("0 / END OF SWITCHED SHUNT DATA, BEGIN IMPEDANCE CORRECTION DATA\n")
            f.write("0 / END OF IMPEDANCE CORRECTION DATA, BEGIN MULTI-TERMINAL DC DATA\n")
            f.write("0 / END OF MULTI-TERMINAL DC DATA, BEGIN MULTI-SECTION LINE DATA\n")
            f.write("0 / END OF MULTI-SECTION LINE DATA, BEGIN ZONE DATA\n")
            f.write("0 / END OF ZONE DATA, BEGIN INTER-AREA TRANSFER DATA\n")
            f.write("0 / END OF INTER-AREA TRANSFER DATA, BEGIN OWNER DATA\n")
            f.write("0 / END OF OWNER DATA, BEGIN FACTS DEVICE DATA\n")
            f.write("Q\n")
        
        print(f"RAW file created: {output_file}")
    
    def write_psse_dyr(self, output_file='ieee24_psse.dyr'):
        """
        Write PSS/E DYR file with standard dynamic models
        Using GENROU (round rotor generator model) for all generators
        
        Args:
            output_file: Output filename for DYR file
        """
        print(f"\nWriting PSS/E DYR file: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("/ IEEE 24-Bus Dynamic Model Data\n")
            f.write("/ Generated from gridfm-datakit output\n\n")
            
            for idx, gen in self.gen_data.iterrows():
                bus_num = int(gen['bus']) + 1  # Match bus numbering
                gen_id = int(gen['idx']) % 100 + 1  # Numeric ID
                
                # GENROU - Round Rotor Generator Model (6th order)
                f.write(f"{bus_num:6d} '{gen_id:2d}' 'GENROU' 1 ")
                f.write("8.0000  0.3000 /\n")  # T'do, T''do
                f.write("                      0.4000  0.0500  ")
                f.write("1.8000  0.1800  1.7000  0.5500 /\n")  # T'qo, T''qo, Xd, X'd, X''d, Xl
                f.write("                      1.6400  0.2340  ")
                f.write("0.0000  4.0000  0.0000  0.0000  /\n\n")  # Xq, X'q, X''q, H, D, S(1.0)
                
                # SEXS - Simplified Excitation System
                f.write(f"{bus_num:6d} '{gen_id:2d}' 'SEXS' 1 ")
                f.write("200.00  5.0000 /\n")  # K, T
                f.write("                      1.0000 -1.0000  /\n\n")  # Emax, Emin
                
                # IEEEG1 - IEEE Type 1 Governor/Turbine
                f.write(f"{bus_num:6d} '{gen_id:2d}' 'IEEEG1' 1 ")
                f.write("0.0000  0.0500  0.2000 /\n")  # K, T1, T2
                f.write("                      0.0000  1.0000  0.0000  ")
                f.write("0.3000  0.2300  0.0000 /\n")  # T3, Uo, Uc, Pmax, Pmin, T4
                f.write("                      0.1000  1.0000  0.0000  ")
                f.write("0.2000  0.0000 /\n\n")  # K1, K2, T5, K3, K4
        
        print(f"DYR file created: {output_file}")
    
    def create_psse_simulation_script(self, output_file='run_psse_dynamics.py'):
        """
        Create a Python script to run PSS/E dynamic simulation
        
        Args:
            output_file: Output filename for Python script
        """
        print(f"\nCreating PSS/E simulation script: {output_file}")
        
        script_content = '''"""
PSS/E Dynamic Simulation Script
Automatically generated for IEEE 24-Bus system
"""
import os
import sys

# Add PSS/E to Python path (modify for your installation)
PSSE_PATH = r"C:\\Program Files\\PTI\\PSSE35\\PSSPY27"
sys.path.append(PSSE_PATH)
os.environ['PATH'] = PSSE_PATH + ';' + os.environ['PATH']

import psspy
import redirect

# Redirect PSS/E output
redirect.psse2py()

# Initialize PSS/E
psspy.psseinit(50000)

print("\\n" + "="*60)
print("PSS/E Dynamic Simulation - IEEE 24-Bus System")
print("="*60)

# Load the power flow case
print("\\n1. Loading power flow case...")
ierr = psspy.case('ieee24_psse.raw')
if ierr != 0:
    print(f"ERROR loading case file (error code: {ierr})")
    sys.exit(1)
print("   Case loaded successfully")

# Solve power flow
print("\\n2. Solving power flow...")
psspy.fnsl([0,0,0,1,1,0,99,0])
print("   Power flow converged")

# Convert generators and loads
print("\\n3. Converting to dynamic models...")
psspy.cong(0)  # Convert generators
psspy.conl(0,1,1,[0,0],[0.0,100.0,0.0,100.0])  # Convert constant power loads
print("   Conversion complete")

# Load dynamic models
print("\\n4. Loading dynamic model data...")
ierr = psspy.dyre_new([1,1,1,1],'ieee24_psse.dyr')
if ierr != 0:
    print(f"ERROR loading dyr file (error code: {ierr})")
    sys.exit(1)
print("   Dynamic models loaded")

# Initialize dynamics
print("\\n5. Initializing dynamic simulation...")
psspy.ordr(0)
psspy.fact()
psspy.tysl(0)
psspy.dyninit([1,1,1,1,1,1],[0.0,0.0])

# Set up channels for monitoring
psspy.delete_all_plot_channels()
psspy.chsb(0,1,[-1,-1,-1,1,1,0])  # Bus frequency channel

# Start simulation
print("\\n6. Running dynamic simulation...")
print("   Simulating 10 seconds...")

# Create output file
psspy.strt(0,'ieee24_dynamics.out')
psspy.run(0,1.0,1,1,0)  # Run to 1 second

# Apply fault at bus 1 (modify as needed)
print("\\n7. Applying 3-phase fault at bus 1...")
psspy.dist_bus_fault(1,1,0.0,[0.0,-0.2E+10])

psspy.run(0,1.1,1,1,0)  # Run during fault (0.1 sec)

# Clear fault
print("\\n8. Clearing fault...")
psspy.dist_clear_fault(1)

# Continue simulation
psspy.run(0,10.0,1,1,0)  # Run to 10 seconds

print("\\n" + "="*60)
print("Simulation completed successfully!")
print("Output file: ieee24_dynamics.out")
print("="*60)
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        print(f"Simulation script created: {output_file}")
        print("\nTo run the simulation:")
        print(f"  1. Modify PSSE_PATH in {output_file} for your installation")
        print(f"  2. Run: python {output_file}")

def main():
    # Path to your gridfm data
    data_path = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out\case24_ieee_rts\raw"
    
    # Create converter instance
    converter = GridFMtoPSSE(data_path)
    
    # Load GridFM data for scenario 0 (base case)
    converter.load_gridfm_data(scenario=0)
    
    # Generate PSS/E files
    converter.write_psse_raw('ieee24_psse.raw', version=35)
    converter.write_psse_dyr('ieee24_psse.dyr')
    converter.create_psse_simulation_script('run_psse_dynamics.py')
    
    print("\n" + "="*60)
    print("Conversion complete! Files created:")
    print("  - ieee24_psse.raw  (Power flow case)")
    print("  - ieee24_psse.dyr  (Dynamic models)")
    print("  - run_psse_dynamics.py  (Simulation script)")
    print("="*60)

if __name__ == "__main__":
    main()