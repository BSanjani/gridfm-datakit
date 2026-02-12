"""
PSS/E Dynamic Simulation Script
Automatically generated for IEEE 24-Bus system
"""
import os
import sys

# Add PSS/E to Python path (modify for your installation)
# PSSE_PATH = r"C:\Program Files\PTI\PSSE35\PSSPY27"
PSSE_PATH = r"C:\Program Files\PTI\PSSE36\36.4\PSSPY314"
sys.path.append(PSSE_PATH)
os.environ['PATH'] = PSSE_PATH + ';' + os.environ['PATH']

import psspy
import redirect

# Redirect PSS/E output
redirect.psse2py()

# Initialize PSS/E
psspy.psseinit(50000)

print("\n" + "="*60)
print("PSS/E Dynamic Simulation - IEEE 24-Bus System")
print("="*60)

# Load the power flow case
print("\n1. Loading power flow case...")
ierr = psspy.case('ieee24_psse.raw')
if ierr != 0:
    print(f"ERROR loading case file (error code: {ierr})")
    sys.exit(1)
print("   Case loaded successfully")

# Solve power flow
print("\n2. Solving power flow...")
psspy.fnsl([0,0,0,1,1,0,99,0])
print("   Power flow converged")

# Convert generators and loads
print("\n3. Converting to dynamic models...")
psspy.cong(0)  # Convert generators
psspy.conl(0,1,1,[0,0],[0.0,100.0,0.0,100.0])  # Convert constant power loads
print("   Conversion complete")

# Load dynamic models
print("\n4. Loading dynamic model data...")
ierr = psspy.dyre_new([1,1,1,1],'ieee24_psse.dyr')
if ierr != 0:
    print(f"ERROR loading dyr file (error code: {ierr})")
    sys.exit(1)
print("   Dynamic models loaded")

# Initialize dynamics
print("\n5. Initializing dynamic simulation...")
psspy.ordr(0)
psspy.fact()
psspy.tysl(0)
psspy.dyninit([1,1,1,1,1,1],[0.0,0.0])

# Set up channels for monitoring
psspy.delete_all_plot_channels()
psspy.chsb(0,1,[-1,-1,-1,1,1,0])  # Bus frequency channel

# Start simulation
print("\n6. Running dynamic simulation...")
print("   Simulating 10 seconds...")

# Create output file
psspy.strt(0,'ieee24_dynamics.out')
psspy.run(0,1.0,1,1,0)  # Run to 1 second

# Apply fault at bus 1 (modify as needed)
print("\n7. Applying 3-phase fault at bus 1...")
psspy.dist_bus_fault(1,1,0.0,[0.0,-0.2E+10])

psspy.run(0,1.1,1,1,0)  # Run during fault (0.1 sec)

# Clear fault
print("\n8. Clearing fault...")
psspy.dist_clear_fault(1)

# Continue simulation
psspy.run(0,10.0,1,1,0)  # Run to 10 seconds

print("\n" + "="*60)
print("Simulation completed successfully!")
print("Output file: ieee24_dynamics.out")
print("="*60)
