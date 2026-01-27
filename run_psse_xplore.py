"""
PSS/E Dynamic Simulation Script for PSS/E Xplore
IEEE 24-Bus system
"""
import os
import sys

# Import PSS/E modules (already configured in Xplore)
import psspy

# Initialize PSS/E
psspy.psseinit(50000)

print("\n" + "="*60)
print("PSS/E Dynamic Simulation - IEEE 24-Bus System")
print("="*60)

# Change to the directory where the files are located
os.chdir(r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit')

# Load the power flow case
print("\n1. Loading power flow case...")
ierr = psspy.case('ieee24_psse.raw')
if ierr != 0:
    print(f"ERROR loading case file (error code: {ierr})")
    sys.exit(1)
print("   Case loaded successfully")

# Solve power flow
print("\n2. Solving power flow...")
ierr = psspy.fnsl([0,0,0,1,1,0,99,0])
if ierr == 0:
    print("   Power flow converged")
else:
    print(f"   WARNING: Power flow convergence issue (code: {ierr})")

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
ierr = psspy.dyninit([1,1,1,1,1,1],[0.0,0.0])
if ierr == 0:
    print("   Dynamic initialization successful")
else:
    print(f"   WARNING: Dynamic initialization issue (code: {ierr})")

# Set up channels for monitoring
print("\n6. Setting up monitoring channels...")
psspy.delete_all_plot_channels()
# Monitor bus 1 frequency
psspy.bus_frequency_channel([1, -1])
# Monitor generator angles and speeds
psspy.machine_array_channel([1, 1, 1], "1", "ANGLED")
psspy.machine_array_channel([1, 1, 1], "1", "SPEED")

# Start simulation
print("\n7. Running dynamic simulation...")
print("   Simulating pre-fault conditions (0 to 1.0 seconds)...")

# Create output file
psspy.strt(0,'ieee24_dynamics.out')
psspy.run(0, 1.0, 1, 1, 0)  # Run to 1 second

# Apply fault at bus 1
print("\n8. Applying 3-phase fault at bus 1 at t=1.0s...")
ierr = psspy.dist_bus_fault(1, 1, 0.0, [0.0, -0.2E+10])
if ierr == 0:
    print("   Fault applied successfully")

# Run during fault
print("   Running with fault (1.0 to 1.1 seconds)...")
psspy.run(0, 1.1, 1, 1, 0)  # Run during fault (0.1 sec)

# Clear fault
print("\n9. Clearing fault at t=1.1s...")
ierr = psspy.dist_clear_fault(1)
if ierr == 0:
    print("   Fault cleared successfully")

# Continue simulation
print("\n10. Running post-fault simulation (1.1 to 10.0 seconds)...")
psspy.run(0, 10.0, 1, 1, 0)  # Run to 10 seconds

print("\n" + "="*60)
print("Simulation completed successfully!")
print("Output file: ieee24_dynamics.out")
print("="*60)
print("\nYou can now plot the results using PSS/E's plotting tools")
print("or use dyntools to read the .out file")