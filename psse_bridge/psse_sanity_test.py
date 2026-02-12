# psse_bridge/psse_sanity_test.py

from psse_bridge.psse_env import init_psse

psspy = init_psse(psse_version=35)

# Load a built-in test case
psspy.case("ieee14.sav")

# Run power flow
psspy.fnsl([0, 0, 0, 1, 1, 0, 99, 0])

# Read one bus voltage
ierr, vpu = psspy.busdat(1, 'PU')

print("Bus 1 voltage (pu):", vpu)
