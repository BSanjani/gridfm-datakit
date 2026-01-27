"""
Dynamic Simulation: System Frequency Response (SFR)
Visualizes the time-domain response of the grid to a disturbance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("="*70)
print("SYSTEM DYNAMICS SIMULATION: FREQUENCY RESPONSE")
print("="*70)

# ============================================================================
# 1. SETUP PARAMETERS (derived from your System Data)
# ============================================================================

# Load your static data to get system size
try:
    gen_data = pd.read_parquet('data_out/case24_ieee_rts/raw/gen_data.parquet')
    total_capacity = gen_data['p_max_mw'].sum()
    print(f"System Total Capacity: {total_capacity:.2f} MW")
except:
    # Fallback if file not found
    print("Warning: Could not load parquet. Using default IEEE 24-bus values.")
    total_capacity = 3405.0 # MW (Approx IEEE 24 bus capacity)

# PHYSICS CONSTANTS (Typical IEEE values)
# H = Inertia Constant (seconds). How much kinetic energy is stored in rotors.
# D = Damping (p.u.). Natural load reduction as frequency drops.
# R = Droop (p.u.). 0.05 means 5% droop (standard).
# T_g = Governor Time Constant (seconds). How fast valves open.

H_sys = 4.0        # seconds (System Inertia)
D_sys = 1.0        # p.u. (Load Damping)
T_g_sys = 0.5      # seconds (Governor Lag - Steam/Hydro average)
R_sys = 0.05       # p.u. (5% Droop)

# DISTURBANCE DEFINITION
base_mva = 100.0
disturbance_mw = 200.0  # We lose 200 MW of generation suddenly
disturbance_pu = disturbance_mw / base_mva
sys_capacity_pu = total_capacity / base_mva

print(f"\nSimulation Parameters:")
print(f"  - Inertia (H): {H_sys} s")
print(f"  - Damping (D): {D_sys} p.u.")
print(f"  - Droop (R):   {R_sys*100}%")
print(f"  - Event:       Loss of {disturbance_mw} MW at t=1.0s")

# ============================================================================
# 2. DEFINE THE DIFFERENTIAL EQUATIONS
# ============================================================================

def system_dynamics(y, t, P_dist, H, D, R, T_g, with_droop=True):
    """
    State variables:
    y[0] = delta_f (Frequency deviation in p.u.)
    y[1] = delta_Pm (Mechanical Power deviation in p.u.)
    """
    delta_f = y[0]
    delta_Pm = y[1]
    
    # If Droop is DISABLED, R becomes infinite (1/R -> 0)
    droop_feedback = -(1/R) * delta_f if with_droop else 0
    
    # 1. Swing Equation (Newton's 2nd Law for Rotation)
    # 2H * d(df)/dt = P_mech - P_elec - D*df
    # P_elec change is the disturbance (load step)
    d_delta_f_dt = (1 / (2 * H)) * (delta_Pm - P_dist - (D * delta_f))
    
    # 2. Governor Equation (First order lag)
    # T_g * d(dPm)/dt = (Setpoint_change - P_mech)
    d_delta_Pm_dt = (1 / T_g) * (droop_feedback - delta_Pm)
    
    return [d_delta_f_dt, d_delta_Pm_dt]

# ============================================================================
# 3. RUN SIMULATION
# ============================================================================

# Time vector: 0 to 20 seconds
t = np.linspace(0, 20, 1000)
dist_start_time = 1.0

# Create step function for disturbance
P_disturbance = np.zeros_like(t)
P_disturbance[t >= dist_start_time] = disturbance_mw / base_mva # Convert to p.u.

# Initial conditions (everything is steady)
y0 = [0.0, 0.0]

print("\nRunning simulations...")

# -- Run WITH Droop --
sol_droop = odeint(
    lambda y, t_val: system_dynamics(
        y, t_val, 
        (disturbance_mw/base_mva if t_val >= dist_start_time else 0), 
        H_sys, D_sys, R_sys, T_g_sys, with_droop=True
    ), 
    y0, t
)

# -- Run WITHOUT Droop --
sol_no_droop = odeint(
    lambda y, t_val: system_dynamics(
        y, t_val, 
        (disturbance_mw/base_mva if t_val >= dist_start_time else 0), 
        H_sys, D_sys, R_sys, T_g_sys, with_droop=False
    ), 
    y0, t
)

# Convert to physical units
freq_base = 60.0 # Hz
f_droop = freq_base + (sol_droop[:, 0] * freq_base)
f_no_droop = freq_base + (sol_no_droop[:, 0] * freq_base)

p_mech_droop = sol_droop[:, 1] * base_mva
p_mech_no_droop = sol_no_droop[:, 1] * base_mva

# ============================================================================
# 4. VISUALIZATION
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# PLOT 1: FREQUENCY
ax1.plot(t, f_droop, 'g-', linewidth=2.5, label='With Droop Control')
ax1.plot(t, f_no_droop, 'b--', linewidth=2, label='Without Droop (Inertia only)')

# Highlight Nadir
nadir_idx = np.argmin(f_droop)
nadir_time = t[nadir_idx]
nadir_freq = f_droop[nadir_idx]
ax1.plot(nadir_time, nadir_freq, 'ro', label=f'Nadir: {nadir_freq:.2f} Hz')

ax1.set_ylabel('Frequency (Hz)', fontweight='bold')
ax1.set_title(f'System Frequency Response: Loss of {disturbance_mw} MW Generator', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.axvline(x=dist_start_time, color='k', linestyle=':', alpha=0.5)

# PLOT 2: MECHANICAL POWER RESPONSE
ax2.plot(t, p_mech_droop, 'g-', linewidth=2.5, label='Governor Response (Total)')
ax2.plot(t, p_mech_no_droop, 'b--', linewidth=2, label='No Governor Response')
ax2.axhline(y=disturbance_mw, color='r', linestyle=':', label='Total Lost Power')

ax2.set_ylabel('Added Generation (MW)', fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontweight='bold')
ax2.set_title('Aggregate Mechanical Power Response', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('system_dynamics_response.png', dpi=300)
print(f"\nSaved plot to: system_dynamics_response.png")
plt.show()

# ============================================================================
# 5. ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("POST-MORTEM ANALYSIS")
print("="*70)

# Steady state calculation
final_freq_droop = f_droop[-1]
final_freq_no_droop = f_no_droop[-1]

print(f"1. Frequency Nadir (Lowest Point):")
print(f"   With Droop:    {nadir_freq:.3f} Hz (Stopped the crash)")
print(f"   Without Droop: {np.min(f_no_droop):.3f} Hz (Kept falling/settled low)")

print(f"\n2. Steady State Frequency (t=20s):")
print(f"   With Droop:    {final_freq_droop:.3f} Hz")
print(f"   Without Droop: {final_freq_no_droop:.3f} Hz")

print(f"\n3. Why this happens:")
print(f"   - The 'Blue' line shows the grid relying ONLY on stored Kinetic Energy (Inertia).")
print(f"     It eventually stabilizes only because load naturally drops with frequency (Damping).")
print(f"   - The 'Green' line shows the Governors kicking in after t={T_g_sys}s.")
print(f"     They inject extra fuel/steam to arrest the fall and recover.")