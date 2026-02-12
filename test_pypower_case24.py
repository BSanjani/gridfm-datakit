import numpy as np
import yaml
import sys
from pypower.api import case24_ieee_rts, ppoption, runpf

def load_config(path):
    """Safely loads the YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class PyPowerDroopSolver:
    """Solves for steady-state frequency equilibrium using droop control."""
    def __init__(self, ppc, config):
        self.ppc = ppc
        self.config = config
        # Get the first value from mp_range
        self.mp = config['droop_control']['mp_range'][0]
        self.db = config['droop_control']['frequency_deadband']
        # Convert string bus IDs to 0-indexed integers
        self.droop_buses = [int(b)-1 for b in config['droop_control']['droop_buses']]

    def solve(self):
        df = 0.0
        max_iter = 50
        tol = 1e-6
        
        print(f"Starting solver for 1.2x load (3420 MW)...")
        
        for i in range(max_iter):
            # Calculate effective df (deadband logic)
            df_eff = 0.0
            if abs(df) > self.db:
                df_eff = df - (self.db if df > 0 else -self.db)
            
            # Update generator outputs based on df_eff
            temp_ppc = self.ppc.copy()
            for bus_idx in self.droop_buses:
                # Find all generators on this bus
                gen_indices = np.where(temp_ppc['gen'][:, 0] == bus_idx + 1)[0]
                for g_idx in gen_indices:
                    # Droop logic: dP = -(1/mp) * df_eff
                    # Scaling by Pmax (column 8) to distribute load based on capacity
                    capacity_ratio = temp_ppc['gen'][g_idx, 8] / 100.0
                    temp_ppc['gen'][g_idx, 1] -= (1.0 / self.mp) * df_eff * capacity_ratio

            # Run standard Power Flow
            options = ppoption(VERBOSE=0, OUT_ALL=0)
            results = runpf(temp_ppc, options)
            
            if not results[0]['success']:
                return {'converged': False, 'iter': i}
            
            # Measure mismatch at the Slack Bus vs its original setpoint
            mismatch = results[0]['gen'][0, 1] - self.ppc['gen'][0, 1]
            
            if abs(mismatch) < tol:
                return {
                    'converged': True,
                    'df': df,
                    'freq_hz': 60.0 * (1.0 + df),
                    'iter': i
                }
            
            # Update frequency guess (Newton-like step)
            # A sensitivity of 15000 is a safe estimate for IEEE 24-bus
            df -= mismatch / 15000.0 
            
        return {'converged': False, 'iter': max_iter}

if __name__ == "__main__":
    try:
        # 1. Load config
        config = load_config("user_config_droop_1500.yaml")
        
        # 2. Load IEEE 24-bus system
        ppc = case24_ieee_rts()
        
        # 3. Apply Scenario 0 scaling (3420 MW / 2850 MW = 1.2x)
        total_p_scenario_0 = 3420.0
        base_load = 2850.0
        scaling = total_p_scenario_0 / base_load
        
        ppc['bus'][:, 2] *= scaling # Active Power
        ppc['bus'][:, 3] *= scaling # Reactive Power
        
        # 4. Initialize and solve
        solver = PyPowerDroopSolver(ppc, config)
        results = solver.solve()
        
        # 5. Output results
        print("\n" + "="*30)
        if results['converged']:
            print(f"✅ SCENARIO 0 CONVERGED")
            print(f"Iterations:  {results['iter']}")
            print(f"Total Load:  {total_p_scenario_0} MW")
            print(f"Frequency:   {results['freq_hz']:.4f} Hz")
            print(f"Deviation:   {results['df']:.6f} p.u.")
        else:
            print(f"❌ SCENARIO 0 FAILED TO CONVERGE")
            print(f"The system reached instability at {total_p_scenario_0} MW.")
        print("="*30)

    except Exception as e:
        print(f"An error occurred: {e}")