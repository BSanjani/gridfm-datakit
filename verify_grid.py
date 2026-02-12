import numpy as np
import yaml
from pypower.api import case24_ieee_rts, ppoption, runpf

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class PyPowerDroopSolver:
    def __init__(self, ppc, config):
        self.ppc = ppc
        self.config = config
        self.mp = config['droop_control']['mp_range'][0]
        self.db = config['droop_control']['frequency_deadband']
        # Bus IDs to indices
        self.droop_buses = [int(b)-1 for b in config['droop_control']['droop_buses']]

    def solve(self):
        df = -0.001  # Start with a tiny deviation to kick the solver
        max_iter = 100
        tol = 1e-4
        
        print(f"{'Iter':<5} | {'Freq (Hz)':<10} | {'df':<12} | {'Mismatch (MW)':<15}")
        print("-" * 50)

        for i in range(max_iter):
            # 1. Effective df
            df_eff = 0.0
            if abs(df) > self.db:
                df_eff = df - (self.db if df > 0 else -self.db)
            
            # 2. Update Gen P-setpoints based on droop
            temp_ppc = self.ppc.copy()
            for bus_idx in self.droop_buses:
                gen_idx = np.where(temp_ppc['gen'][:, 0] == bus_idx + 1)[0]
                for g in gen_idx:
                    # Power increases as frequency drops
                    capacity_factor = temp_ppc['gen'][g, 8] / 100.0
                    temp_ppc['gen'][g, 1] -= (1.0 / self.mp) * df_eff * capacity_factor

            # 3. Run PF
            res = runpf(temp_ppc, ppoption(VERBOSE=0, OUT_ALL=0))
            if not res[0]['success']:
                print("Power flow failed to converge.")
                return {'converged': False}

            # 4. Measure Slack Error
            # We want the slack bus (Gen 0) to also obey droop. 
            slack_out = res[0]['gen'][0, 1]
            slack_target = self.ppc['gen'][0, 1] - (1.0 / self.mp) * df_eff * (self.ppc['gen'][0, 8] / 100.0)
            mismatch = slack_out - slack_target

            print(f"{i+1:<5} | {60*(1+df):<10.4f} | {df:<12.6f} | {mismatch:<15.4f}")

            if abs(mismatch) < tol:
                return {'converged': True, 'df': df, 'freq_hz': 60*(1+df), 'iter': i+1}

            # 5. Feedback: if mismatch > 0 (excess power), frequency must drop further
            df -= mismatch / 20000.0 

        return {'converged': False}

if __name__ == "__main__":
    config = load_config("user_config_droop_1500.yaml")
    ppc = case24_ieee_rts()
    
    # Force 1.2x Load
    scaling = 3420.0 / 2850.0
    ppc['bus'][:, 2] *= scaling
    ppc['bus'][:, 3] *= scaling
    
    solver = PyPowerDroopSolver(ppc, config)
    results = solver.solve()
    
    if results['converged']:
        print(f"\nFinal Result: {results['freq_hz']:.4f} Hz in {results['iter']} iterations.")