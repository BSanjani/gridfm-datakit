"""
Simple Progress Monitor for GridFM Data Generation
===================================================
Watches the output directory and shows progress.

Run this in a SEPARATE terminal while GridFM is running.
"""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

def count_scenarios(data_dir):
    """Count how many scenarios have been generated"""
    data_path = Path(data_dir)
    
    # Try different possible paths
    search_paths = [
        data_path / 'raw' / 'bus_data.parquet',
        data_path / 'case24_ieee_rts' / 'raw' / 'bus_data.parquet',
    ]
    
    for parquet_file in search_paths:
        if parquet_file.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_file)
                n_scenarios = df['scenario'].nunique() if 'scenario' in df.columns else 0
                n_records = len(df)
                return n_scenarios, n_records, parquet_file
            except Exception as e:
                continue
    
    return 0, 0, None

def main():
    # Configuration
    data_dir = './data_out'
    expected_scenarios = 10000
    expected_topology_variants = 20
    total_expected = expected_scenarios * expected_topology_variants  # 200,000 solves!
    check_interval = 10  # Check every 10 seconds
    
    print("=" * 80)
    print("GRIDFM PROGRESS MONITOR")
    print("=" * 80)
    print(f"Start time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Monitoring dir:    {data_dir}")
    print(f"Base scenarios:    {expected_scenarios:,}")
    print(f"Topology variants: {expected_topology_variants}")
    print(f"Total PF solves:   {total_expected:,}")
    print(f"Check interval:    {check_interval}s")
    print("=" * 80)
    print("\nWaiting for data generation to start...")
    print("(If nothing appears, check that GridFM is running)\n")
    
    start_time = time.time()
    last_scenarios = 0
    last_time = start_time
    
    try:
        while True:
            current_scenarios, n_records, file_path = count_scenarios(data_dir)
            
            if current_scenarios > 0 or n_records > 0:
                # Calculate progress
                elapsed = time.time() - start_time
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                
                # Scenarios progress
                scenario_progress = (current_scenarios / total_expected) * 100 if total_expected > 0 else 0
                
                # Speed calculation
                time_since_last = time.time() - last_time
                scenarios_since_last = current_scenarios - last_scenarios
                
                if time_since_last > 0 and scenarios_since_last > 0:
                    speed = scenarios_since_last / time_since_last
                    remaining = total_expected - current_scenarios
                    eta_seconds = remaining / speed if speed > 0 else 0
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                else:
                    speed = 0
                    eta_str = "calculating..."
                
                # Print progress
                print(f"[{elapsed_str}] "
                      f"Scenarios: {current_scenarios:>6,}/{total_expected:,} ({scenario_progress:>5.1f}%) | "
                      f"Records: {n_records:>8,} | "
                      f"Speed: {speed:>5.1f} sc/s | "
                      f"ETA: {eta_str}")
                
                sys.stdout.flush()
                
                # Update tracking
                last_scenarios = current_scenarios
                last_time = time.time()
                
                # Check if complete
                if current_scenarios >= total_expected:
                    print("\n" + "=" * 80)
                    print("✓ GENERATION COMPLETE!")
                    print("=" * 80)
                    print(f"Total time: {elapsed_str}")
                    print(f"Final count: {current_scenarios:,} scenarios")
                    print(f"Total records: {n_records:,}")
                    print("=" * 80)
                    break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("MONITORING STOPPED")
        print("=" * 80)
        elapsed = time.time() - start_time
        print(f"Time: {str(timedelta(seconds=int(elapsed)))}")
        print(f"Last count: {current_scenarios:,} scenarios")
        print("=" * 80)

if __name__ == "__main__":
    main()