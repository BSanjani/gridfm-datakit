using PowerModels
using JuMP
using Ipopt
using JSON

# Load case9
network_data = PowerModels.parse_file("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/gridfm_datakit/cases/case9.m")

# Configure droop - MUST match PyPower exactly!
droop_config = Dict(
    "enabled" => true,
    "mp" => 0.05,
    "mq" => 0.05,
    "V_0" => 1.0,
    "frequency_deadband" => 0.0006,
    "voltage_deadband" => 0.0,
    "droop_buses" => ["1", "2", "3"]
)

network_data["droop_config"] = droop_config

# Run your droop solver (adjust function name if needed)
result = solve_ac_pf_unified(network_data, droop_config)

# Save results
output = Dict(
    "converged" => (result["termination_status"] in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]),
    "df" => get(result, "df", 0.0),
    "bus" => Dict(string(i) => Dict("vm" => b["vm"], "va" => b["va"]) 
                  for (i,b) in result["solution"]["bus"]),
    "gen" => Dict(string(i) => Dict("pg" => g["pg"], "qg" => g["qg"]) 
                  for (i,g) in result["solution"]["gen"])
)

open("julia_result_case9.json", "w") do f
    JSON.print(f, output, 2)
end

println("✓ Saved to julia_result_case9.json")
println("  df = ", output["df"])