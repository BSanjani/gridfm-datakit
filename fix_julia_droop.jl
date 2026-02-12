# Quick diagnostic - check if df is actually being saved
using PowerModels
using DataFrames
using Parquet

# Load one scenario's runtime data
runtime = DataFrame(read_parquet("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/data_out_ieee24_droop_WithDeadband_10k/case24_ieee_rts/raw/runtime_data.parquet"))

println("Runtime data columns: ", names(runtime))
println("\nFirst 10 rows:")
println(first(runtime, 10))

println("\nFrequency deviation stats:")
println("  Min: ", minimum(runtime.frequency_deviation))
println("  Max: ", maximum(runtime.frequency_deviation))
println("  Mean: ", mean(runtime.frequency_deviation))
println("  Unique values: ", length(unique(runtime.frequency_deviation)))