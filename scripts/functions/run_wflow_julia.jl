using Wflow

# ARGS[1] = $(inputs.catchment_name_p)
println(ARGS[1]);
catchment_name_p = ARGS[1]

Wflow.run(string(catchment_name_p, "/wflow_sbm.toml"))