#! /bin/bash

module load 2022
module load Julia/1.8.2-linux-x86_64

# $1 = $(inputs.nr_threads)
# $2 = $(inputs.catchment_name_p1)
# $3 = $(inputs.catchment_name_p2)
# $4 = $(inputs.catchment_name_p3)
# $5 = $(inputs.catchment_name_p4)
# $6 = $(inputs.catchment_name_p5)
# $7 = $(inputs.catchment_name_p6)
# $8 = $(inputs.catchment_name_p7)
# $9 = $(inputs.catchment_name_p8)

(julia -t "$1" ./Functions/run_wflow_julia.jl "$2") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$3") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$4") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$5") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$6") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$7") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$8") &
(julia -t "$1" ./Functions/run_wflow_julia.jl "$9") &
wait
