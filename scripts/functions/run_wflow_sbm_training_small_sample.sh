#! /bin/bash

module load 2022
module load Julia/1.8.2-linux-x86_64

# $1 = $(inputs.nr_threads)
# $2 = $(inputs.catchment_name_p1)
# $3 = $(inputs.catchment_name_p2)
# $4 = $(inputs.catchment_name_p3)
# $5 = $(inputs.catchment_name_p4)
# $6 = $(inputs.catchment_name_p5)


(julia -t "$1" ./functions/run_wflow_julia.jl "$2") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$3") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$4") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$5") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$6") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$7") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$8") &
(julia -t "$1" ./functions/run_wflow_julia.jl "$9") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${10}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${11}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${12}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${13}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${14}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${15}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${16}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${17}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${18}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${19}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${20}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${21}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${22}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${23}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${24}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${25}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${26}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${27}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${28}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${29}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${30}") &
(julia -t "$1" ./functions/run_wflow_julia.jl "${31}") &

wait
