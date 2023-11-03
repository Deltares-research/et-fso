# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:49:59 2023

@author: imhof_rn

Iteratively adjust the toml file of all basins in a provided folder and run wflow. 
"""

import os
import subprocess
import time

import hydromt
from hydromt_wflow import WflowModel

start = time.time()


# -------------------------------------------------------------------------------- #
# Initial settings
# -------------------------------------------------------------------------------- #
# Set the input folder
input_folder = "c:/Users/imhof_rn/Documents/ET_FSO/UK_basins/Test"

# Give the ksathorfrac method that we are using
ksathorfrac_method = "KsatHorFrac_RF"

# Give the output name
output_csv_name = "benchmark_run/output_ksathorfrac_RF.csv"

# Set the start and end time of the run
startdate = "1970-01-01T00:00:00"
enddate = "2015-01-01T00:00:00" 

# Set the number of threads for the run
n_threads = 4


# -------------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------------- #
# Adjust the wflow_sbm parameters with a given array of parameter values.
def update_parameters(
        catchment_folder, 
        ksathorfrac_method, 
        output_csv_name,
        starttime_run, 
        endtime_run,
        ):
    """
    Function to adjust the toml file a wflow model

    Parameters
    ----------
    catchment_folder : str
        Folder location of the wflow_model.
    ksathorfrac_method : str
        The used KsatHorFrac value / approach for the toml file.
    output_csv_name : str
        The name of the output csv run file as registered in the toml file.
    starttime_run : str
        The starttime of the requested wflow_sbm run, formatted as: 
        %YYYY-%mm-%DDT%HH:%MM:%SS.
    endtime_run : str
        The end time of the requested wflow_sbm run, formatted as: 
        %YYYY-%mm-%DDT%HH:%MM:%SS.

    Returns
    -------
    None.

    """
    # Open the wflow model
    mod = WflowModel(catchment_folder, config_fn="wflow_sbm.toml", mode="r+")
    
    # Edit model config with the provided start and end times.
    setting_toml = {
        "input.lateral.subsurface.ksathorfrac": ksathorfrac_method,
        "csv.path": output_csv_name,
        "starttime": starttime_run,
        "endtime": endtime_run,
    }
    
    # Loop through each setting defined in setting_toml and update 
    # it in the model configuration
    for option in setting_toml:
        mod.set_config(option, setting_toml[option])

    # Write staticmaps and new TOML config
    mod.write_config(config_name="wflow_sbm.toml")

    return None


# The function to run wflow_sbm multi-threaded using Julia.
def run_wflow_sbm_julia(catchment_folder, n_threads=1):
    """
    Run the Julia code of wflow_sbm for the given set of basins. 

    Parameters
    ----------
    catchment_folder : str
        Folder location of the wflow_model.
    n_threads: int
        The number of threads used to run the wflow code. Default is 1.

    Returns
    -------
    None.

    """

    # Create the call argument for subprocess
    call_argument = [
        # "JULIA_EXCLUSIVE=1",
        "julia", 
        "-t", str(n_threads),
        "-e", "using Wflow; Wflow.run()",
        f"{catchment_folder}/wflow_sbm.toml"
        ]
    
    # Call the shell script to run wflow_sbm.jl
    subprocess.run(call_argument)
    
    return None
        

# -------------------------------------------------------------------------------- #
# The work
# -------------------------------------------------------------------------------- #  
# Loop through all basin folders
for catchment_folder in sorted(os.listdir(input_folder)):
    print(catchment_folder)
    
    # Check if output already exists. If yes, we'll continue to the next
    # catchment folder
    if os.path.exists(os.path.join(input_folder, catchment_folder, output_csv_name)):
        print("Result already exists, we'll continue")
    else:
        # First update the toml file for the run
        update_parameters(
                catchment_folder=os.path.join(input_folder,catchment_folder), 
                ksathorfrac_method=ksathorfrac_method, 
                output_csv_name=output_csv_name,
                starttime_run=startdate, 
                endtime_run=enddate,
                )
        
        # Then, run the model and store the results
        run_wflow_sbm_julia(catchment_folder=os.path.join(input_folder,catchment_folder), 
                            n_threads=n_threads)
    
   
end = time.time()
print("Done!")
print(f"Runtime is {(end - start)/3600} hours")
      