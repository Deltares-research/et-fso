# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 06:56:14 2023

@author: imhof_rn

Script to run a shell script that calls Julia to run wflow_sbm multi-threaded with
multiple models on one node. Depending on whether the training or test set is called,
a different number of model locations (the arguments to the shell) script will be 
provided, as this is based on the integer by which the total number of basins in the set can
be divided (5 models per run for test, with 115 basins that makes 23 runs | 8 models
per run for training, with 344 basins that makes 43 runs). 
"""

import os
import subprocess


# The function to run wflow_sbm multi-threaded using Julia. This function calls, per
# set of basin a shell script that runs the models for the basins.
# TODO: make this more flexible for different number of basins and requested cores
# per basin
def run_wflow_sbm_julia(basin_folder, run_type):
    """
    Run the Julia code of wflow_sbm for the given set of basins.

    Parameters
    ----------
    basin_folder : str
        Folder location of the folder containing all wflow_sbm catchment folders.
    run_type : str
        String indicating 'training' or 'test'.Depending on whether the training or
        test set is called, a different number of models will be provided to the
        shell script, as this is based on the integer by which the total number of
        basins in the set can be divided (5 models per run for test, with 115 basins
        that makes 23 runs | 8 models per run for training, with 344 basins that
        makes 43 runs).

    Returns
    -------
    None.

    """
    # The shell script for the training set
    if run_type == "training":
        # The run settings
        nr_cores = 5
        nr_models_per_run = 8
        # Get the cathment names
        catchment_names = os.listdir(basin_folder)
        # Split the basin folder in lists with a length of "nr_models_per_run"
        for i in range(0, len(catchment_names), nr_models_per_run):
            catchments_run = catchment_names[i : (i + nr_models_per_run)]
            # Add the path to the catchment names
            folders_with_path = [
                os.path.join(basin_folder, catchment_run)
                for catchment_run in catchments_run
            ]
            # Create the call argument for subprocess
            call_argument = [
                # TODO: Change for Linux to sh
                "bash",
                "./functions/run_wflow_sbm_training_small_sample.sh",
                str(nr_cores),
            ]
            # Add the folder path to the call argument
            call_argument = call_argument + folders_with_path
            # Call the shell script to run wflow_sbm.jl
            subprocess.run(call_argument)

    # The shell script for the test set
    if run_type == "test":
        # The run settings
        nr_cores = 12
        nr_models_per_run = 10
        # Get the cathment names
        catchment_names = os.listdir(basin_folder)
        # Split the basin folder in lists with a length of "nr_models_per_run"
        for i in range(0, len(catchment_names), nr_models_per_run):
            catchments_run = catchment_names[i : (i + nr_models_per_run)]
            # Add the path to the catchment names
            folders_with_path = [
                os.path.join(basin_folder, catchment_run)
                for catchment_run in catchments_run
            ]
            # Create the call argument for subprocess
            call_argument = ["sh", "./Functions/run_wflow_sbm_test.sh", nr_cores]
            # Add the folder path to the call argument
            call_argument = call_argument + folders_with_path
            # Call the shell script to run wflow_sbm.jl
            subprocess.run(call_argument)

    return None
