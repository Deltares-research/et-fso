# -*- coding: utf-8 -*-
"""
@author: imhof_rn

Move the test result output from Snellius to the correct folder structure
on the P-drive
"""

import os
import shutil


# ----------------------------------------------------------------------- #
# Initial settings
# ----------------------------------------------------------------------- #
# Provide the location with the results from Snellius
input_folder = "p:/11209205-034-et-fso/Results/Optimization2_30trainingbasins/Test_runs"

# Give the output folder where all (benchmark) runs are stored
output_folder = "p:/11209205-034-et-fso/UK_basins/Test"

# Give the run a name
run_name = "Test2_30trainingbasins"


# ----------------------------------------------------------------------- #
# The work
# ----------------------------------------------------------------------- #
# Loop through the input folder and copy all results to the correct folder
# structure in the output folder
for basin_name in os.listdir(input_folder):
    # Get the run file
    run_filename = os.path.join(
        input_folder, basin_name, "run_default", "output.csv"
        )
    # First create the run folder in the output folder
    if not os.path.isdir(os.path.join(output_folder, basin_name, run_name)):
        os.mkdir(os.path.join(output_folder, basin_name, run_name))
    # Then, copy the result to the new folder
    shutil.copy2(
        run_filename, 
        os.path.join(output_folder, basin_name, run_name, "output.csv")
        )

print("Done!")