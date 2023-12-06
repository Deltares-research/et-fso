# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:37:20 2021

@author: imhof_rn

Determine the KGE given an observed and simulated discharge time series for a folder
with wflow_sbm models and a provided folder with observations for these basins. This
scripts also loops through all provided methods and determines the KGE per method and
basin. It stores a large csv with per basin number the KGE per method.
"""

import numpy as np
import os
import pandas as pd


# -------------------------------------------------------------------------------- #
# The initial settings
# -------------------------------------------------------------------------------- #
# Define the input folder where the wflow_sbm models and observations are stored
input_folder = "p:/11209205-034-et-fso/UK_basins/Test"

# Define the input folder where the observations are stored
input_folder_obs = "p:/11209205-034-et-fso/UK_basins/Obs"

# Define the list of methods to loop through
methods = ["ksathorfrac100","ksathorfrac_AXA","ksathorfrac_RF","ksathorfrac_BRT","Test2_30trainingbasins"]
# Give the folder name of the test run
folder_test = "Test2_30trainingbasins"

# Set the start and end time of the validation period
start_time_test_validation = "1972-01-01 00:00:00"
end_time_test_validation = "2009-12-31 00:00:00"

# Define the output csv file name where the results will be stored
output_csv_filename = "p:/11209205-034-et-fso/Verification/Results_Benchmark_Methods_Optimization2_30trainingbasins.csv"


# -------------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------------- #
# Define a function to calculate the Kling-Gupta Efficiency (KGE) between two time series
def kge_original_func(obs, sim):
    """
    Calculates the original KGE metric, given an observed
    and simulated time series.

    Parameters
    ----------
    obs : array-like
        The observations in m3/s.
    mod : array-like
        The simulated discharge in m3/s.

    Returns
    -------
    kge_score : float
        The KGE score.

    """
    # Calculate the mean, standard deviation and pearson correlation of the observed 
    # and simulated values
    df = pd.DataFrame({'Qobs':obs, 'Qm':sim})
    df = df.dropna()
    pearson_r = df.corr(method = 'pearson')
    pearson_r = pearson_r.loc["Qm","Qobs"]
    std_o = df.Qobs.std() 
    std_m = df.Qm.std()
    mean_o = df.Qobs.mean()
    mean_m= df.Qm.mean()

    # Calculate the KGE score using the formula
    kge_score = 1 - np.sqrt(
        (pearson_r - 1) ** 2 + (std_m / std_o - 1) ** 2 + (mean_m / mean_o - 1) ** 2
        )
    
    # Return the KGE score
    return kge_score


def modified_kge_func(obs, mod):
    """
    Calculates the modified KGE (Kling et al., 2012) metric, given an observed
    and simulated time series.

    Parameters
    ----------
    obs : array-like
        The observations in m3/s.
    mod : array-like
        The simulated discharge in m3/s.

    Returns
    -------
    kge_score : float
        The KGE score.

    """
    # Calculate the mean, standard deviation and pearson correlation of the observed 
    # and simulated values
    df = pd.DataFrame({'Qobs':obs, 'Qm':mod})
    df = df.dropna()
    pearson_r = df.corr(method = 'pearson')
    pearson_r = pearson_r.loc["Qm","Qobs"]
    std_o = df.Qobs.std() 
    std_m = df.Qm.std()
    mean_o = df.Qobs.mean()
    mean_m= df.Qm.mean()
    
    # Determine the alpha and beta parameters of the KGE
    alpha_daily = (std_m/mean_m) / (std_o/mean_o)
    beta_daily = mean_m / mean_o

    # Calculate the KGE score using the formula
    kge_score = 1 - ((pearson_r - 1)**2 + (alpha_daily - 1)**2 + (beta_daily - 1)**2)**0.5
    
    # Return the KGE score
    return kge_score


def determine_kge(Q_obs_file, Q_sim_file, starttime, endtime, Q_sim_variable_name):
    """
    Determines the kge for a given starttime and endtime given an observed and 
    simulated discharge timeseries csv file. Both the original and modified KGE 
    scores are calculated and returned.

    Parameters
    ----------
    Q_obs_file : str
        Location of the observed discharge time series.
    Q_sim_file : str
        Location of the simulated discharge time series.
    starttime : str
        Date string giving the start time that will be used for the time series 
        (in "YYYY-mm-dd hh:MM:SS").
    endtime : str
        Date string giving the end time that will be used for the time series 
        (in "YYYY-mm-dd hh:MM:SS").
    Q_sim_variable_name : str
        The discharge variable name for the selected basin in the simulation
        file.

    Returns
    -------
    kge_original : float
        The KGE of the discharge simulation as floating point number.
    kge_modified : float
        The modified KGE of the discharge simulation as floating point number.

    """

    # Open the observed and simulated discharge and only keep the values within the
    # date range
    df_obs = pd.read_csv(Q_obs_file, sep=",")
    df_sim = pd.read_csv(Q_sim_file, sep=",")
    
    # Only keep the discharge in between the given dates
    df_obs = df_obs.set_index(pd.to_datetime(df_obs["date"]))
    Q_obs = df_obs[starttime:endtime]["discharge_vol"] 
    # Get the simulated discharge
    df_sim = df_sim.set_index(pd.to_datetime(df_sim["time"]))
    Q_sim = df_sim[starttime:endtime][Q_sim_variable_name]
    
    # Finally, determine the KGE
    kge_original = kge_original_func(np.array(Q_obs), np.array(Q_sim))
    kge_modified = modified_kge_func(np.array(Q_obs), np.array(Q_sim))

    return kge_original, kge_modified


# -------------------------------------------------------------------------------- #
# The work - loop through all model results, calculate and store the KGE
# -------------------------------------------------------------------------------- #
# Get the list of basins from the input folder
basins = os.listdir(input_folder)

# Make a list of methods and the two kge_methods used:
# [KsatHorFrac100_kge_orig, KsatHorFrac100_kge_modified, KsatHorFrac_AXA_kge_orig, etc.]
columns = []
for method in methods:
    columns.append(f"{method}_kge_orig")
    columns.append(f"{method}_kge_modified")
# Create an empty dataframe with columns: basin, method_kge_orig, method_kge_modified, etc.
df_kge = pd.DataFrame(columns=["basin"] + columns)

# Set the number of rows as the number of basins
df_kge["basin"] = basins

# Now, loop through all basins and methods
for basin in basins:
    print(f"We are at basin: {basin}")
    # Get the file with observations
    Q_obs_file = os.path.join(
        input_folder_obs, f"CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv"
        )
    # Now, loop through all methods and determine the KGE values
    for method in methods:
        try:
            if method == folder_test:
                Q_sim_file = os.path.join(input_folder, basin, folder_test, "output.csv")
            else:
                Q_sim_file = os.path.join(input_folder, basin, "benchmark_run", f"output_{method}.csv")
            # Determine the KGE
            kge_original, kge_modified = determine_kge(
                Q_obs_file=Q_obs_file, 
                Q_sim_file=Q_sim_file, 
                starttime=start_time_test_validation, 
                endtime=end_time_test_validation, 
                Q_sim_variable_name="Q_1",
                )
            # Add the result to the dataframe at the right method and kge method column
            df_kge.loc[df_kge["basin"] == basin, f"{method}_kge_orig"] = kge_original
            df_kge.loc[df_kge["basin"] == basin, f"{method}_kge_modified"] = kge_modified
        except (FileNotFoundError, ZeroDivisionError, ValueError):
            print(f"{basin} has no results")


# -------------------------------------------------------------------------------- #
# Write the results to the output csv file
# -------------------------------------------------------------------------------- #
print("Store the results in a csv file")
# Set the basin number as the index of the dataframe
df_kge = df_kge.set_index("basin")
# Store it
df_kge.to_csv(output_csv_filename)