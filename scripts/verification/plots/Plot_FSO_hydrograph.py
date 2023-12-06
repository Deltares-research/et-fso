# -*- coding: utf-8 -*-
"""
@author: imhof_rn

Plot the hydrographs of the FSO runs for inspection
"""

import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -------------------------------------------------------------------------------- #
# The initial settings
# -------------------------------------------------------------------------------- #
# Define the input folder where the wflow_sbm models and observations are stored
input_folder = "p:/11209205-034-et-fso/UK_basins/Test"

# Define the input folder where the observations are stored
input_folder_obs = "p:/11209205-034-et-fso/UK_basins/Obs"

# Define the list of methods to loop through
method = "Test2_30trainingbasins"

# Set the start and end time of the validation period
start_time_test_validation = "1972-01-01 00:00:00"
end_time_test_validation = "2009-12-31 00:00:00"

# Define the output csv file name where the results will be stored
output_folder = "p:/11209205-034-et-fso/Verification/Figs/Hydrographs/Test2_30trainingbasins"


# -------------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------------- #
def slice_data(Q_obs_file, Q_sim_file, starttime, endtime, Q_sim_variable_name):
    """
    Slices the observations and simulations for a given starttime and endtime, given 
    an observed and simulated discharge timeseries csv file.

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
    Q_obs : array-like
        The sliced observations.
    Q_sim : array-like
        The sliced simulations.

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

    return Q_obs, Q_sim


# -------------------------------------------------------------------------------- #
# The work - loop through all model results, open the observations and simulations
# and plot them
# -------------------------------------------------------------------------------- #
# Get the list of basins from the input folder
basins = os.listdir(input_folder)

# Now, loop through all basins and methods
for basin in basins:
    print(f"We are at basin: {basin}")
    # Get the file with observations
    Q_obs_file = os.path.join(
        input_folder_obs, f"CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv"
        )
    # Get the simulation file
    Q_sim_file = os.path.join(input_folder, basin, method, "output.csv")

    # Open and slice the observations and simulations
    Q_obs_temp, Q_sim_temp = slice_data(
        Q_obs_file=Q_obs_file, 
        Q_sim_file=Q_sim_file, 
        starttime=start_time_test_validation, 
        endtime=end_time_test_validation, 
        Q_sim_variable_name="Q_1",
        )

    # Also make a pandas data range of the dates
    date_range = pd.date_range(
        start=start_time_test_validation,
        end=end_time_test_validation,
        freq="1d",
        )

    ###
    # Plot the results
    ###
    # Set the figure outline
    fig = plt.figure(figsize = (16,12))
    fig1 = fig.add_gridspec(ncols = 1, nrows = 3, bottom = 0.0, top = 1.0, wspace=0.1)
    ax1 = fig.add_subplot(fig1[0])
    ax2 = fig.add_subplot(fig1[1])
    ax3 = fig.add_subplot(fig1[2])

    axs = [ax1,ax2,ax3]

    # Plot the entire time series over three different graphs (to spread the
    # information amount a bit)
    slice_len = int(len(Q_obs_temp)/3)
    index = 0

    for ax in axs:
        index_prev = index
        index += slice_len
        
        ax.plot(date_range[index_prev:index], Q_obs_temp[index_prev:index], color="dimgrey", lw=2)
        ax.plot(date_range[index_prev:index], Q_sim_temp[index_prev:index], color="deepskyblue", lw=2)

        # Set the x- and y-lim
        # ax.set_ylim(0,90.0)
        # ax.set_xlim(dates[0], dates[-1])

        # Set the xticks and yticks
        # if axs_Q[i] == ax8:
        #     axs_Q[i].set_xticks([dates[0],dates[12],dates[24],dates[36],dates[48],dates[60],dates[72],dates[84],dates[96],dates[108],dates[120],dates[132],dates[144]])
        #     axs_Q[i].set_xticklabels(["0", " ", "2", " ", "4", " ", "6", " ", "8", " ", "10", " ", "12"], fontsize = 18)
        # else:
        #     axs_Q[i].set_xticks([dates[0],dates[12],dates[24],dates[36],dates[48],dates[60],dates[72],dates[84],dates[96],dates[108],dates[120],dates[132],dates[144]])
        #     axs_Q[i].set_xticklabels([" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "], fontsize = 18)
        
        ax.tick_params(axis='x', which='major', labelsize=15)
        
        # ax.set_yticks([0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        # ax.set_yticklabels(["0", " ", "20", " ", "40", " ", "60", " ", "80", " "], fontsize = 15)        

    # Set the axis labels      
    ax2.set_ylabel(r"Discharge (m$^{3}$ s$^{-1}$)", fontsize = 17)

    # Set the legend
    legend_elements_cmls = [Line2D([0], [0], color='dimgrey', label=r"$Q_\mathrm{obs}$", linewidth = 3),
                            Line2D([0], [0], color='deepskyblue', label=r"$Q_\mathrm{sim}$", linewidth = 3),
                            ]

    ax1.legend(handles=legend_elements_cmls, loc='upper right', frameon = False, fontsize = 15, ncol = 1)

    # Save it
    outfile = os.path.join(output_folder, f"Hydrograph_{basin}.png")
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    # plt.show()
