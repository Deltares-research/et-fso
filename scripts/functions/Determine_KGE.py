# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:37:20 2021

@author: imhof_rn

Determine the KGE given an observed and simulated discharge time series.
"""

import numpy as np
import pandas as pd


def modified_KGE_func(obs, mod):
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
    KGE : float
        The KGE score.

    """
    df = pd.DataFrame({'Qobs':obs, 'Qm':mod})
    df = df.dropna()
    r_daily = df.corr(method = 'pearson')
    r_daily = r_daily.loc["Qm","Qobs"]
    std_o = df.Qobs.std() 
    std_m = df.Qm.std()
    mean_o = df.Qobs.mean()
    mean_m= df.Qm.mean()

    alpha_daily = (std_m/mean_m) / (std_o/mean_o)
    beta_daily = mean_m / mean_o
    
    KGE = 1 - ((r_daily - 1)**2 + (alpha_daily - 1)**2 + (beta_daily - 1)**2)**0.5
    
    return KGE

def determine_KGE(Q_obs_file, Q_sim_file, starttime, endtime, Q_sim_variable_name):
    """
    

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
    KGE : float
        The KGE of the discharge simulation as floating point number.

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
    KGE = modified_KGE_func(np.array(Q_obs), np.array(Q_sim))

    return KGE