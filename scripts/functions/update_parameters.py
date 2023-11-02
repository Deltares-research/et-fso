# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:49:59 2023

@author: imhof_rn

Iteratively adjust the staticmaps of all basins in a provided folder given a newly
provided 1D array of the parameter value that results from the FSO VAE. 
"""

import os
import numpy as np
import pathlib

import hydromt
from hydromt_wflow import WflowModel


# Adjust the wflow_sbm parameters with a given array of parameter values.
def update_parameters(
    basin_folder,
    pars_to_adjust,
    parameter_files,
    wflow_toml_par_names,
    case_name,
    starttime_run,
    endtime_run,
):
    """
    Main function to collect all information (model folders, parameter values, start
    and end times, and lists with parameters to adjust) and to adjust the wflow_sbm
    parameters with a given array of parameter values.

    Parameters
    ----------
    basin_folder : str
        Folder location of the folder containing all wflow_sbm catchment folders.
    pars_to_adjust : array-like
        List containing the wflow_sbm parameter names that should be updated.
    parameter_files : dict
        Dictionary containing the parameter names of pars_to_adjust with per
        parameter to adjust a filename to the geoTIFF file containing the new
        parameter map that should be added to the models.
    wflow_toml_par_names : array-like
        List containing the wflow_sbm internal parameter names that should be used in
        the toml file (e.g. pars_to_adjust "KsatHorFrac" = "input.lateral.subsurface.ksathorfrac")
        internally. This list will be used to update the toml file after the staticmaps
        have been adjusted.
    case_name : str
        Case name that is used as suffix (e.g. "1" --> KsatHorFrac_1) when storing
        the new parameter map in the staticmaps.nc of the model.
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
    # List all catchments
    catchment_list = os.listdir(basin_folder)

    # Loop through all catchments and adjust the parameter values
    for catchment_name in catchment_list:
        model_folder = os.path.join(basin_folder, catchment_name)
        # Open the wflow model
        print(f"Reading wflow model at {model_folder}")
        mod = WflowModel(model_folder, config_fn="wflow_sbm.toml", mode="r+")

        # Loop through all requested parameters and change the parameter valeus to
        # the provided values
        for var_name in pars_to_adjust:
            # Get the filename of the geoTIFF file for this parameter
            parameter_filename = parameter_files[var_name]

            # read (clipped by extent) KsatHorFrac with data_catalog.get_rasterdataset
            da_parameter = mod.data_catalog.get_rasterdataset(
                data_like=parameter_filename, geom=mod.staticgeoms["basins"]
            )

            # Tranform to a logarithmic scale if we are dealing with KsatVer or
            # KsatHorFrac
            if var_name == "KsatVer" or var_name == "KsatHorFrac":
                da_parameter.values = np.log10(da_parameter.values)

            # Resample the parameter values to the model grid resolution (upscaling)
            # with the grid_from_rasterdataset function
            resampled_parameter = hydromt.workflows.grid.grid_from_rasterdataset(
                grid_like=mod.staticmaps[var_name],
                ds=da_parameter,
                reproject_method=["average"],
                rename={
                    pathlib.Path(parameter_filename)
                    .stem: pathlib.Path(parameter_filename)
                    .stem[:-5]
                },
            )

            # Tranform the logarithm back in case we are dealing with the parameters
            # KsatVer or KsatHorFrac
            if var_name == "KsatVer" or var_name == "KsatHorFrac":
                resampled_parameter[var_name] = 10 ** resampled_parameter[var_name]

            # fill missing values
            filled_parameter = resampled_parameter
            if "lon" in filled_parameter._coord_names:
                filled_parameter[var_name] = filled_parameter[var_name].interpolate_na(
                    dim="lon", method="linear"
                )

                # fill NA's around the boundary
                filled_parameter[var_name] = filled_parameter[var_name].interpolate_na(
                    dim="lon", method="nearest", fill_value="extrapolate"
                )

            elif "longitude" in filled_parameter._coord_names:
                filled_parameter[var_name] = filled_parameter[var_name].interpolate_na(
                    dim="longitude", method="linear"
                )

                # fill NA's around the boundary
                filled_parameter[var_name] = filled_parameter[var_name].interpolate_na(
                    dim="longitude", method="nearest", fill_value="extrapolate"
                )
            else:
                filled_parameter[var_name] = filled_parameter[var_name].interpolate_na(
                    dim="x", method="linear"
                )

                # fill NA's around the boundary
                filled_parameter[var_name] = filled_parameter[var_name].interpolate_na(
                    dim="x", method="nearest", fill_value="extrapolate"
                )

            # add and write to staticmap
            mod.set_staticmaps(
                filled_parameter[var_name], name=var_name + "_" + case_name
            )

        # Edit model config with the provided start and end times.
        setting_toml = {
            "starttime": starttime_run,
            "endtime": endtime_run,
        }

        # Now also make the updated parameter(s) the standard for the simulation
        for i in range(len(pars_to_adjust)):
            setting_toml[wflow_toml_par_names[i]] = pars_to_adjust[i] + "_" + case_name

        # Loop through each setting defined in setting_toml and update
        # it in the model configuration
        for option in setting_toml:
            mod.set_config(option, setting_toml[option])

        # Write staticmaps and new TOML config
        mod.write_staticmaps()
        mod.write_config(config_name="wflow_sbm.toml")

    return None
