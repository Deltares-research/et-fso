import os
import numpy as np
import pathlib
import pandas as pd

from functions import FSO_functions
from functions import FSO_VAE_generator
from functions import update_parameters
from functions import (
    run_wflow_sbm_multi_threaded_small_sample as run_wflow_sbm_multi_threaded,
)
from functions import Determine_KGE

from typing import Union


def add_to_dataframe(result_tracker_df, column_dict):
    result_tracker_df.loc[len(result_tracker_df)] = column_dict

    return result_tracker_df


counter = 0
path = pathlib.Path(__file__).parent.parent.parent.resolve()

example_tif = path / "Data/spatial_predictors/BD/VK_BD_wavg_250m.tif"


def objective_function(
    point: np.array,
    test_number: int,
    spatial_predictors: pd.DataFrame,
    parameter_names: list,
    parameter_bounds: pd.DataFrame,
    result_tracker_df: pd.DataFrame,
    counter: int,
    run_folder: Union[str, pathlib.Path],
    training_basin_folder_name: Union[str, pathlib.Path],
    test_basin_folder_name: Union[str, pathlib.Path],
    Q_obs_folder: Union[str, pathlib.Path],
    Q_sim_variable_names: str,
    start_time_test_run: str,
    end_time_test_run: str,
    start_time_test_validation: str,
    end_time_test_validation: str,
    start_time_training_run: str,
    end_time_training_run: str,
    start_time_training_validation: str,
    end_time_training_validation: str,
    pars_to_adjust,
    wflow_toml_par_names,
):
    # Calculate the transfer function of the point
    point_tf = pd.DataFrame()
    for p in range(len(parameter_names)):
        point_tf[parameter_names[p]] = FSO_VAE_generator.tf_generator(
            np.expand_dims(point[6 * p : (6 + 6 * p)], axis=0)
        )

    print(f"x1 = {point_tf[parameter_names[0]].loc[0]} \n")

    # Check if a spatial predictor is always in any function
    function_splitted = FSO_functions.function_splitter(
        point_tf[parameter_names[0]].loc[0]
    )
    spatial_predictor_check = False
    for names in spatial_predictors.columns:
        if names in function_splitted:
            spatial_predictor_check = True

    if not spatial_predictor_check:
        print("No valid function found!\nNo spatial predictors in function \n")
        result_tracker_df = add_to_dataframe(
            result_tracker_df,
            {
                "best_x1": result_tracker_df["best_x1"].loc[len(result_tracker_df) - 1],
                "full_loss": result_tracker_df["full_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "KGE": result_tracker_df["KGE"].loc[len(result_tracker_df) - 1],
                "wKGE": result_tracker_df["wKGE"].loc[len(result_tracker_df) - 1],
                "model_loss": result_tracker_df["model_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "x1": point_tf[parameter_names[0]].loc[0],
                "KGE_run": -9999.0,
                "wKGE_run": -9999.0,
                "loss_run": -9999.0,
                "mean_par_value": -9999.0,
                "n_iterations_used": 0,
                "n_iterations_since_BF_change": result_tracker_df[
                    "n_iterations_since_BF_change"
                ].loc[len(result_tracker_df) - 1]
                + 1,
                "stringsAsFactors": False,
            },
        )

        return (
            pd.DataFrame({"loss": np.nan, "Current point in function space": [point]}),
            result_tracker_df,
        )

    model_size_loss = FSO_functions.size_loss(function_splitted)

    try:
        new_wflow_para = FSO_functions.create_wflow_para(
            transfer_functions=point_tf,
            l0=spatial_predictors,
            parameter_bounds=parameter_bounds,
            parameter_names=parameter_names,
        )
    except:
        print("No valid function found!\n Problem creating new_wflow_para \n")
        result_tracker_df = add_to_dataframe(
            result_tracker_df,
            {
                "best_x1": result_tracker_df["best_x1"].loc[len(result_tracker_df) - 1],
                "full_loss": result_tracker_df["full_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "KGE": result_tracker_df["KGE"].loc[len(result_tracker_df) - 1],
                "wKGE": result_tracker_df["wKGE"].loc[len(result_tracker_df) - 1],
                "model_loss": result_tracker_df["model_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "x1": point_tf[parameter_names[0]].loc[0],
                "KGE_run": -9999.0,
                "wKGE_run": -9999.0,
                "loss_run": -9999.0,
                "mean_par_value": -9999.0,
                "n_iterations_used": 0,
                "n_iterations_since_BF_change": result_tracker_df[
                    "n_iterations_since_BF_change"
                ].loc[len(result_tracker_df) - 1]
                + 1,
                "stringsAsFactors": False,
            },
        )

        return (
            pd.DataFrame({"loss": np.nan, "Current point in function space": [point]}),
            result_tracker_df,
        )

    if np.isnan(np.nanmean(new_wflow_para["KsatHorFrac"])):
        print("Function produces only NaN values!\n")
        result_tracker_df = add_to_dataframe(
            result_tracker_df,
            {
                "best_x1": result_tracker_df["best_x1"].loc[len(result_tracker_df) - 1],
                "full_loss": result_tracker_df["full_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "KGE": result_tracker_df["KGE"].loc[len(result_tracker_df) - 1],
                "wKGE": result_tracker_df["wKGE"].loc[len(result_tracker_df) - 1],
                "model_loss": result_tracker_df["model_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "x1": point_tf[parameter_names[0]].loc[0],
                "KGE_run": -9999.0,
                "wKGE_run": -9999.0,
                "loss_run": -9999.0,
                "mean_par_value": -9999.0,
                "n_iterations_used": 0,
                "n_iterations_since_BF_change": result_tracker_df[
                    "n_iterations_since_BF_change"
                ].loc[len(result_tracker_df) - 1]
                + 1,
                "stringsAsFactors": False,
            },
        )

        return (
            pd.DataFrame({"loss": np.nan, "Current point in function space": [point]}),
            result_tracker_df,
        )
    
    print(point_tf[parameter_names[0]].loc[0])
    print(result_tracker_df["x1"].values)
    print(point_tf[parameter_names[0]].loc[0] in result_tracker_df["x1"].values)
    if point_tf[parameter_names[0]].loc[0] in result_tracker_df["x1"].values:
        print("Function already chosen before, choosing new function \n")
        result_tracker_df = add_to_dataframe(
            result_tracker_df,
            result_tracker_df.loc[
                result_tracker_df["x1"] == point_tf[parameter_names[0]].loc[0]
                ].iloc[0])
        
        if result_tracker_df["loss_run"].loc[len(result_tracker_df) - 1] == -9999.0:
            loss = np.nan
        else:
            loss = result_tracker_df["loss_run"].loc[len(result_tracker_df) - 1]

        return (
            pd.DataFrame({"loss": loss, "Current point in function space": [point]}),
            result_tracker_df,
        )

    # TODO: hydromt run
    # Output VAE op parameter kaarten naar TIFF
    KGE_values = []
    uk_tif_path = pathlib.Path(f"./KsatHorFrac_{str(counter).zfill(4)}.tif")
    try:
        os.remove(uk_tif_path)
    except:
        pass
    FSO_functions.convert_1d_array_to_tiff(
        new_wflow_para["KsatHorFrac"].values, example_tif, uk_tif_path
    )
    print("tif created")
    print(training_basin_folder_name)
    # Update parameters aanroepen
    update_parameters.update_parameters(
        basin_folder=training_basin_folder_name,
        pars_to_adjust=pars_to_adjust,
        parameter_files={"KsatHorFrac": uk_tif_path},
        wflow_toml_par_names=wflow_toml_par_names,
        case_name=str(counter),
        starttime_run=start_time_training_run,
        endtime_run=end_time_training_run,
    )
    print("parameters updated")
    # run_wflow
    run_wflow_sbm_multi_threaded.run_wflow_sbm_julia(
        training_basin_folder_name, "training"
    )
    print("wflow gedraait")
    # Determine KGE (loop of in KGE functie loop)
    for training_basin_file in training_basin_folder_name.glob("*"):
        training_basin = training_basin_file.stem
        print(training_basin)
        Q_obs_file = (
            Q_obs_folder
            / f"CAMELS_GB_hydromet_timeseries_{training_basin}_19701001-20150930.csv"
        )
        Q_sim_file = (
            training_basin_folder_name / f"{training_basin}/run_default/output.csv"
        )
        KGE_values.append(
            Determine_KGE.determine_KGE(
                Q_obs_file=Q_obs_file,
                Q_sim_file=Q_sim_file,
                starttime=start_time_training_validation,
                endtime=end_time_training_validation,
                Q_sim_variable_name=Q_sim_variable_names,
            )
        )
    print("KGE is: ")
    print(KGE_values)

    evaluation = FSO_functions.wflow_model_quality(
        test_number=test_number, model_size_loss=model_size_loss, KGE=KGE_values
    )
    mean_KGE = evaluation["mean_KGE"].loc[0]
    wmean_KGE = evaluation["wmean_KGE"].loc[0]
    full_loss = evaluation["full_loss"].loc[0]
    model_loss = evaluation["model_loss"].loc[0]

    old_best = result_tracker_df["full_loss"].loc[len(result_tracker_df) - 1]

    if not np.isnan(full_loss):
        if full_loss > old_best:
            result_tracker_df = add_to_dataframe(
                result_tracker_df,
                {
                    "best_x1": point_tf[parameter_names[0]].loc[0],
                    "full_loss": full_loss,
                    "KGE": mean_KGE,
                    "wKGE": wmean_KGE,
                    "model_loss": model_loss,
                    "x1": point_tf[parameter_names[0]].loc[0],
                    "KGE_run": mean_KGE,
                    "wKGE_run": wmean_KGE,
                    "loss_run": full_loss,
                    "mean_par_value": np.nanmean(new_wflow_para["KsatHorFrac"]),
                    "n_iterations_used": 1,
                    "n_iterations_since_BF_change": 1,
                    "stringsAsFactors": False,
                },
            )
        else:
            result_tracker_df = add_to_dataframe(
                result_tracker_df,
                {
                    "best_x1": result_tracker_df["best_x1"].loc[
                        len(result_tracker_df) - 1
                    ],
                    "full_loss": result_tracker_df["full_loss"].loc[
                        len(result_tracker_df) - 1
                    ],
                    "KGE": result_tracker_df["KGE"].loc[len(result_tracker_df) - 1],
                    "wKGE": result_tracker_df["wKGE"].loc[len(result_tracker_df) - 1],
                    "model_loss": result_tracker_df["model_loss"].loc[
                        len(result_tracker_df) - 1
                    ],
                    "x1": point_tf[parameter_names[0]].loc[0],
                    "KGE_run": mean_KGE,
                    "wKGE_run": wmean_KGE,
                    "loss_run": full_loss,
                    "mean_par_value": np.nanmean(new_wflow_para["KsatHorFrac"]),
                    "n_iterations_used": 1,
                    "n_iterations_since_BF_change": result_tracker_df[
                        "n_iterations_since_BF_change"
                    ].loc[len(result_tracker_df) - 1]
                    + 1,
                    "stringsAsFactors": False,
                },
            )
    else:
        result_tracker_df = add_to_dataframe(
            result_tracker_df,
            {
                "best_x1": result_tracker_df["best_x1"].loc[len(result_tracker_df) - 1],
                "full_loss": result_tracker_df["full_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "KGE": result_tracker_df["KGE"].loc[len(result_tracker_df) - 1],
                "wKGE": result_tracker_df["wKGE"].loc[len(result_tracker_df) - 1],
                "model_loss": result_tracker_df["model_loss"].loc[
                    len(result_tracker_df) - 1
                ],
                "x1": point_tf[parameter_names[0]].loc[0],
                "KGE_run": mean_KGE,
                "wKGE_run": wmean_KGE,
                "loss_run": full_loss,
                "mean_par_value": np.nanmean(new_wflow_para["KsatHorFrac"]),
                "n_iterations_used": 1,
                "n_iterations_since_BF_change": result_tracker_df[
                    "n_iterations_since_BF_change"
                ].loc[len(result_tracker_df) - 1]
                + 1,
                "stringsAsFactors": False,
            },
        )

    print(f"\nDDS optimization for test nr {test_number} results:\n")
    print(f"mean KGE: {mean_KGE}\n")
    print(f"wmean KGE: {wmean_KGE}\n")
    print(f"spaef/model_loss KGE: {model_loss}\n")
    print(f"overall loss: {full_loss}\n")

    print("\nThe best functions are:\n")
    print(f"x1 = {result_tracker_df['best_x1'].loc[len(result_tracker_df) - 1]}\n")

    return (
        pd.DataFrame({"loss": full_loss, "Current point in function space": [point]}),
        result_tracker_df,
    )
