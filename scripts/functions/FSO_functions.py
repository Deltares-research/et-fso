import os
import pathlib
import re
import time

import numpy as np
import pandas as pd
import rasterio
import traceback

# import upscale_parameter_change_staticmap
from functions import update_parameters
from functions import (
    run_wflow_sbm_multi_threaded_small_sample as run_wflow_sbm_multi_threaded,
)
from functions import Determine_KGE
from functions import DDS_optimization_wflow
from functions import dds_fs


# import rioxarray as rxr
def rescale(x, from_range, to_range):
    if np.std(x) != 0:
        return (x - from_range[0]) / (from_range[1] - from_range[0]) * (
            to_range[1] - to_range[0]
        ) + to_range[0]
    else:
        return x


def range01(x, bounds):
    # Function for scaling x to [0, 1]
    minimum = np.nanmin(x)
    maximum = np.nanmax(x)
    return (x - minimum) / (maximum - minimum)


def weighted_mean(x, weights):
    """
    Computes the weighted mean of x

    Input:
        - x vector of values to compute the mean over
        - weights vector of weights corresponding to values of x

    Output:
        - weighted mean of x

    """
    try:
        assert x.shape == weights.shape
    except AttributeError:
        assert len(x) == len(weights)

    weighted_sum = np.sum(np.array(x) * np.array(weights))
    total_weights = np.sum(weights)

    weighted_mean = weighted_sum / total_weights

    return weighted_mean


def load_sp_wflow(
    path,
    parameters,
    bounds={
        "silt": [350, 600],
        "sand": [157, 437],
        "clay": [133, 411],
        "OC": [421, 1367],
        "BD": [87, 123],
        "pH": [49, 68],
        "BL": [842, 1571],
    },
    scale=True,
    nanapprox=False,
) -> pd.DataFrame:
    path = pathlib.Path(path)
    spatial_predictors = {}
    for parameter in parameters:
        src = rasterio.open(
            path.joinpath(parameter + "/" + "VK_" + parameter + "_wavg_250m.tif")
        )
        raster = src.read(1).astype(np.float64)
        nan_mask = src.read_masks(1)
        raster[nan_mask == 0] = np.nan

        if nanapprox:
            nans = np.where(np.isnan(raster))
            for i, j in zip(nans[0], nans[1]):
                i_l = int(np.clip(i - 1, 0, raster.shape[0] - 1))
                i_h = int(np.clip(i + 1, 0, raster.shape[0] - 1))
                raster[i, j] = np.nanmean([raster[i_l, j], raster[i_h, j]])

        if scale:
            raster = range01(raster, bounds[parameter])

        raster[np.where(raster == 0)] = 0.0001

        spatial_predictors[parameter] = raster.flatten()
    spatial_predictors = pd.DataFrame(spatial_predictors)
    return spatial_predictors


def evaluate_function_from_string(string: str, l0: pd.DataFrame):
    """
    Apply function string to parameter map

    Input:
        - function string, e.g. '5+exp(sand)'
        - pandas dataframe l0 containing flattened parameter maps for (multiple) parameter(s)

    Output:
        - flattened array with values corresponding to the function string applied to the parameter maps

    """
    # Define delimeters with operations /, ^, (, ), *, +, -
    delimeters = "\/|\^|\(|\)|\*|\+|\-"
    func_string = string

    # Get list with all components of the function
    # Change exp, log and ^ to python syntax
    tf = re.split(delimeters, string)
    tf = [re.sub(" ", "", i) for i in tf]
    tf = [re.sub("exp", "np.exp", i) for i in tf]
    tf = [re.sub("log", "np.log", i) for i in tf]
    tf = [re.sub("\^", "**", i) for i in tf]

    func_string = re.sub(" ", "", func_string)
    func_string = re.sub("exp", "np.exp", func_string)
    func_string = re.sub("log", "np.log", func_string)
    func_string = re.sub("\^", "**", func_string)

    # Define dictionary for flattened array for different parameters
    df = {}
    names = []

    # Loop over parameter names in l0, add to names if they appear in function string
    # Rename parameter name in func_string to df['parameter name'], e.g. sand becomes df['sand']
    for i in l0.columns:
        if i in tf:
            names.append(i)
            df[i] = l0[i]
        tf = [re.sub(i, "df['" + i + "']", j) for j in tf]
        func_string = re.sub(i, "df['" + i + "']", func_string)
    # If no parameter names in l0 appear in function string, return array filled with the function string
    if len(names) == 0:
        f_evaluated = np.full(l0.shape[0], string)
    # Otherwise, create a function of the function string and apply it to df
    else:
        func = "def f(df):\n\t" + "return " + func_string
        exec(func, globals())
        f_evaluated = f(df).values

    return f_evaluated


def function_splitter(point_tf):
    delimeters = "\/|\^|\(|\)|\*|\+|\-"
    function_splitted = re.split(delimeters, point_tf)
    # function_splitted = re.sub(" ", "", function_splitted)
    # function_splitted = function_splitted[function_splitted != ""]
    function_splitted = [i for i in function_splitted if i != ""]
    return function_splitted


def size_loss(functions_splitted):
    return len(functions_splitted) * 0.001


def create_wflow_para(transfer_functions, l0, parameter_bounds, parameter_names):
    """
    Create wflow parameter from a list with transfer functions

    Input:
        - transfer_functions: dictionary or dataframe with transfer strings for parameters
        - l0: dataframe with l0 layer
        - parameter_bounds: dictionary or dataframe with parameter bounds for parameters
    Output:
        - dataframe with new wflow parameters

    """
    new_wflow_para = pd.DataFrame({"KsatHorFrac": np.full(len(l0["clay"]), np.nan)})

    for p in parameter_names:
        if type(transfer_functions) is pd.DataFrame:
            new_wflow_para[p] = evaluate_function_from_string(
                transfer_functions[p].loc[0], l0=l0
            )
        elif type(transfer_functions) is str:
            new_wflow_para[p] = evaluate_function_from_string(transfer_functions, l0=l0)
            new_wflow_para[p] = np.round(
                rescale(
                    new_wflow_para[p],
                    from_range=[-11, 11],
                    to_range=parameter_bounds[p],
                ),
                2,
            )

    return new_wflow_para


def wflow_model_quality(test_number, model_size_loss, KGE):
    mean_KGE = np.mean(KGE)
    wmean_KGE = weighted_mean(KGE, weights=1.01 - np.array(KGE))

    full_loss = wmean_KGE - model_size_loss
    model_loss = wmean_KGE - model_size_loss

    output = pd.DataFrame(
        {
            "mean_KGE": mean_KGE,
            "wmean_KGE": wmean_KGE,
            "model_loss": model_loss,
            "full_loss": full_loss,
        },
        index=[0],
    )

    return output


def convert_1d_array_to_tiff(src_array, example_tif, output_file):
    """
    Converts the 1D array output from 'evaluate_function_from_string' to a
    2D GEOTiff file, containing the metadata of an example GEOTiff file
    Writes the new file to output_file

    Input:
        - 1D array src_array
        - Example tiff file example_tif to be used as reference for shape and metadata
        - output_file file of destination
    Output:
        - None
    """

    ref = rasterio.open(example_tif)

    src_array_2D = src_array.reshape(ref.read().shape)
    meta = ref.meta
    src_array_2D[np.where(np.isnan(src_array_2D))] = meta["nodata"]

    with rasterio.open(output_file, "w", **meta) as dst:
        dst.update_tags(**meta)
        dst.write(src_array_2D)
        dst.close()

    return


def evaluate_basins(
    functions,
    optimizer,
    test_number,
    spatial_predictors,
    run,
    counter,
    parameter_bounds,
    parameter_names,
    run_type: str,
    save_folder,
    run_folder,
    test_basin_folder_name,
    training_basin_folder_name,
    Q_obs_folder,
    Q_sim_variable_names,
    start_time_test_run,
    end_time_test_run,
    start_time_test_validation,
    end_time_test_validation,
    start_time_training_run,
    end_time_training_run,
    start_time_training_validation,
    end_time_training_validation,
    pars_to_adjust,
    wflow_toml_par_names,
):
    src_dir = save_folder

    if run_type == "training":
        functions = pd.DataFrame({"KsatHorFrac": functions["best_x1"]})
        functions = functions["KsatHorFrac"].loc[counter]
    elif run_type == "test":
        functions = functions["KsatHorFrac"].loc[0]

    functions_splitted = function_splitter(functions)
    model_size_loss = size_loss(functions_splitted=functions_splitted)

    new_wflow_para = create_wflow_para(
        transfer_functions=functions,
        l0=spatial_predictors,
        parameter_bounds=parameter_bounds,
        parameter_names=parameter_names,
    )

    KGE_model_run_train = []
    KGE_model_run_test = []

    # TODO: Look at where and when to write uk_tif_path (dds opti or fso?)
    path = pathlib.Path(__file__).parent.parent.parent.resolve()

    example_tif = path / "Data/spatial_predictors"
    uk_tif_path = pathlib.Path(f"./KsatHorFrac_{str(counter+1).zfill(4)}.tif")
    try:
        os.remove(uk_tif_path)
    except:
        pass
    convert_1d_array_to_tiff(
        new_wflow_para["KsatHorFrac"].values, example_tif, uk_tif_path
    )

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

    # run_wflow
    run_wflow_sbm_multi_threaded.run_wflow_sbm_julia(
        training_basin_folder_name, "training"
    )

    # Determine KGE (loop of in KGE functie loop)
    for training_basin_file in pathlib.Path(training_basin_folder_name).glob("*"):
        training_basin = training_basin_file.stem
        Q_obs_file = (
            Q_obs_folder
            / f"CAMELS_GB_hydromet_timeseries_{training_basin}_19701001-20150930.csv"
        )
        Q_sim_file = (
            training_basin_folder_name / f"{training_basin}/run_default/output.csv"
        )
        KGE_model_run_train.append(
            Determine_KGE.determine_KGE(
                Q_obs_file=Q_obs_file,
                Q_sim_file=Q_sim_file,
                starttime=start_time_training_validation,
                endtime=end_time_training_validation,
                Q_sim_variable_name=Q_sim_variable_names,
            )
        )

    train_evaluation = wflow_model_quality(
        test_number=test_number,
        model_size_loss=model_size_loss,
        KGE=KGE_model_run_train,
    )

    train_results = pd.DataFrame(
        {
            "mean_KGE": np.round(train_evaluation["mean_KGE"], 3),
            "weighted_mean_KGE": np.round(train_evaluation["wmean_KGE"], 3),
            "SPAEF/model_loss": np.round(train_evaluation["model_loss"], 3),
            "full_loss": np.round(train_evaluation["full_loss"], 3),
        }
    )

    # Also run for test if run_type = test
    if run_type == "test":
        # Update parameters aanroepen
        update_parameters.update_parameters(
            basin_folder=test_basin_folder_name,
            pars_to_adjust=pars_to_adjust,
            parameter_files={"KsatHorFrac": uk_tif_path},
            wflow_toml_par_names=wflow_toml_par_names,
            case_name=str(counter),
            starttime_run=start_time_test_run,
            endtime_run=end_time_test_run,
        )

        # run_wflow
        run_wflow_sbm_multi_threaded.run_wflow_sbm_julia(test_basin_folder_name, "test")

        # Determine KGE (loop of in KGE functie loop)
        for test_basin_file in pathlib.Path(test_basin_folder_name).glob("*"):
            test_basin = test_basin_file.stem
            Q_obs_file = (
                Q_obs_folder
                / f"CAMELS_GB_hydromet_timeseries_{test_basin}_19701001-20150930.csv"
            )
            Q_sim_file = test_basin_folder_name / f"{test_basin}/run_default/output.csv"
            KGE_model_run_test.append(
                Determine_KGE.determine_KGE(
                    Q_obs_file=Q_obs_file,
                    Q_sim_file=Q_sim_file,
                    starttime=start_time_test_validation,
                    endtime=end_time_test_validation,
                    Q_sim_variable_name=Q_sim_variable_names,
                )
            )

        KGE_model_run_total = KGE_model_run_train + KGE_model_run_test

        mean_train_KGE = np.mean(KGE_model_run_train)
        mean_test_KGE = np.mean(KGE_model_run_test)

        test_evaluation = wflow_model_quality(
            test_number=test_number,
            model_size_loss=model_size_loss,
            KGE=KGE_model_run_test,
        )

        total_evaluation = wflow_model_quality(
            test_number=test_number,
            model_size_loss=model_size_loss,
            KGE=KGE_model_run_total,
        )

        mean_results = pd.DataFrame(
            {
                "mean training KGE": np.round(mean_train_KGE, 3),
                "mean test KGE": np.round(mean_test_KGE, 3),
            }
        )

        test_results = pd.DataFrame(
            {
                "mean_KGE": np.round(test_evaluation["mean_KGE"], 3),
                "weighted_mean_KGE": np.round(test_evaluation["wmean_KGE"], 3),
                "SPAEF/model_loss": np.round(test_evaluation["model_loss"], 3),
                "full_loss": np.round(test_evaluation["full_loss"], 3),
            }
        )

        total_results = pd.DataFrame(
            {
                "mean_KGE": np.round(total_evaluation["mean_KGE"], 3),
                "weighted_mean_KGE": np.round(total_evaluation["wmean_KGE"], 3),
                "SPAEF/model_loss": np.round(total_evaluation["model_loss"], 3),
                "full_loss": np.round(total_evaluation["full_loss"], 3),
            }
        )

        results = {
            "mean": mean_results,
            "train": train_results,
            "test": test_results,
            "total": total_results,
        }
        print(
            f"\nSaved testing results of test nr. {test_number} in corresponding folder.\n"
        )
        file_name = src_dir / f"{optimizer}_testing_{test_number}_run{run}.txt"
        with open(file_name, "w") as file:
            print(
                f"Testing results for test number {test_number}: {optimizer} - run {run}\n\n",
                file=file,
            )
            print(f"General results:\n ", file=file)
            total_results.to_csv(file, index=False)
            print(f"\nTraining basins:\n", file=file)
            train_results.to_csv(file, index=False)
            print(f"\nTest basins:\n", file=file)
            test_results.to_csv(file, index=False)
            print(f"\n\n\nThe tested functions are:\n", file=file)
            print(functions, file=file)

        return results

    file_name = src_dir / f"{optimizer}_training_{test_number}_run{run}.txt"

    with open(file_name, "w") as file:
        print(
            f"Training results for test number {test_number}: {optimizer} - run {run}\n\n",
            file=file,
        )
        train_results.to_csv(file, index=False)
        print(f"\n\n\nThe optimized functions are:\n", file=file)
        print(f"x1 = {functions}\n", file=file)

    return


# path = pathlib.Path('C:/Users/hemert/OneDrive - Stichting Deltares/Desktop/Projects/ET_FSO/Data/spatial_predictors')
# parameters = ["silt", "sand"]
# bounds = {"silt": [350, 600], "sand": [157, 437], "clay": [133, 411], "OC": [421, 1367], "BD": [87, 123], "pH": [49, 68]}

# print(load_sp_wflow(path, parameters, bounds, nanapprox=True))


def FSO_setup():
    # Give the run folder
    # src_path = pathlib.Path(
    #     __file__
    # ).parent.parent.parent.resolve()  # ET_FSO Snellius map
    src_path = pathlib.Path("/gpfs/work2/0/drse0610")
    print("\nCase study is setup as defined in Functions/case_study_setup.\n")
    run_folder = src_path / "UK_basins"

    training_basin_folder_name = run_folder / "Training_small"
    test_basin_folder_name = run_folder / "Test"

    # 2. Information about the observations
    # The location with the observations
    Q_obs_folder = run_folder / "Obs"
    # Also give the names of the discharge variable (assumed to be the same for every basin)
    Q_sim_variable_names = "Q_1"

    # 3. Set the start and end time for the training and test run and the start and end
    # times for the validation (different start time than the run due to need for warming
    # up of the model)
    # 3.1 First for the wflow_sbm model runs, the dates should be formatted as: %YYYY-%mm-%DDT%HH:%MM:%SS
    start_time_training_run = "2010-01-01T00:00:00"
    end_time_training_run = "2015-01-01T00:00:00"
    start_time_test_run = "1970-01-01T00:00:00"
    end_time_test_run = "2009-12-31T00:00:00"
    # 3.2 Then the validation times, the dates should be formatted as: %YYYY-%mm-%DD %HH:%MM:%SS
    start_time_training_validation = "2011-01-01 00:00:00"
    end_time_training_validation = "2015-01-01 00:00:00"
    start_time_test_validation = "1972-01-01 00:00:00"
    end_time_test_validation = "2009-12-31 00:00:00"

    # 4. Information about the parameters that have to be adjusted
    # List them as the names given in staticmaps.nc
    pars_to_adjust = ["KsatHorFrac"]
    # Also list the internal names used in the .toml (configuration) file, as this gets
    # updated as well
    wflow_toml_par_names = ["input.lateral.subsurface.ksathorfrac"]

    args = {
        "run_folder": run_folder,
        "training_basin_folder_name": training_basin_folder_name,
        "test_basin_folder_name": test_basin_folder_name,
        "Q_obs_folder": Q_obs_folder,
        "Q_sim_variable_names": Q_sim_variable_names,
        "start_time_training_run": start_time_training_run,
        "end_time_training_run": end_time_training_run,
        "start_time_test_run": start_time_test_run,
        "end_time_test_run": end_time_test_run,
        "start_time_training_validation": start_time_training_validation,
        "end_time_training_validation": end_time_training_validation,
        "start_time_test_validation": start_time_test_validation,
        "end_time_test_validation": end_time_test_validation,
        "pars_to_adjust": pars_to_adjust,
        "wflow_toml_par_names": wflow_toml_par_names,
    }

    return args


def FSO(
    path: pathlib.Path,
    optimizer: str,
    test_number,
    run,
    spatial_predictors,
    iterations,
    parameter_bounds,
    parameter_names,
    run_folder,
    training_basin_folder_name,
    test_basin_folder_name,
    Q_obs_folder,
    Q_sim_variable_names,
    start_time_test_run,
    end_time_test_run,
    start_time_test_validation,
    end_time_test_validation,
    start_time_training_run,
    end_time_training_run,
    start_time_training_validation,
    end_time_training_validation,
    pars_to_adjust,
    wflow_toml_par_names,
):
    args = {
        "run_folder": run_folder,
        "training_basin_folder_name": training_basin_folder_name,
        "test_basin_folder_name": test_basin_folder_name,
        "Q_obs_folder": Q_obs_folder,
        "Q_sim_variable_names": Q_sim_variable_names,
        "start_time_training_run": start_time_training_run,
        "end_time_training_run": end_time_training_run,
        "start_time_test_run": start_time_test_run,
        "end_time_test_run": end_time_test_run,
        "start_time_training_validation": start_time_training_validation,
        "end_time_training_validation": end_time_training_validation,
        "start_time_test_validation": start_time_test_validation,
        "end_time_test_validation": end_time_test_validation,
        "pars_to_adjust": pars_to_adjust,
        "wflow_toml_par_names": wflow_toml_par_names,
    }

    # 1. Setup
    print(
        f"\n*** Test number {test_number} - {optimizer}, optimization run {run} ***\n"
    )

    # Generate test specific folders in directory
    general_test_folder = path / f"Test {str(test_number)[0]}"
    subtest_folder = general_test_folder / f"Test {test_number}"
    training_folder = subtest_folder / "training"
    testing_folder = subtest_folder / "testing"

    general_test_folder.mkdir(exist_ok=True)
    subtest_folder.mkdir(exist_ok=True)
    training_folder.mkdir(exist_ok=True)
    testing_folder.mkdir(exist_ok=True)

    # Plot paths
    # Path for saving rasters
    para_fields = subtest_folder / "parameter_fields"

    para_fields.mkdir(exist_ok=True)
    # Paths for specific para fields
    para_fields2 = para_fields.joinpath(f"run_{run}")
    para_fields2.mkdir(exist_ok=True)
    para_fields2.joinpath("Plots").mkdir(exist_ok=True)
    # Path for plots
    diag_path = para_fields.joinpath(f"diagnostic_plots/run_{run}/{optimizer}")
    diag_path.mkdir(parents=True, exist_ok=True)

    # 2. Training
    print("start training")

    # TODO: not sure if 6 is correct. Originally was 18
    xbounds_df = pd.DataFrame({"lower": np.full(6, -5), "upper": np.full(6, 5)})
    result_tracker_df = pd.DataFrame(
        {
            "best_x1": "init",
            "full_loss": -9999.0,
            "KGE": -9999.0,
            "wKGE": -9999.0,
            "model_loss": -9999.0,
            "x1": "init",
            "KGE_run": -9999.0,
            "wKGE_run": -9999.0,
            "loss_run": -9999.0,
            "mean_par_value": -9999.0,
            "n_iterations_used": 0,
            "n_iterations_since_BF_change": 0,
            "stringsAsFactors": False,
        },
        index=[0],
    )

    # new_wflow_para = create_wflow_para(
    #     transfer_functions="sand+1.2*clay",
    #     l0=spatial_predictors,
    #     parameter_bounds=parameter_bounds,
    #     parameter_names=parameter_names,
    #     )
    dds_fs_error = True
    # uk_tif_path = pathlib.Path(f"./KsatHorFrac_{str(1).zfill(4)}.tif")
    # example_tif = path / "Data/spatial_predictors/BD/VK_BD_wavg_250m.tif"
    # convert_1d_array_to_tiff(
    #     new_wflow_para["KsatHorFrac"].values, example_tif, uk_tif_path
    # )
    for try_n in range(100):
        try:
            results = dds_fs.dds_fs(
                xbounds_df=xbounds_df,
                num_iter=iterations,
                obj_fun=DDS_optimization_wflow.objective_function,
                test_number=test_number,
                search_dim=len(parameter_names) * 6,
                spatial_predictors=spatial_predictors,
                parameter_bounds=parameter_bounds,
                parameter_names=parameter_names,
                result_tracker_df=result_tracker_df,
                **args,
            )[1]
            dds_fs_error = None
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass

        if dds_fs_error:
            time.sleep(1)
        elif dds_fs_error is None:
            break
    # TODO: look at src_dir
    src_dir = pathlib.Path(path)
    runs = np.sum(results["n_iterations_used"])
    results.attrs = {
        "method": "dds",
        "n_runs": runs,
        "stringAsFactors": False,
    }
    results.to_csv(
        src_dir
        / f"Test {str(test_number)[0]}/Test {test_number}/{optimizer}_wflow_optimization_{test_number}_run{run}.csv"
    )

    end_results = results.tail(1)
    evaluate_basins(
        functions=end_results,
        optimizer=optimizer,
        test_number=test_number,
        spatial_predictors=spatial_predictors,
        run=run,
        counter=iterations,
        parameter_bounds=parameter_bounds,
        parameter_names=parameter_names,
        run_type="training",
        save_folder=training_folder,
        **args,
    )

    result_functions = pd.DataFrame(
        {"KsatHorFrac": results["best_x1"].loc[len(results) - 1]}, index=[0]
    )

    print("\n------------------------------------------\n")
    print("Finished Function Space Optimization\nOptimized function:\n")

    for name in result_functions.columns:
        print(f"{name}: {result_functions[name]}\n")

    # 3. Testing
    print("Start testing")
    # Start test evaluation
    test_results = evaluate_basins(
        functions=result_functions,
        optimizer=optimizer,
        test_number=test_number,
        spatial_predictors=spatial_predictors,
        run=run,
        counter=iterations,
        parameter_bounds=parameter_bounds,
        parameter_names=parameter_names,
        run_type="test",
        save_folder=testing_folder,
        **args,
    )
