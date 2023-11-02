from typing import Union
import numpy as np
import pandas as pd
import pathlib


def prob_perturb(x, num_iter):
    """
    Perturber function for DDS

    Inputs:
        - xbounds

    Output:
        - num_iter entry list with the indices which will be perturbed

    """

    x_dims = x.shape[0]
    probability_vector = 1 - np.log(np.arange(1, num_iter + 1)) / np.log(num_iter)
    perturb_idx = []
    for p in probability_vector:
        perturb_idx.append(np.where(np.random.binomial(1, p, x_dims) == 1)[0])

    return perturb_idx


def dds_fs(
    xbounds_df: pd.DataFrame,
    num_iter: int,
    obj_fun,
    search_dim: int,
    test_number: int,
    spatial_predictors: pd.DataFrame,
    parameter_bounds: pd.DataFrame,
    parameter_names: list,
    result_tracker_df: pd.DataFrame,
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
    """
    Inputs:
        - xbounds_df dataframe with 1st column as minimum and 2nd column as maximum
        - num_iter number of iterations
        - obj_fun objective function which returns a scalar value, which we want to minimize

    Output:
        - output_df dataframe containing x_best and y_best as they evolve over num_iter iterations
    """

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

    counter = 1
    print(f"\n****** Point nr. {counter} / {num_iter} ******\n")

    xbounds_df.columns = ["min", "max"]
    # Generate initial x value
    x_init = np.random.standard_normal(search_dim)

    x_evaluated, result_tracker_df = obj_fun(
        point=x_init,
        test_number=test_number,
        spatial_predictors=spatial_predictors,
        parameter_bounds=parameter_bounds,
        parameter_names=parameter_names,
        result_tracker_df=result_tracker_df,
        counter=counter,
        **args,
    )

    x_init = x_evaluated["Current point in function space"]
    x_best = pd.DataFrame({0: [x_init.values[0]]}, index=[0])

    if not np.any(np.isnan(x_evaluated["loss"].values)):
        y_init = x_evaluated["loss"]
    else:
        y_init = pd.DataFrame({0: -999}, index=[0])[0]
    y_best = y_init
    r = 0.2

    # Select which entry to perturb at each iteration
    perturb_idx = prob_perturb(xbounds_df, num_iter)
    # Perturb each entry by N(0, 1) * r(x_max - x_min) reflecting if at boundaries
    sigma = xbounds_df["max"] - xbounds_df["min"]
    for i in range(1, num_iter):
        # Set up test x
        x_test = x_best[i - 1].loc[0]
        # Get entries we will perturb
        idx = perturb_idx[i]
        # Initialize vector of perturbations initially zeros with same length of x so we will add this vector to perturb x
        perturb_vec = np.zeros(len(x_test))
        # Generate the required number of random normal variables
        N = np.random.standard_normal(len(x_test))
        # Set up vector of perturbations
        perturb_vec[idx] = r * N[idx] * sigma[idx]
        # Temporary resulting x value if we perturbed it
        test_perturb = x_test + perturb_vec
        # Find the values in testPerturb OBJFUN <- wrapper_ofthat have boundary violations.  Store the indices in boundaryViolationsIdx
        # TODO: Hier naar kijken
        boundary_violation_idx = np.where(
            (test_perturb < xbounds_df["min"]) | (test_perturb > xbounds_df["max"])
        )[0]
        # Reset those violated indices to the opposite peturbation direction
        perturb_vec[boundary_violation_idx] = (
            -r * N[boundary_violation_idx] * sigma[boundary_violation_idx]
        )
        # Find values still at violations of min or max and set them to the minimum or maximum values
        test_perturb = x_test + perturb_vec
        min_violation_idx = np.where(test_perturb < xbounds_df["min"])[0]
        max_violation_idx = np.where(test_perturb > xbounds_df["max"])[0]
        test_perturb[min_violation_idx] = xbounds_df["min"][min_violation_idx]
        test_perturb[max_violation_idx] = xbounds_df["max"][max_violation_idx]
        # Perturb the test vector
        x_test = x_test + perturb_vec

        counter += 1
        print(f"\n****** Point nr. {counter} / {num_iter} ******\n")
        # Evaluate objective function
        x_evaluated, result_tracker_df = obj_fun(
            point=x_test,
            test_number=test_number,
            spatial_predictors=spatial_predictors,
            parameter_bounds=parameter_bounds,
            parameter_names=parameter_names,
            result_tracker_df=result_tracker_df,
            counter=counter,
            **args,
        )
        x_test = x_evaluated["Current point in function space"]
        y_test = x_evaluated["loss"].values
        if not np.isnan(y_test):
            y_best[i] = np.max([y_test, y_best[i - 1]])
            best_idx = np.argmax([y_test, y_best[i - 1]])
        else:
            y_best[i] = y_best[i - 1]
            best_idx = 1
        x_choices = pd.DataFrame({0: x_test, 1: x_best[i - 1]})
        x_best[x_best.shape[1]] = x_choices[
            best_idx
        ]  # pd.DataFrame(x_best, x_choices[:, best_idx])
        output_list = {"x_best": x_best, "y_best": y_best}

    return output_list, result_tracker_df
