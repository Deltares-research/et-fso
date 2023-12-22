import re
import rasterio
import pathlib

import pandas as pd
import xarray as xr
import numpy as np

import multiprocessing as mp
import pyarrow.feather as feather

from functools import partial


def range01(x, bounds):
    # Function for scaling x to [0, 1]
    minimum = np.nanmin(x)
    maximum = np.nanmax(x)
    return (x - minimum) / (maximum - minimum)


def load_sp_wflow(
    path,
    parameters,
    bounds={
        "silt": [350, 600],
        "sand": [157, 437],
        "clay": [133, 411],
        "oc": [421, 1367],
        "bd": [87, 123],
        "ph": [49, 68],
        "bl": [842, 1571],
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
    Function for evaluating the function string on the parameter maps l0

    Inputs:
        string: a function string to be applied to parameter maps e.g. 'exp(bd)+1.1*ph'
        l0: dataframe containing flattened parameter maps for one or multiple parameters
    Output:
        f_evaluated: a flattened array as outcome of the function string applied to the relevant parameter maps
    """

    delimeters = "\/|\^|\(|\)|\*|\+|\-"
    func_string = string
    # Split function string into relevant components separated by delimeters
    tf = re.split(delimeters, string)
    tf = [re.sub(" ", "", i) for i in tf]
    tf = [re.sub("exp", "np.exp", i) for i in tf]
    tf = [re.sub("log", "np.log", i) for i in tf]
    tf = [re.sub("\^", "**", i) for i in tf]

    # Substite certain patterns in the string with python syntax
    func_string = re.sub(" ", "", func_string)
    func_string = re.sub("exp", "np.exp", func_string)
    func_string = re.sub("log", "np.log", func_string)
    func_string = re.sub("\^", "**", func_string)

    df = {}
    locals_parameter = {"df": df, "l0": l0}
    names = []
    # Loop over parameters in l0 and add relevant parameter maps to dictionary df
    for i in l0.columns:
        if i in tf:
            names.append(i)
            df[i] = l0[i]
        # Substitute parameter name with df['parameter name'] in function string
        func_string = re.sub(i, "df['" + i + "']", func_string)
    if len(names) == 0:
        # If there are no parameters in the function string, return flattened array filled with function string (number) instead
        f_evaluated = np.full(l0.shape[0], string)
    else:
        # Create python function from function string with input dictionary and apply it to dictionary df
        func = "def f(df):\n\t" + "return " + func_string
        exec(func, globals())
        f_evaluated = f(df).values
    return f_evaluated


def distribution_values_from_tf(tf, spatial_predictors, cut_off=None):
    """
    Function for computing the distribution of the values of the evaluated parameter map

    Inputs:
        tf: function string
        spatial_predictors: dataframe containing flattened parameter maps for one or multiple parameters
        cut_off: list, tuple or array of length 2 with minimum and maximum allowable value in the evaluated maps
    Output:
        distribution_values: dataframe containing some quantiles of the evaluated parameter map
    """
    tf_evaluated = evaluate_function_from_string(tf, l0=spatial_predictors)
    if cut_off is not None:
        tf_evaluated[np.where(tf_evaluated < cut_off[0])] = cut_off[0]
        tf_evaluated[np.where(tf_evaluated > cut_off[1])] = cut_off[1]

    dist = np.nanquantile(
        tf_evaluated,
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        method="nearest",
    ).reshape(1, 11)
    distribution_values = np.round(dist, 4)
    column_names = [
        "min",
        "10%",
        "20%",
        "30%",
        "40%",
        "mean",
        "60%",
        "70%",
        "80%",
        "90%",
        "max",
    ]
    distribution_values = pd.DataFrame(distribution_values, columns=column_names)

    return distribution_values


if __name__ == "__main__":
    # Load parameter maps
    path = pathlib.Path("/gpfs/home4/aweerts/Data")
    parameters = ["silt", "clay", "sand", "oc", "bd", "ph", "bl"]
    spatial_predictors = load_sp_wflow(path.joinpath("spatial_predictors"), parameters)
    # Load generated function strings
    functions_simple_10_numerics = pd.read_feather(
        path.joinpath("functions_simple_10_numerics_pp.feather")
    )
    functions_simple_10_numerics = functions_simple_10_numerics.iloc[
        : int(len(functions_simple_10_numerics) / 2)
    ]

    partial_function = partial(
        distribution_values_from_tf, spatial_predictors=spatial_predictors
    )
    # Create pools for parallel processing
    p = mp.Pool(80)
    # Concatenate the parallely computed disttribution dataframes
    result = pd.concat(
        p.map(
            partial_function,
            functions_simple_10_numerics["TF_simple_10numerics"].values,
        )
    )
    result.insert(
        0,
        column="transfer_function",
        value=functions_simple_10_numerics["TF_simple_10numerics"].values,
    )
    result.reset_index(drop=True).to_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_allBasins_1.feather"
        )
    )

    generator_data = pd.read_feather(
        path.joinpath("generator_data_simple_10numerics.feather")
    )
    generator_data = generator_data.iloc[: int(len(generator_data) / 2)]
    tf_dist = pd.read_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_allBasins_1.feather"
        )
    )

    generator_data = generator_data.dropna()
    tf_dist = tf_dist.dropna()

    threshold = 11
    or_ind = (abs(tf_dist["min"]) > threshold) | (abs(tf_dist["max"]) > threshold)

    tf_dist_recalc = tf_dist[or_ind]
    generator_data_recalc = generator_data[or_ind]

    tf_dist = tf_dist[~or_ind]
    generator_data = generator_data[~or_ind]

    out_of_range = (
        (tf_dist_recalc["max"] > threshold) & (tf_dist_recalc["min"] > threshold)
    ) | ((tf_dist_recalc["max"] < -threshold) & (tf_dist_recalc["min"] < -threshold))

    tf_dist_recalc = tf_dist_recalc[~out_of_range]
    generator_data_recalc = generator_data_recalc[~out_of_range]

    partial_function = partial(
        distribution_values_from_tf,
        spatial_predictors=spatial_predictors,
        cut_off=[-threshold, threshold],
    )
    tf_distributions_recalc = pd.concat(
        p.map(partial_function, tf_dist_recalc["transfer_function"])
    )

    p.close()

    tf_distributions_recalc.insert(
        0, column="transfer_function", value=tf_dist_recalc["transfer_function"].values
    )

    pd.concat([tf_dist, tf_distributions_recalc], axis=0).reset_index(
        drop=True
    ).to_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_1.feather"
        )
    )
    pd.concat([generator_data, generator_data_recalc], axis=0).reset_index(
        drop=True
    ).to_feather(
        path.joinpath("generator_data_simple_10numerics_wrecalc_allBasins_1.feather")
    )

    tf_dist = pd.read_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_1.feather"
        )
    )
    generator_data = pd.read_feather(
        path.joinpath("generator_data_simple_10numerics_wrecalc_allBasins_1.feather")
    )

    extreme_tfs_ind = (
        (tf_dist["min"] == -threshold)
        & (tf_dist["10%"] == -threshold)
        & (tf_dist["20%"] == -threshold)
    ) | (
        (tf_dist["80%"] == threshold)
        & (tf_dist["90%"] == threshold)
        & (tf_dist["max"] == threshold)
    )
    tf_dist = tf_dist[~extreme_tfs_ind]
    generator_data = generator_data[~extreme_tfs_ind]

    tf_dist.reset_index(drop=True).to_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_no_extremes_1.feather"
        )
    )
    generator_data.reset_index(drop=True).to_feather(
        path.joinpath(
            "generator_data_simple_10numerics_wrecalc_allBasins_no_extremes_1.feather"
        )
    )
