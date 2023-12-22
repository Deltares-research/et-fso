import pathlib

import pandas as pd
import xarray as xr
import numpy as np


# Script for combining dataframes from parallel computation steps 1-2-1 and 1-2-2, to rescale


if __name__ == "__main__":
    path = pathlib.Path("/gpfs/home4/aweerts/Data")
    threshold = 11

    tf_dist_1 = pd.read_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_1.feather"
        )
    )

    tf_dist_2 = pd.read_feather(
        path.joinpath(
            "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_2.feather"
        )
    )

    tf_dist = pd.concat([tf_dist_1, tf_dist_2], axis=0).reset_index(drop=True)

    generator_data_1 = pd.read_feather(
        path.joinpath("generator_data_simple_10numerics_wrecalc_allBasins_1.feather")
    )

    generator_data_2 = pd.read_feather(
        path.joinpath("generator_data_simple_10numerics_wrecalc_allBasins_2.feather")
    )

    generator_data = pd.concat(
        [generator_data_1, generator_data_2], axis=0
    ).reset_index(drop=True)

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
            "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_no_extremes.feather"
        )
    )
    generator_data.reset_index(drop=True).to_feather(
        path.joinpath(
            "generator_data_simple_10numerics_wrecalc_allBasins_no_extremes.feather"
        )
    )
