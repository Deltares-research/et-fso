import pathlib
import pandas as pd
import functions
from functions import FSO_functions

from itertools import product

path = pathlib.Path(__file__).parent.parent.resolve()

(path / "True parameters").mkdir(parents=True, exist_ok=True)

(path / "True parameters/Plots").mkdir(parents=True, exist_ok=True)

spatial_predictors_path = path / "Data/spatial_predictors"

parameters = ["silt", "sand", "clay", "OC", "BD", "pH", "BL"]

l0 = FSO_functions.load_sp_wflow(
    path=spatial_predictors_path, parameters=parameters, scale=True, nanapprox=False
)
print("l0 loaded")

parameter_bounds = {"KsatHorFrac": [0.01, 10000]}
print("Parameter bounds loaded")

args = FSO_functions.FSO_setup()

test_number = [2]
optimizer = ["DDS"]
run = list(range(1, 2))

columns = {"test_number": test_number, "optimizer": optimizer, "run": run}
combinations = list(product(*columns.values()))

grid = pd.DataFrame(combinations, columns=columns.keys())

FSO_functions.FSO(
    path=path,
    optimizer=grid["optimizer"].loc[0],
    test_number=grid["test_number"].loc[0],
    run=grid["run"].loc[0],
    iterations=5000,
    spatial_predictors=l0,
    parameter_bounds=parameter_bounds,
    parameter_names=["KsatHorFrac"],
    **args,
)
