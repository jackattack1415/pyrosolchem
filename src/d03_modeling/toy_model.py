import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from src.d00_utils.data_utils import import_treated_csv_data, save_data_frame
from src.d03_modeling.model_functions import g
from src.d02_extraction.extract_least_sq_fit import get_coefficients_with_lmfit, generate_coefficients


def model_data_with_odes(f_function, residuals_function, solve_ode_function, params, experiments_dict, experiment,
                         x_col_name, y_col_names, ts, vars_init, confidence_interval=False, save_data=False):
    """"""

    file_name = experiments_dict[experiment]['paths']['processed_data']
    df = import_treated_csv_data(file_name=file_name,
                                 experiment_label=experiment)

    df = df.sort_values(by=x_col_name, ascending=True)

    results = get_coefficients_with_lmfit(residuals_function, params, df, x_col_name, y_col_names)

    data_array = g(ts, vars_init, results.params, f_function)
    df_out = pd.DataFrame()
    df_out[x_col_name] = ts
    count = 0
    for y_col_name in y_col_names:
        df_out[y_col_name] = data_array[:, count]
        count += 1

    if confidence_interval:  # need to make this usable for multiple variables (3d)

        df_coefs, df_coef_statistics = generate_coefficients(results_dict=results, N_runs=10000)
        solutions_array = np.empty([len(df_coefs), len(ts)])

        for tick in range(len(df_coefs)):
            coefficients = df_coefs.iloc[tick].values[1:]
            solutions_array[tick, :] = solve_ode_function(ts, coefficients)

        solutions_min = np.min(solutions_array, 0)
        solutions_max = np.max(solutions_array, 0)

        y_col_name_max = y_col_name + '_MAX'
        y_col_name_min = y_col_name + '_MIN'

        df_out[y_col_name_min] = solutions_min
        df_out[y_col_name_max] = solutions_max

    if save_data:
        save_data_frame(df_to_save=df_out,
                        experiment_label=experiment,
                        level_of_treatment='LMFIT')

        save_data_frame(df_to_save=df_coef_statistics,
                        experiment_label=experiment,
                        level_of_treatment='LMFIT_PARAMS')