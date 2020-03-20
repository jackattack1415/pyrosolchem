import numpy as np
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit

from src.d00_utils.data_utils import save_data_frame, import_treated_csv_data
from src.d03_modeling.model_functions import *


def get_coefficients_with_lmfit(residuals_function, params, df, x_col_name, y_col_names):
    """"""

    xs = df[x_col_name].values
    ys = []
    for y_col in y_col_names:
        ys.append(df[y_col].values)
    ys = np.asarray(ys)

    results = minimize(residuals_function, params, args=(xs, ys), method='leastsq', nan_policy='omit')

    return results


def generate_coefficients(results_dict, N_runs=10000):
    """ Generates a dataframe of normally distributed coefficients from the std error and best fit
    returned by get_coefficients_with_lmfit.

    :param results_dict: odict. Dictionary containing the std errors and best fits from lmfitting.
    :param N_runs: number of random values to report in dataframe.
    :return: df. Dataframe of normally distributed coefficients from fitting.
    """

    df_coefs = pd.DataFrame()
    df_coef_statistics = pd.DataFrame()
    df_coef_statistics['METRIC'] = ['avg', 'se']

    for key in results_dict.params:
        df_coefs[key] = np.random.normal(results_dict.params[key].value, results_dict.params[key].stderr, N_runs)
        df_coef_statistics[key] = np.array([results_dict.params[key].value, results_dict.params[key].stderr])

    df_coefs = block_coefficients_outside_confidence_interval(df_coefs=df_coefs, ci=95)

    return df_coefs, df_coef_statistics


def block_coefficients_outside_confidence_interval(df_coefs, ci=95):
    """ Retains only the parameters that fit within the prescribed level of confidence.

    :param df_coefs: df. Total coefficients unscreened in confidence interval.
    :param ci: float. Confidence interval in percentage.
    :return: df. Screened dataframe with only the coefficients within confidence interval.
    """

    lower_percentile = 0.5 - (ci * 0.01 / 2)
    upper_percentile = 0.5 + (ci * 0.01 / 2)

    lower_bound = df_coefs.quantile(lower_percentile)
    higher_bound = df_coefs.quantile(upper_percentile)

    df_coefs_ci = df_coefs[~((df_coefs < lower_bound) | (df_coefs > higher_bound)).any(axis=1)]
    df_coefs_ci.reset_index(inplace=True)

    return df_coefs_ci
