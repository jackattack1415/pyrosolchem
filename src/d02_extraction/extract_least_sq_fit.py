import numpy as np
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit, conf_interval, Minimizer
import lmfit
from sklearn.linear_model import LinearRegression

from src.d00_utils.data_utils import save_data_frame, import_treated_csv_data
from src.d03_modeling.model_functions import *


def get_coefficients_with_lmfit(residuals_function, params, df, x_col_name, y_col_names):
    """Fit parameters (params) with lmfit minimize package. Takes in data (df) as observations for constraining,
    and residuals function (residuals_function) which calculates the residuals on params through constraining routine.
    Residuals_function contains the required model functions that explain the params.

    :param residuals_function:
    :param params: Parameters. lmfit parameters object, in which each parameter corresponds to
    a "coefficient" to be fit.
    :param df: dataframe. Contains x_col_name (time values, usually) and y_col_names (corresponding variables).
    :param x_col_name: str. Column name containing x variable. Usually column of times at which y values are measured.
    :param y_col_names: List of str. Column names containing y variables. Must be ordered as written in model functions.
    :return: results. dict? typical of lmfit, contains the fits and standard errors of fits.
    """

    xs = df[x_col_name].values
    ys = []
    for y_col in y_col_names:
        ys.append(df[y_col].values)
    ys = np.asarray(ys)

    results = minimize(residuals_function, params, args=(xs, ys), method='leastsq', nan_policy='raise')
    return results


def generate_coefficients(results_dict, N_runs=10000):
    """ Generates a dataframe of normally distributed coefficients from the std error and best fit
    returned by get_coefficients_with_lmfit.

    :param results_dict: odict. Dictionary containing the std errors and best fits from lmfitting.
    :param N_runs: number of random values to report in dataframe.
    :return: df. Dataframe of normally distributed coefficients from fitting.
    """

    df_coefs = pd.DataFrame()

    for key in results_dict.params:
        df_coefs[key] = np.random.normal(results_dict.params[key].value, results_dict.params[key].stderr, N_runs)

    df_coefs = block_coefficients_outside_confidence_interval(df_coefs=df_coefs, ci=95)

    return df_coefs


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


def perform_regression(X, y):
    """ Simple linear regression.
    """

    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)

    bs = reg.coef_

    b0 = reg.intercept_

    return b0, bs, score


def report_characteristic_time(b1):
    """ Calculates first order characteristic time from rate constant, k. """

    b1 = float(b1)
    tau = -1/b1

    return tau