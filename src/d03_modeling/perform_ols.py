import numpy as np
import pandas as pd

from src.d00_utils.data_utils import import_treated_csv_data, save_data_frame
from src.d02_extraction.extract_least_sq_fit import *


def create_ordinary_least_squares_data(experiments_dict, experiment, x_col_name, y_col_name, subset_col_name=None,
                                       take_log=False, save_data=False):
    """ Create a dataframe of least squares data fit on experimental already processed data ('processed_data').

    :param experiments_dict: dict. Contains the information necessary for the filtering and the paths.
    :param experiment: str. Label of the experiment, the key of the experiments_dict in which to find processed data.
    :param x_col_name: str. Column name of the x data.
    :param y_col_name: str. Column name of the y data (observations).
    :param subset_col_name: ?? I think a work around if there is an issue with data selection.
    :param take_log: Boolean. If true, perform OLS on log(y). If false, perform OLS on y.
    :param save_data: Boolean. Tells you whether to save data in the experiments into the
    subdirectories of the treated_data directory.
    :return: df_ols. dataframe. Dataframe comprised of generated x values and y values from OLS fit.
    """

    file_name = experiments_dict[experiment]['paths']['processed_data']
    df = import_treated_csv_data(file_name=file_name,
                                 experiment_label=experiment)

    if subset_col_name is not None:
        df = df[subset_col_name]  # fix at later time

    x_data = df[x_col_name].values.reshape(-1, 1)
    y_data = df[y_col_name].values

    if take_log:
        lny_data = np.log(y_data)
        b0, b1 = perform_regression(x_data, lny_data)
        xs, lnyhats = generate_linear_data(x_data, b0, b1)
        yhats = np.exp(lnyhats)

    else:
        b0, b1, score = perform_regression(x_data, y_data)
        xs, yhats = generate_linear_data(x_data, b0, b1)

    df_ols = pd.DataFrame()
    df_ols[x_col_name] = xs
    df_ols[y_col_name] = yhats
    df_ols['score'] = score

    if take_log:
        tau = report_characteristic_time(b1)
        df_ols['tau'] = tau

    if save_data:
        save_data_frame(df_to_save=df_ols,
                        experiment_label=experiment,
                        level_of_treatment='OLS')

    return df_ols


def generate_linear_data(x, b0, b1):
    """Generates an array of x values and an array of y values, predicted from a linear fit."""

    xs = np.linspace(x.min(), x.max(), 100)
    b0s = np.full(100, b0)
    yhats = b0s + b1 * xs

    return xs, yhats