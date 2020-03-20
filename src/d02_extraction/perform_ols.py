import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.d00_utils.data_utils import import_ms_data, save_data_frame


def create_ordinary_least_squares_data(processed_ms_file_name, experiments_dict, x_col_name, y_col_name,
                                       take_log=False, save_data=False, experiment=None):

    experiment_name = [*experiments_dict].pop()
    df = import_ms_data(file_name=processed_ms_file_name,
                        subdirectory=experiment_name)

    if experiment:
        df = df[df.experiment == experiment]

    x_data = df[x_col_name].values.reshape(-1, 1)

    y_data = df[y_col_name].values
    if take_log:
        lny_data = np.log(y_data)
        b0, b1, score = perform_regression(x_data, lny_data)
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
                        experiment_label=experiment_name,
                        level_of_cleaning='OLS')

    return df_ols


def perform_regression(X, y):
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)

    bs = reg.coef_

    b0 = reg.intercept_

    return b0, bs, score


def generate_linear_data(x, b0, b1):
    xs = np.linspace(x.min(), x.max(), 100)
    b0s = np.full(100, b0)
    yhats = b0s + b1 * xs

    return xs, yhats


def report_characteristic_time(b1):
    b1 = float(b1)
    tau = -1/b1

    return tau
