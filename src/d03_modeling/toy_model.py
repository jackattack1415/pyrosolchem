import numpy as np
import pandas as pd
from scipy.integrate import odeint

from src.d00_utils.data_utils import import_treated_csv_data, save_data_frame
from src.d02_extraction.extract_least_sq_fit import get_coefficients_with_lmfit, generate_coefficients, \
    block_coefficients_outside_confidence_interval


def model_data_with_odes(residuals_function, solve_ode_function, params, experiments_dict, experiment,
                         x_col_name, y_col_names, ts, confidence_interval=False, save_data=False):
    """ Create 'modeled_data' dataframe using system of ODEs fit to measurement data from an experiment. 95% confidence
    interval can be applied.

    :param residuals_function: function. Function handle on which to calculate residuals, as per lmfit method.
    :param solve_ode_function: function. Function containing the ODEs
    :param params: Parameters. Lmfit parameter object containing the coefficients to be fit.
    :param experiments_dict: dict. Dictionary of experiments containing information about experimental parameters.
    :param experiment: str. Key of specific experiment to be selected from experiments_dict.
    :param x_col_name: str. Column label containing x data in processed_data dataframe.
    :param y_col_names: list of str. Column labels containing y data in processed_data dataframe. Must be in order of
    the ODE variables.
    :param ts: np.array. Times on which to report the model output.
    :param confidence_interval: Boolean. State whether a 95% confidence interval is desired or not.
    :param save_data: Boolean. True saves data as dataframe in data_treated, false does not.
    :return: None.
    """

    # import data to be fitted and sort for fitting as DF
    file_name = experiments_dict[experiment]['paths']['processed_data']
    df = import_treated_csv_data(file_name=file_name,
                                 experiment_label=experiment)

    df = df.sort_values(by=x_col_name, ascending=True)
    # retrieve RESULTS from the fitting onto DF
    results = get_coefficients_with_lmfit(residuals_function, params, df, x_col_name, y_col_names)

    df_coef_statistics = pd.DataFrame()
    df_coef_statistics['METRIC'] = ['avg', 'se']
    for key in results.params:
        df_coef_statistics[key] = np.array([results.params[key].value, results.params[key].stderr])

    # synthesize modeled data from RESULTS
    df_out = pd.DataFrame()
    df_out[x_col_name] = ts

    if confidence_interval is True:
        # synthesize gaussian distribution of coefficients, as well as statistics for N_runs iterations
        # monte carlo style uncertainty estimation
        df_coefs = generate_coefficients(results_dict=results, N_runs=10000)
        solutions_array = np.empty([len(df_coefs), len(ts), len(y_col_names)])

        # execute odes to get data for each row of synthesized coefficients, N_run iterations
        for tick in range(len(df_coefs)):
            coefficients = df_coefs.iloc[tick].values[1:]
            solutions = solve_ode_function(ts, coefficients)
            if solutions.ndim == 1:
                solutions_array[tick, :, :] = np.reshape(solve_ode_function(ts, coefficients), (-1, 1))
            else:
                solutions_array[tick, :, :] = solve_ode_function(ts, coefficients)

        # report the average values for each Y_COL_NAME for list of ts
        solutions_avg = np.median(solutions_array, 0)

        for tick in range(len(y_col_names)):
            y_col_name = y_col_names[tick]
            y_col_name_avg = y_col_name
            df_out[y_col_name_avg] = solutions_avg[:, tick].reshape(-1)

            # get the confidence range (reported as min and max) for each Y_COL_NAME for list of ts
            solutions_min = np.min(solutions_array, 0)
            solutions_max = np.max(solutions_array, 0)
            y_col_name_max = y_col_name + '_MAX'
            y_col_name_min = y_col_name + '_MIN'
            df_out[y_col_name_min] = solutions_min[:, tick].reshape(-1)
            df_out[y_col_name_max] = solutions_max[:, tick].reshape(-1)

    elif confidence_interval is False:
        coefficients = df_coef_statistics.loc[0].values[1:]
        coefficients = np.array(coefficients, dtype='float64')
        solutions = solve_ode_function(ts, coefficients)

        if len(solutions.shape) == 1:
            solutions = np.reshape(solutions, (solutions.size, 1))

        for tick in range(len(y_col_names)):
            y_col_name = y_col_names[tick]
            df_out[y_col_name] = solutions[:, tick].reshape(-1)

    # save data to corresponding experiment folder in "data_treated"
    if save_data:
        save_data_frame(df_to_save=df_out,
                        experiment_label=experiment,
                        level_of_treatment='LMFIT')

        save_data_frame(df_to_save=df_coef_statistics,
                        experiment_label=experiment,
                        level_of_treatment='LMFIT_PARAMS')


def predict_data_with_odes(solve_ode_function, ts, coefs, experiment, x_col_name, y_col_names,
                           confidence_interval=False, save_data=True):
    """ Create 'predicted_data' dataframe using system of ODEs ALREADY fit to measurement data from an experiment.
    95% confidence interval can be applied. Unlike model_data_with_odes, does not perform fit.

    :param solve_ode_function: function. Function containing the ODEs
    :param coefs: list of lists of floats or list of floats. List of floats contains fitted coefficients for the
    solve_ode_function. Must be pared with confidence_interval=False. In other case, each sub list contains avg, std err
    of coefficients. Must be paired with confidence_interval=True.
    :param experiment: str. Key of specific experiment to be selected from experiments_dict.
    :param x_col_name: str. Column label containing x data in processed_data dataframe.
    :param y_col_names: list of str. Column labels containing y data in processed_data dataframe. Must be in order of
    the ODE variables.
    :param confidence_interval: Boolean. State whether a 95% confidence interval is desired or not.
    :param save_data: Boolean. True saves data as dataframe in data_treated, false does not.
    :return: None.
    """

    # import data to be fitted and sort for fitting as DF
    df_out = pd.DataFrame()
    df_out[x_col_name] = ts

    if confidence_interval is True:
        # monte carlo style uncertainty estimation
        df_coefs = pd.DataFrame()

        # synthesize gaussian distribution of coefficients, as well as statistics for N_runs iterations
        count = 0
        for coef in coefs:
            mean = coef[0]
            stderr = coef[1]
            df_coefs['coef_' + str(count)] = np.random.normal(mean, stderr, 10000)
            count += 1

        df_coefs = block_coefficients_outside_confidence_interval(df_coefs=df_coefs, ci=95)

        # execute odes to get data for each row of synthesized coefficients, N_run iterations
        solutions_array = np.empty([len(df_coefs), len(ts), len(y_col_names)])

        for tick in range(len(df_coefs)):
            coefficients = df_coefs.iloc[tick].values[1:]
            solutions = solve_ode_function(ts, coefficients)
            if solutions.ndim == 1:
                solutions_array[tick, :, :] = np.reshape(solve_ode_function(ts, coefficients), (-1, 1))
            else:
                solutions_array[tick, :, :] = solve_ode_function(ts, coefficients)

        # report the average values for each Y_COL_NAME for list of ts
        solutions_avg = np.median(solutions_array, 0)

        for tick in range(len(y_col_names)):
            y_col_name = y_col_names[tick]
            y_col_name_avg = y_col_name
            df_out[y_col_name_avg] = solutions_avg[:, tick].reshape(-1)

            # get the confidence range (reported as min and max) for each Y_COL_NAME for list of ts
            solutions_min = np.min(solutions_array, 0)
            solutions_max = np.max(solutions_array, 0)
            y_col_name_max = y_col_name + '_MAX'
            y_col_name_min = y_col_name + '_MIN'
            df_out[y_col_name_min] = solutions_min[:, tick].reshape(-1)
            df_out[y_col_name_max] = solutions_max[:, tick].reshape(-1)

    elif confidence_interval is False:
        coefs = np.array(coefs, dtype='float64')
        solutions = solve_ode_function(ts, coefs)

        if len(solutions.shape) == 1:
            solutions = np.reshape(solutions, (solutions.size, 1))

        for tick in range(len(y_col_names)):
            y_col_name = y_col_names[tick]
            df_out[y_col_name] = solutions[:, tick].reshape(-1)

    # save data to corresponding experiment folder in "data_treated"
    if save_data:
        save_data_frame(df_to_save=df_out,
                        experiment_label=experiment,
                        level_of_treatment='PREDICTED')
