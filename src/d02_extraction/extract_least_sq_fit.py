import numpy as np
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

from src.d00_utils.data_utils import save_data_frame, import_ms_data


def model_data_with_odes(g_function, residual_function, params, experiments_dict,
                         x_col_name, y_col_names, ts, vars_init, save_data=False):

    experiment_name = [*experiments_dict].pop()
    df = import_ms_data(file_name=experiments_dict[experiment_name]['paths']['processed_data'],
                        subdirectory=experiment_name)

    df = df.sort_values(by=x_col_name, ascending=True)

    coefs = get_coefficients_with_lmfit(residual_function, params, df, x_col_name, y_col_names)
    data_array = g_function(ts, vars_init, coefs)

    df_out = pd.DataFrame()
    df_out[x_col_name] = ts
    count = 0
    for y_col_name in y_col_names:
        df_out[y_col_name] = data_array[:, count]
        count += 1

    if save_data:
        save_data_frame(df_to_save=df_out,
                        experiment_label=experiment_name,
                        level_of_cleaning='LMFIT')

    return df_out


def get_coefficients_with_lmfit(f_residual, params, df, x_col_name, y_col_names):
    """"""
    xs = df[x_col_name].values
    ys = []
    for y_col in y_col_names:
        ys.append(df[y_col].values)
    ys = np.asarray(ys)

    results = minimize(f_residual, params, args=(xs, ys), method='leastsq')

    return results.params


def g_bd_pyr_dia(t, y0, coefs):
    """
    Solution to the ODE y'(t) = f(t,y,k) with initial condition y0
    """

    y = odeint(f_bd_pyr_dia, y0, t, args=(coefs,))

    return y


def f_residual(coefs, t, data):
    """
    compute the residual between actual data and fitted data
    """

    y0 = coefs['P_0'].value, coefs['B_0'].value, coefs['D_0'].value
    model = g_bd_pyr_dia(t, y0, coefs).T

    return (model - data).ravel()


def f_bd_pyr_dia(y, t, coefs):
    """
    Your system of differential equations
    """

    P = y[0]
    B = y[1]
    D = y[2]

    k0 = coefs['k0'].value
    k1 = coefs['k1'].value
    k_LD = coefs['k_LD'].value
    a = coefs['a'].value
    b = coefs['b'].value

    k_LB = 1 / (60 * 1.4)
    k_LP = 1 / (60 * 4.6)

    # the model equations
    dPdt = a * (k0 * B * B - k1 * B * P - k_LP * P)
    dBdt = -2 * k0 * B * B - k1 * B * P - k_LB * B
    dDdt = b * (k1 * B * P - k_LD * D)

    return [dPdt, dBdt, dDdt]
