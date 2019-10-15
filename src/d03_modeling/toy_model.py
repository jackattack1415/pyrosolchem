import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def run_toy_model(toy_model, y0s, ts, y_labels, time_step_label, scaling=None):

    results = odeint(toy_model, y0=y0s, t=ts)

    df_model = pd.DataFrame()
    df_model[time_step_label]

    for tick in range(len(y_labels)):
        if scaling:
            results[:, tick] = results[:, tick] * scaling[tick]
        df_model[y_labels[tick]] = results[:, tick]

    return df_model
