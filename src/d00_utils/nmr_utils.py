import pandas as pd
import numpy as np

def scale_nmr(df, col_name, indicator_signal):
    ''' scale to 100 '''
    df[col_name] = df[col_name] * 100 / indicator_signal
    return df


def choose_ymax_nmr_subplot(df, x_col_name, xrange):
    ''' get window for subplot based on intensity within the range of interest '''
    xmin = min(xrange)
    xmax = max(xrange)

    ymax = df.loc[(df[x_col_name] > xmin) & (df[x_col_name] < xmax)].SIG.max()

    return ymax * 1.1