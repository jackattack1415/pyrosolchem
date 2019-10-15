import os
import pandas as pd
from datetime import date

from src.d00_utils.conf_utils import get_project_directory


def import_ms_data(file_name, subdirectory=None):
    """

    :param file_name:
    :param subdirectory:
    :return:
    """

    project_dir = get_project_directory()
    if subdirectory is not None:
        data_path = os.path.join(project_dir, 'data', subdirectory, file_name)
    else:

        data_path = os.path.join(project_dir, 'data', file_name)

    df = pd.read_csv(data_path)

    return df


def extract_calibration_data(df, t_init_cutoff, cal_data_col):
    """"""

    calibration_data_query = "mins<={}".format(t_init_cutoff)
    ms_signal_inits = df.query(calibration_data_query)[cal_data_col].values

    return ms_signal_inits


def save_data_frame(df_to_save, experiment_label, level_of_cleaning):
    """

    :param df_to_save:
    :param experiment_name:
    :param level_of_cleaning:
    :return:
    """

    project_dir = get_project_directory()
    today_str = date.today().strftime("%Y%m%d")
    file_dir = os.path.join(project_dir, 'data', experiment_label)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    file_name = today_str[2:] + '-' + experiment_label + '-' + level_of_cleaning + '.csv'
    file_path = os.path.join(file_dir, file_name)

    df_to_save.to_csv(path_or_buf=file_path, index=False)
