import os
import pandas as pd
from datetime import date

from src.d00_utils.conf_utils import get_project_directory


def import_ms_data(file_name, directory=None):
    """

    :param file_name:
    :param directory:
    :return:
    """

    if directory is not None:
        data_path = os.path.join(directory, file_name)
    else:
        project_dir = get_project_directory()
        data_path = os.path.join(project_dir, 'data', file_name)

    df = pd.read_csv(data_path)

    return df


def extract_calibration_data(df, t_init_cutoff, cal_data_col):
    """"""

    calibration_data_query = "mins<={}".format(t_init_cutoff)
    ms_signal_inits = df.query(calibration_data_query)[cal_data_col].values

    return ms_signal_inits


def save_data_frame(df_to_save, raw_data_file_name, level_of_cleaning):
    """

    :param df_to_save:
    :param raw_data_file_name:
    :param level_of_cleaning:
    :return:
    """

    project_dir = get_project_directory()
    today_str = date.today().strftime("%Y%m%d")

    file_name = today_str[2:] + '-' + raw_data_file_name.split('-')[1:] + '-' + level_of_cleaning + '.csv'
    file_path = os.path.join(project_dir, 'data', file_name)

    df_to_save.to_csv(path_or_buf=file_path, index=False)
