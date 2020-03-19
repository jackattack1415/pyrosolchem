import os
import pandas as pd
from datetime import date

from src.d00_utils.conf_utils import get_project_directory


def import_raw_csv_data(file_name):
    """ Imports the csv file as a dataframe from the data_raw directory.

    :param file_name: str. Only the file name (not the path) of the raw csv file, located in data_raw folder.
    :return: df. Contains the data embedded in the csv of the file name selected.
    """

    project_dir = get_project_directory()
    data_path = os.path.join(project_dir, 'data_raw', 'csvs', file_name)
    df = pd.read_csv(data_path)

    return df


def import_treated_csv_data(file_name, experiment_label):
    """ Imports the csv file as a dataframe from the data_treated directory.

    :param file_name: str. Only the file name (not the path) of the raw csv file, located in data_treated.
    :param experiment_label: str. The experiment shorthand used additionally as the directory name in data_treated.
    :return: df. Contains the data embedded in the csv of the file name selected.
    """

    project_dir = get_project_directory()
    data_path = os.path.join(project_dir, 'data_treated', experiment_label, file_name)
    df = pd.read_csv(data_path)

    return df


def extract_calibration_data(df, t_init_cutoff, cal_data_col):
    """"""

    calibration_data_query = "mins<={}".format(t_init_cutoff)
    ms_signal_inits = df.query(calibration_data_query)[cal_data_col].values

    return ms_signal_inits


def save_data_frame(df_to_save, experiment_label, level_of_treatment):
    """ Saves cleaned data frame under experiment subdirectory in the data_treated directory.

    :param df_to_save: df. Dataframe that will be saved.
    :param experiment_label: str. Name of the experiment to which the dataframe belongs.
    :param level_of_treatment: str. Level of cleaning done to df (FILTERED, PROCESSED, CLUSTERED, MODELED)
    :return: None
    """

    project_dir = get_project_directory()
    today_str = date.today().strftime("%Y%m%d")
    file_dir = os.path.join(project_dir, 'data', experiment_label)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    file_name = today_str + '_' + experiment_label + '_' + level_of_treatment + '.csv'
    file_path = os.path.join(file_dir, file_name)

    df_to_save.to_csv(path_or_buf=file_path, index=False)
