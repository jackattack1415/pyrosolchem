import os
import pandas as pd
import numpy as np
from datetime import date

from src.d00_utils.conf_utils import get_project_directory
from src.d00_utils.calc_utils import convert_mass_to_molar_composition
from src.d00_utils.processing_utils import perform_bootstrap


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


def extract_calibration_data(df, t0_cutoff):
    """

    :param df:
    :param data_col:
    :param t0_cutoff:
    :return:
    """

    calibration_data_query = "trapped<={}".format(t0_cutoff)
    data_for_calibration = df.query(calibration_data_query)

    return data_for_calibration


def convert_normalized_ms_signals_to_molar_ratio(df, norm_ms_signal_col, experiments, compounds,
                                                 starting_molar_composition, internal_standard='PEG-6'):

    data_for_calibration = extract_calibration_data(df, experiments['cal_data_time'])

    ms_signals = perform_bootstrap(data_for_calibration[norm_ms_signal_col])
    ms_signal_means = np.mean(ms_signals, axis=0)
    avg_ms_signal = np.mean(ms_signal_means)
    std_ms_signal_means = np.std(ms_signal_means)

    # relative standard deviation
    rel_std_ms_signal = std_ms_signal_means / avg_ms_signal

    # experimental molar ratio relative to internal standard
    molar_composition = convert_mass_to_molar_composition(compounds, mass_composition)

    mole_ratio = molar_composition[compounds]
    # Get conversion factor: (mole ratio) / (bootstrapped MS signal ratio)
    # defined as distribution from bootstrapping with mean and std
    scale_avg = mole_ratio / avg_ms_signal
    scale_std = rel_std_ms_signal * scale_avg

    return scale_avg, scale_std


def save_data_frame(df_to_save, raw_data_path, level_of_cleaning):
    """

    :param df_to_save:
    :param raw_data_path:
    :param level_of_cleaning:
    :return:
    """

    project_dir = get_project_directory()
    today_str = date.today().strftime("%Y%m%d")

    file_name = today_str[2:] + '-' + raw_data_path.split('-')[1:] + '-' + level_of_cleaning + '.csv'
    file_path = os.path.join(project_dir, 'data', file_name)

    df_to_save.to_csv(path_or_buf=file_path, index=False)
