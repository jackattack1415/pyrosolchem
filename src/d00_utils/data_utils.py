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


def add_normalized_intensity_column(df, internal_std='p283'):
    """Add "n###" columns to DataFrame of normalized peak intensities.

    """

    p_cols = [col for col in df.columns if col[0] == 'p']  # first column of mass spec peaks is p

    for tick, p_col in enumerate(p_cols):
        df['n' + p_col[1:]] = df[p_col] / df[internal_std]

    return df


def filter_ms_data_in_experiments(df, experiment_parameters):
    """
    """

    df_filtered = pd.DataFrame()
    for experiment_name, experiment in experiment_parameters.items():
        query_parts = []
        query_parts.append("comp == '{}'".format(experiment['solution_name']))
        query_parts.append("trapped>={} and trapped<{}".format(*experiment['trap_time']))

        if experiment['other_query'] is not None:
            query_parts.append(experiment['other_query'])

        query = " and ".join(query_parts)

        df_experiment = (df.query(query).
                             loc[experiment['idx_range'][0]:experiment['idx_range'][1]])
        if experiment['bad_idx'] is not None:
            df_experiment = df_experiment.drop(experiment['bad_idx'])

        df_filtered = df_filtered.append(df_experiment)

    return df_filtered


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