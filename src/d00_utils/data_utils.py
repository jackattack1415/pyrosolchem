import os
import pandas as pd

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
