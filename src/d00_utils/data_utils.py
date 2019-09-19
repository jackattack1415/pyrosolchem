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
