import pandas as pd

from src.d00_utils.data_utils import *


def filter_raw_data_for_experiments(experiments_dict, save_filtered_data=False):
    """ Removes rows from the dataframe based on queries in the experiments dictionary from
    which processing parameters comes.

    :param experiments_dict: dict. Contains the information necessary for the filtering and the paths.
    :param save_filtered_data: Boolean. Tells you whether to save data in the experiments into the
    subdirectories of the treated_data directory.
    """

    experiment_labels = [*experiments_dict]

    for experiment in experiment_labels:
        raw_ms_file_name = experiments_dict[experiment]['paths']['raw_data']
        df_imported = import_raw_csv_data(file_name=raw_ms_file_name)

        filtering_queries = experiments_dict[experiment]['data_treatment']['filtering_queries']
        row_range_to_keep = experiments_dict[experiment]['data_treatment']['row_range_to_keep']
        rows_to_remove = experiments_dict[experiment]['data_treatment']['rows_to_remove']
        solution_id = experiments_dict[experiment]['data_treatment']['solution_id']

        df_filtered = filter_ms_data_in_experiment(df_unfiltered=df_imported,
                                                   filtering_queries=filtering_queries,
                                                   rows_to_remove=rows_to_remove,
                                                   row_range_to_keep=row_range_to_keep,
                                                   solution_id=solution_id)

        if save_filtered_data:
            save_data_frame(df_to_save=df_filtered,
                            experiment_label=experiment,
                            level_of_treatment='FILTERED')

    return


def filter_ms_data_in_experiment(df_unfiltered, filtering_queries, rows_to_remove, row_range_to_keep, solution_id):
    """ Removes rows from the dataframe based on queries in the experiments dictionary from
    which processing parameters comes. FIX THIS!

    :param df_unfiltered: df. Contains the unfiltered, raw dataset.
    :param filtering_queries: str. Contains the queries to be used to filter out data.
    :param solution_id: str. Contains the solution id which must be a match in the csv, representing the experiment.
    :return: df. Filtered dataset.
    """

    df_filtered = df_unfiltered.copy()

    if solution_id is not None:
        df_filtered = df_filtered.loc[df_filtered['SOLUTION_ID'].isin(solution_id)]

    if filtering_queries:

        if row_range_to_keep is not None:
            df_filtered = df_filtered.loc[row_range_to_keep[0]:row_range_to_keep[1]]
            df_filtered.query(filtering_queries, inplace=True)

        else:
            df_filtered.query(filtering_queries, inplace=True)

    if rows_to_remove:
        df_filtered.drop(rows_to_remove, inplace=True)

    return df_filtered
