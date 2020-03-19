import pandas as pd

from src.d00_utils.data_utils import import_ms_data, save_data_frame


def filter_raw_data_for_experiments(experiments_dict, save_filtered_data=False):
    """ Removes rows from the dataframe based on queries in the experiments dictionary from
    which processing parameters comes.

    :param experiments_dict: dict. Contains the information necessary for the filtering and the paths.
    :param save_filtered_data: Boolean. Tells you whether to save data in the experiments into the
    subdirectories of the treated_data directory.
    """

    experiment_labels = [*experiments_dict].pop()

    for experiment in experiment_labels:
        raw_ms_file_name = experiment_dict[experiment]['paths']['raw_data']
        df_imported = import_raw_csv_data(file_name=raw_ms_file_name)

        filtering_queries = experiment_dict[experiment]['data_treatment']['filtering_queries']
        solution_id = experiment_dict[experiment]['data_treatment']['solution_id']

        df_filtered = filter_ms_data_in_experiment(df_unfiltered=df_imported,
                                                   filtering_queries=filtering_queries,
                                                   solution_id=solution_id)

        if save_filtered_data:
            save_data_frame(df_to_save=df_filtered,
                            experiment_label=experiment_name,
                            level_of_cleaning='FILTERED')

    return


def filter_ms_data_in_experiment(df_unfiltered, filtering_queries, solution_id):
    """ Removes rows from the dataframe based on queries in the experiments dictionary from
    which processing parameters comes. FIX THIS!

    :param df_unfiltered: df. Contains the unfiltered, raw dataset.
    :param filtering_queries: str. Contains the queries to be used to filter out data.
    :param solution_id: str. Contains the solution id which must be a match in the csv, representing the experiment.
    :return: df. Filtered dataset.
    """

    df_filtered = df_unfiltered[SOLUTION_ID == solution_id]

    if filtering_queries is not null:

        query_parts = []
        if processing_parameters['trap_time']:
            query_parts.append("trapped>={} and trapped<{}".format(*processing_parameters['trap_time']))

        if processing_parameters['other_query']:
            query_parts.append(processing_parameters['other_query'])

        query = " and ".join(query_parts)

        if processing_parameters['idx_range']:
            df_filtered = (df_filtered.query(query).loc[
                           processing_parameters['idx_range'][0]:processing_parameters['idx_range'][1]])
        else:
            df_filtered = (df.query(query))

        if processing_parameters['bad_idx']:
            df_filtered = df_filtered.drop(processing_parameters['bad_idx'])

    if save_cleaned_data:
        save_data_frame(df_to_save=df_filtered,
                        experiment_label=experiment_name,
                        level_of_cleaning='FILTERED')

    return df_filtered
