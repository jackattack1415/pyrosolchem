import pandas as pd

from src.d00_utils.data_utils import import_ms_data, save_data_frame


def filter_and_clean_data_in_experiment(ms_file_name, experiments_dict, save_cleaned_data=False):
    """"""

    experiment_name = [*experiments_dict].pop()
    df_imported = import_ms_data(file_name=ms_file_name)

    df_filtered = filter_ms_data_in_experiment(df=df_imported,
                                               processing_parameters=experiments_dict[experiment_name]['processing'])

    df_cleaned = clean_ms_data_in_experiment(df=df_filtered,
                                             processing_parameters=experiments_dict[experiment_name]['processing'])

    if save_cleaned_data:
        save_data_frame(df_to_save=df_cleaned,
                        experiment_label=experiment_name,
                        level_of_cleaning='CLEANED')

    return df_cleaned


def filter_ms_data_in_experiment(df, processing_parameters):
    """
    """

    query_parts = []
    if processing_parameters['trap_time']:
        query_parts.append("trapped>={} and trapped<{}".format(*processing_parameters['trap_time']))

    if processing_parameters['other_query']:
        query_parts.append(processing_parameters['other_query'])

    query = " and ".join(query_parts)

    if processing_parameters['idx_range']:
        df_filtered = (df.query(query).loc[
                       processing_parameters['idx_range'][0]:processing_parameters['idx_range'][1]])
    else:
        df_filtered = (df.query(query))

    if processing_parameters['bad_idx']:
        df_filtered = df_filtered.drop(processing_parameters['bad_idx'])

    return df_filtered


def clean_ms_data_in_experiment(df, processing_parameters):
    """
    """

    df = add_normalized_intensity_column(df)  # make this more general eventually to non p283...

    df_cleaned = pd.DataFrame()

    experiment_cleaned = df[processing_parameters['columns_to_keep']]
    df_cleaned = df_cleaned.append(experiment_cleaned)

    df_cleaned = df_cleaned.rename(columns={'trapped': 'mins', 'comp': 'solution_name'})

    return df_cleaned


def add_normalized_intensity_column(df, internal_std_col='p283'):
    """Add "mz###_mz###" columns to DataFrame of normalized peak intensities.
    """

    p_cols = [col for col in df.columns if col[0] == 'p']  # first column of mass spec peaks is p

    for tick, p_col in enumerate(p_cols):
        df['mz' + p_col[1:] + '_mz' + internal_std_col[1:]] = df[p_col] / df[internal_std_col]

    return df
