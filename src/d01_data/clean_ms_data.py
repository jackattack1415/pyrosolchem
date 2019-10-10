import pandas as pd

from src.d00_utils.data_utils import import_ms_data, save_data_frame


def filter_and_clean_data(ms_file_name, experiment_dict, save_cleaned_data=False):
    """"""

    experiment_name = [*experiment_dict].pop()
    df_imported = import_ms_data(file_name=ms_file_name,
                                 directory=None)

    df_filtered = filter_ms_data_in_experiments(df=df_imported,
                                                experiment_parameters=experiment_dict)

    df_cleaned = clean_ms_data_in_experiments(df=df_filtered,
                                              experiment_parameters=experiment_dict)

    if save_cleaned_data:
        save_data_frame(df_to_save=df_cleaned,
                        experiment_label=experiment_name,
                        level_of_cleaning='CLEANED')

    return df_cleaned


def filter_ms_data_in_experiments(df, experiment_parameters):
    """
    """

    df_filtered = pd.DataFrame()
    for experiment_name, experiment in experiment_parameters.items():
        query_parts = []
        if experiment['trap_time']:
            query_parts.append("trapped>={} and trapped<{}".format(*experiment['trap_time']))

        if experiment['other_query']:
            query_parts.append(experiment['other_query'])

        query = " and ".join(query_parts)

        if experiment['idx_range']:
            df_experiment = (df.query(query).loc[experiment['idx_range'][0]:experiment['idx_range'][1]])
        else:
            df_experiment = (df.query(query))

        if experiment['bad_idx']:
            df_experiment = df_experiment.drop(experiment['bad_idx'])

        df_experiment = df_experiment.assign(experiment=experiment_name)
        df_filtered = df_filtered.append(df_experiment)

    return df_filtered


def clean_ms_data_in_experiments(df, experiment_parameters):
    """
    """

    df = add_normalized_intensity_column(df)  # make this more general eventually to non p283...

    df_cleaned = pd.DataFrame()

    for name, experiment in experiment_parameters.items():
        experiment_cleaned = df[df.experiment == name][experiment['columns_to_keep']]

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
