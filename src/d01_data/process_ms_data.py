import pandas as pd


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

        df_experiment = (df.query(query).loc[experiment['idx_range'][0]:experiment['idx_range'][1]])
        if experiment['bad_idx'] is not None:
            df_experiment = df_experiment.drop(experiment['bad_idx'])

        df_experiment['experiment_name'] = experiment_name
        df_filtered = df_filtered.append(df_experiment)

    return df_filtered


def clean_ms_data_in_experiments(df_filtered, experiment_parameters):
    """
    """

    df_cleaned = pd.DataFrame()

    for name, experiment in experiment_parameters.items():
        experiment_cleaned = df_filtered[df_filtered.experiment_name == name][experiment['columns_to_keep']]

        df_cleaned = df_cleaned.append(experiment_cleaned)

    df_cleaned.rename(columns={'trapped': 'mins'})

    return df_cleaned


def process_ms_data_in_experiments(df_cleaned, compounds):
    """
    """

    df_processed = df_cleaned.copy(deep=True)

    df_processed.rename(columns={'mins': 'hrs'})
    df_processed.hrs = df_cleaned.hrs / 60

    return df_cleaned


def add_normalized_intensity_column(df, internal_std='p283'):
    """Add "n###" columns to DataFrame of normalized peak intensities.

    """

    p_cols = [col for col in df.columns if col[0] == 'p']  # first column of mass spec peaks is p

    for tick, p_col in enumerate(p_cols):
        df['n' + p_col[1:]] = df[p_col] / df[internal_std]

    return df
