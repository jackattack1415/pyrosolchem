import pandas as pd
import sklearn.cluster as cluster
import math

from src.d00_utils.data_utils import save_data_frame, import_treated_csv_data


def cluster_data_for_experiments(experiments_dict, max_points_per_cluster=3, save_clustered_data=False):
    """ Removes rows from the dataframe based on queries in the experiments dictionary from
    which processing parameters comes.

    :param experiments_dict: dict. Contains the information necessary for the filtering and the paths.
    :param save_clustered_data: Boolean. Tells you whether to save data in the experiments into the
    subdirectories of the treated_data directory.
    """

    experiment_labels = [*experiments_dict]

    for experiment in experiment_labels:
        if 'clustered_data' in experiments_dict[experiment]['paths']:
            file_name = experiments_dict[experiment]['paths']['processed_data']

            df_processed = import_treated_csv_data(file_name=file_name,
                                                   experiment_label=experiment)

            df_clustered_stats = pd.DataFrame()
            if len(df_processed) >= 4:
                N_clusters = math.floor(len(df_processed) / max_points_per_cluster)
            elif len(df_processed) < 4:
                N_clusters = len(df_processed)

            df_with_clusters = add_clusters_to_dataframe(df=df_processed,
                                                         N_clusters=N_clusters)

            df_means = df_with_clusters.groupby('clusters', as_index=False).mean()
            df_stds = df_with_clusters.groupby('clusters', as_index=False).std()
            df_combined = pd.merge(df_means, df_stds, on=None,
                                   suffixes=('', '_std'), left_index=True, right_index=True, how='outer')
            df_combined = df_combined.drop(columns=['clusters', 'clusters_std'])
            df_combined = df_combined.assign(experiment=experiment)

            df_clustered_stats = df_clustered_stats.append(df_combined)

            df_clustered_stats.dropna(0, inplace=True)

            if save_clustered_data:
                save_data_frame(df_to_save=df_clustered_stats,
                                experiment_label=experiment,
                                level_of_treatment='CLUSTERED')

    return


def add_clusters_to_dataframe(df, N_clusters):
    """ Adds a column of labels, corresponding to cluster assignment, to dataframe.
    Nested within cluster_data_for_experiments.

    :param df: dataframe.
    :param N_clusters: float. Integer number of clusters to assign rows.

    :return df_with_clusters. dataframe. df with additional row with 'clusters' column.
    """

    clusters = cluster.KMeans(N_clusters).fit_predict(df)
    df_with_clusters = df.assign(clusters=clusters)

    return df_with_clusters
