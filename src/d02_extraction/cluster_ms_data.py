import pandas as pd
import sklearn.cluster as cluster
import math


def add_clusters_to_dataframe(df, col_to_cluster, N_clusters):
    """"""
    data = df[[col_to_cluster]].copy()
    clusters = cluster.KMeans(N_clusters).fit_predict(data)
    df_with_clusters = df.assign(clusters=clusters)

    return df_with_clusters


def create_clustered_statistics_dataframe(df_cleaned, col_to_cluster, y_cols_to_keep, max_points_per_cluster=3):
    """"""

    cols_to_keep = y_cols_to_keep
    cols_to_keep.append(col_to_cluster)

    df_clustered_stats = pd.DataFrame()
    experiment_names = list(df_cleaned.experiment.unique())
    for experiment_name in experiment_names:
        df_exp = df_cleaned[df_cleaned.experiment == experiment_name]

        if len(df_exp) >= 4:
            N_clusters = math.floor(len(df_exp) / max_points_per_cluster)
        elif len(df_exp) < 4:
            N_clusters = len(df_exp)
        df_with_clusters = add_clusters_to_dataframe(df=df_exp,
                                                     col_to_cluster=col_to_cluster,
                                                     N_clusters=N_clusters)

        df_means = df_with_clusters.groupby('clusters', as_index=False)[cols_to_keep].mean()
        df_ses = df_with_clusters.groupby('clusters', as_index=False)[cols_to_keep].sem()
        df_combined = pd.merge(df_means, df_ses, on=None,
                               suffixes=('_mean', '_se'), left_index=True, right_index=True, how='outer')
        df_combined = df_combined.drop(columns=['clusters_mean', 'clusters_se'])
        df_combined = df_combined.assign(experiment=experiment_name)

        df_clustered_stats = df_clustered_stats.append(df_combined)

    return df_clustered_stats
