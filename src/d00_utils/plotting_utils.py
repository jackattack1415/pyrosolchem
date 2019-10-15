import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_ms_data(df_data, x_data_col, y_data_cols,
                 series_labels, series_colors, ax=None,
                 df_model=None, x_model_col=None, y_model_cols=None, model_label=None,
                 df_cluster=None, x_cluster_col=None, y_cluster_cols=None):
    """
    """

    sns.set(style="ticks")
    sns.set_context("talk")

    if ax is not None:
        ax = ax
    else:
        fig, ax = plt.subplots()

    for tick in range(len(y_data_cols)):
        xs = df_data[x_data_col]
        ys = df_data[y_data_cols[tick]]

        if df_cluster is not None:
            x_clusters = df_cluster[x_cluster_col]
            y_clusters = df_cluster[y_cluster_cols[tick]]
            x_err_col = x_cluster_col + '_std'
            y_err_col = y_cluster_cols[tick] + '_std'
            x_errs = df_cluster[x_err_col]
            y_errs = df_cluster[y_err_col]

            ax.errorbar(x=x_clusters, y=y_clusters, xerr=x_errs, yerr=y_errs,
                        ls='', marker='o', markersize=10, markeredgecolor='black',
                        capsize=3, capthick=2, ecolor='black',
                        color=series_colors[tick], label='%s (clustered)' % series_labels[tick])

            ax.scatter(xs, ys, label='%s (observed)' % series_labels[tick], color=series_colors[tick],
                       s=200, alpha=0.2, edgecolor='black')
        else:
            ax.scatter(xs, ys, label='%s' % series_labels[tick], color=series_colors[tick],
                       s=200, alpha=1, edgecolor='black')

        if df_model is not None:
            xs_model = df_model[x_model_col]
            ys_model = df_model[y_model_cols[tick]]
            if model_label is not None:
                ax.plot(xs_model, ys_model, lw=2, color=series_colors[tick], label='%s' % model_label)
            else:
                ax.plot(xs_model, ys_model, lw=2, color=series_colors[tick],
                        label='%s (predicted)' % series_labels[tick])

    return ax


def get_plot_limits(df_data, x_data_col, y_data_cols,
                    df_cluster=None, x_cluster_col=None, y_cluster_cols=[None],
                    df_model=None, x_model_col=None, y_model_cols=[None]):

    # find x and y max
    x_max_data = x_max_model = x_max_cluster = 0
    y_max_data = y_max_model = y_max_cluster = 0
    x_max_data = df_data[x_data_col].max()
    y_max_data = np.nanmax(df_data[y_data_cols].max().values)

    if df_model is not None:
        x_max_model = df_model[x_model_col].max()
        y_max_model = np.nanmax(df_model[y_model_cols].max().values)
    if df_cluster is not None:
        x_max_cluster = df_cluster[x_cluster_col].max()
        y_max_cluster = np.nanmax(df_cluster[y_cluster_cols].max().values)

    x_max = np.nanmax([x_max_data, x_max_model, x_max_cluster])
    y_max = np.nanmax([y_max_data, y_max_model, y_max_cluster])

    return x_max, y_max


def format_plot():

    return ax