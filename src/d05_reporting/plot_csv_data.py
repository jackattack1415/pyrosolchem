import os
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

from src.d00_utils.conf_utils import get_project_directory
from src.d00_utils.data_utils import import_treated_csv_data
from src.d00_utils.plotting_utils import get_plot_limits


def plot_csv_data(df_data, x_data_col, y_data_cols, series_labels, series_colors, series_markers=['o'], ax=None,
                  df_model=None, x_model_col=None, y_model_cols=None, model_label=None,
                  df_cluster=None, x_cluster_col=None, y_cluster_cols=None):
    """
    """

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

            if series_labels[tick] is not None:
                label = series_labels[tick]
            else:
                label = 'Observations'

            ax.errorbar(x=x_clusters.values, y=y_clusters.values, xerr=x_errs.values, yerr=y_errs.values,
                        ls='', marker='o', markersize=10, markeredgecolor='0.25',
                        capsize=3, capthick=2, ecolor='0.25',
                        color=series_colors[tick], label='Clustered ' + label, zorder=3)

            ax.scatter(xs, ys, label=label, facecolors=series_colors[tick],
                       marker=series_markers[tick], s=25, alpha=1, zorder=2)
        else:
            if series_labels is None:
                ax.scatter(xs, ys, label='Observations', facecolors=series_colors[tick], edgecolor='0.25',
                           marker=series_markers[tick], s=50, alpha=1)
            else:
                ax.scatter(xs, ys, label=series_labels[tick], facecolors=series_colors[tick], edgecolor='0.25',
                           marker=series_markers[tick], s=50, alpha=1)



        if df_model is not None:
            xs_model = df_model[x_model_col]
            ys_model = df_model[y_model_cols[tick]]
            if model_label is not None:
                ax.plot(xs_model, ys_model, label='%s' % model_label,
                        lw=3, ls='-', alpha=0.4, color=series_colors[tick])
            else:
                ax.plot(xs_model, ys_model, lw=3, ls='-', alpha=0.4, color=series_colors[tick], zorder=1)

    return ax


def plot_csv_data_across_experiments(experiments_dict, experiment_names, x_col_name, y_col_names,
                                     series_labels, series_colors, series_markers, x_label, y_label,
                                     series_title=None, save_fig=False, ax=None, legend=True,
                                     add_clusters=False, add_modeled=False, y_model_col_names=[None], model_label=None):
    """"""

    N_expts = len(experiment_names)

    if ax is not None:
        ax = ax
    else:
        fig, ax = plt.subplots()

    x_max = y_max = 0
    for tick in range(N_expts):
        experiment_name = experiment_names[tick]
        processed_data_file_name = experiments_dict[experiment_name]['paths']['processed_data']
        df_processed = import_treated_csv_data(processed_data_file_name, experiment_label=experiment_name)

        if add_clusters:
            clustered_data_file_name = experiments_dict[experiment_name]['paths']['clustered_data']
            df_clustered = import_treated_csv_data(clustered_data_file_name, experiment_label=experiment_name)
        else:
            df_clustered = None

        if add_modeled:
            modeled_data_file_name = experiments_dict[experiment_name]['paths']['modeled_data']
            df_modeled = import_treated_csv_data(modeled_data_file_name, experiment_label=experiment_name)
        else:
            df_modeled = None

        ax = plot_csv_data(df_processed, x_col_name, y_col_names,
                           series_labels[tick], series_colors[tick], series_markers[tick], ax=ax,
                           df_model=df_modeled, x_model_col=x_col_name, y_model_cols=y_model_col_names,
                           model_label=model_label,
                           df_cluster=df_clustered, x_cluster_col=x_col_name, y_cluster_cols=y_col_names)

        x_max_expt, y_max_expt = get_plot_limits(df_data=df_processed, x_data_col=x_col_name, y_data_cols=y_col_names,
                                                 df_cluster=df_clustered, x_cluster_col=x_col_name,
                                                 y_cluster_cols=y_col_names, df_model=df_modeled,
                                                 x_model_col=x_col_name, y_model_cols=y_model_col_names)

        if x_max_expt > x_max:
            x_max = x_max_expt

        if y_max_expt > y_max:
            y_max = y_max_expt

    ax.set_ylim(-y_max * 0.05, y_max * 1.1)
    ax.set_xlim(-x_max * 0.05, x_max * 1.05)

    if legend:
        if series_title is None:
            ax.legend(fancybox=False, fontsize=14, loc='lower right')
        else:
            ax.legend(title=series_title, fancybox=False, fontsize=14, loc='lower right')

    if save_fig:
        project_dir = get_project_directory()
        today_str = date.today().strftime("%Y%m%d")
        expt_strings = '_'.join(experiment_names)
        file_name = today_str + '_' + expt_strings + '.png'
        file_path = os.path.join(project_dir, 'results', 'figs_out', file_name)
        plt.savefig(file_path, bbox_inches='tight', dpi=300, transparent=False)

    return ax


def plot_csv_data_with_break(experiments_dict, experiment_names, x_data_col, y_data_cols,
                             series_labels, series_colors, series_markers, x_label, y_label, left_xlims, right_xlims,
                             fig_title=None, series_title=None):
    """
    """

    # code makes two plots (ax_left, ax_right) that share a y
    # selecting appropriate x_lims and plotting twice (for each ax) will mimic a single plot with a break in it
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(4, 3), sharey=True)

    if fig_title:
        fig.suptitle(fig_title)

    # hides the shared axis line
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_left.yaxis.tick_left()
    ax_left.tick_params(labelright=False)
    ax_right.yaxis.tick_right()

    # creates diagonal line to simulate the break
    d = .01
    kwargs = dict(transform=ax_left.transAxes, color='0.25', clip_on=False)
    ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax_right.transAxes)  # switch to the bottom axes
    ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)

    ax_left = plot_csv_data_across_experiments(experiments_dict, experiment_names, x_data_col, y_data_cols,
                                               series_labels, series_colors, series_markers, x_label=None, y_label=None,
                                               series_title=series_title, save_fig=False, legend=False, ax=ax_left)

    ax_right = plot_csv_data_across_experiments(experiments_dict, experiment_names, x_data_col, y_data_cols,
                                                series_labels, series_colors, series_markers, x_label=None, y_label=None,
                                                series_title=series_title, save_fig=False, legend=False, ax=ax_right)

    fig.text(0.5, -0.04, x_label, ha='center', size=16)
    ax_left.set_ylabel(y_label)
    ax_left.set_xlim(left_xlims[0], left_xlims[1])
    ax_right.set_xlim(right_xlims[0], right_xlims[1])
    ax_left.legend(title=series_title, fontsize='10', edgecolor='0.25', fancybox=False, loc='center left')
    plt.setp(ax_left.get_legend().get_title(), fontsize='12')
    ax_right.tick_params(axis='y', length=0)

    return ax_left, ax_right


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
