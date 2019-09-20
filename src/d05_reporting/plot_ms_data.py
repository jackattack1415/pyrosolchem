import os
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

from src.d00_utils.conf_utils import get_project_directory


def plot_ms_data(df, experiment_names, t_col_name, y_col_name, y_axis_label):
    """
    """

    sns.set(style="ticks")  # sets sns as the rule
    sns.set_context("talk")
    sns.set_palette("Spectral")

    N_expt = len(experiment_names)
    x_max = 0
    y_max = 0
    for tick in range(N_expt):
        experiment_name = experiment_names[tick]
        xs = df[df.experiment == experiment_name][t_col_name]
        ys = df[df.experiment == experiment_name][y_col_name]

        if xs.max() > x_max:
            x_max = xs.max()
        if ys.max() > y_max:
            y_max = ys.max()

        ax = sns.scatterplot(x=xs, y=ys, label=experiment_name, alpha=1, s=200)

    ax.set(xlabel='time (%s)' % t_col_name, ylabel=y_axis_label,
           ylim=(-y_max * 0.1, y_max * 1.1), xlim=(-x_max * 0.05, x_max * 1.05))
    ax.legend(title='Experiment')

    project_dir = get_project_directory()
    today_str = date.today().strftime("%Y%m%d")
    expt_strings = '-'.join(experiment_names)
    file_name = today_str[2:] + '-' + expt_strings + '.png'
    file_path = os.path.join(project_dir, 'results', 'figs_out', file_name)

    plt.savefig(file_path, bbox_inches='tight', dpi=300, transparent=True)
