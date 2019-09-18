import os
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt

from src.d00_utils.conf_utils import get_project_directory


def plot_composition_evolution(compounds, ts, ys, y_axis, rs=None):
    """ plots time evolution of compounds in solution (e.g., droplet). Figure saved in results as png.

    :param compounds: (dict) dictionary of definitions of each compound.
    :param ts: (ndarray(floats)) 1D array of floats of times in simulation.
    :param ys: (ndarray(floats)) 2D array of floats of y parameter to plot by compound and time in simulation.
    :param y_axis: (str) can either be M, n, or N and governs the resulting y axis label. corresponds to ys.
    :param rs: (ndarray(floats)) 1D array of floats of radii (m) in time in simulation. Not plotted if None.
    """

    sns.set(style="ticks")  # sets sns as the rule
    sns.set_context("talk")
    sns.set_palette("Spectral")

    N_cmpd = len(compounds)
    compound_names = [defs['name'] for name, defs in compounds.items()]

    hrs = ts / 3600
    for tick in range(N_cmpd):
        ax = sns.lineplot(x=hrs, y=ys[:, tick], label=compound_names[tick], alpha=1)

    if y_axis == 'M':
        y_label = 'M (mol  L$^{-1}$)'
    elif y_axis == 'n':
        y_label = 'moles'
    elif y_axis == 'N':
        y_label = 'molecules'
    else:
        print('y parameter not valid for plotting')

    ax.set(xlabel='time (hr)', ylabel=y_label)
    ax.legend(title='Compound')

    if rs is not None:
        ums = rs * 1e6

        ax2 = ax.twinx()
        ax2.plot(hrs, ums, color='black', linewidth=2, linestyle='--', alpha=0.4)
        ax2.set(ylabel='r (um)')

    project_dir = get_project_directory()
    today_str = date.today().strftime("%Y%m%d")
    cmpd_strings = '_'.join(compound_names)
    fig = today_str[2:] + '_' + cmpd_strings + '_evap.png'
    fig_path = os.path.join(project_dir, 'results', fig)

    plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
