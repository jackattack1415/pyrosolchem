import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc

from src.d00_utils.conf_utils import *
from src.d00_utils.data_utils import *
from src.d00_utils.plotting_utils import *
from src.d01_data.filter_ms_data import *
from src.d02_extraction.extract_least_sq_fit import perform_regression
from src.d03_modeling.perform_ols import generate_linear_data
from src.d05_reporting.plot_csv_data import *

# activate latex text rendering
rc('text', usetex=True)


sns_style_dict = {'axes.spines.left': False, 'axes.spines.right': False,
                  'axes.spines.top': False, 'ytick.color': '0.15', 'xtick.color': '0.15', 'grid.color': '.9',
                  'ytick.left': False, 'grid.linestyle': '--', 'axes.labelcolor': '0.15'}

sns.set_style("whitegrid", sns_style_dict)
sns.set_context("talk", font_scale=0.85)


compounds, water = load_compounds()
expts = load_experiments('bulk_droplet_experiments.yml')


# bd07as03_bulk_nmr plot: time reacted vs. butenedial molarity
expt_label = 'bd07as03_bulk_nmr'
processed_file_name = expts[expt_label]['paths']['processed_data']
modeled_file_name = expts[expt_label]['paths']['modeled_data']
model_params_file_name = expts[expt_label]['paths']['model_parameters_data']
df_proc_bd07as03_bulk_nmr = import_treated_csv_data(processed_file_name, expt_label)
df_modeled_bd07as03_bulk_nmr = import_treated_csv_data(modeled_file_name, expt_label)
df_model_params_bd07as03_bulk_nmr = import_treated_csv_data(model_params_file_name, expt_label)

k = df_model_params_bd07as03_bulk_nmr.k[0]
k_se = df_model_params_bd07as03_bulk_nmr.k[1]

ax = plot_csv_data(df_data=df_proc_bd07as03_bulk_nmr, x_data_col='MINS_ELAPSED', y_data_cols=['M_BUTENEDIAL'],
                   series_labels=['Observations'], series_markers=['o'], series_colors=['cornflowerblue'],
                   df_model=df_modeled_bd07as03_bulk_nmr, x_model_col='MINS_ELAPSED', y_model_cols=['M_BUTENEDIAL'],
                   model_label=None)
ax.fill_between(df_modeled_bd07as03_bulk_nmr.MINS_ELAPSED, df_modeled_bd07as03_bulk_nmr.M_BUTENEDIAL_MIN,
                df_modeled_bd07as03_bulk_nmr.M_BUTENEDIAL_MAX,
                facecolor='cornflowerblue', alpha=0.1, label='Prediction (95\% Confidence)')

k = df_model_params_bd07as03_bulk_nmr.k[0]
k_se = df_model_params_bd07as03_bulk_nmr.k[1]
annotation_str = "lifetime against \nreaction: %.f $\pm$ %.f hrs" % (1/(k*60), (k_se/k)*(1/(k*60)))
arrow_properties = dict(facecolor="black", arrowstyle='-', color='0.6',
                        lw=1, ls='-', connectionstyle="angle3,angleA=0,angleB=90")
ax.annotate(annotation_str, color='0.35', xy=(800, 0.37), xytext=(900, 0.47), xycoords='data', fontsize=14,
           arrowprops=arrow_properties)

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax.set_yticklabels(ax.get_yticks())
labels = [item.get_text() for item in ax.get_yticklabels()]
labels[-1] = labels[-1][0:3] + r' M \textbf{Butenedial}'
labels.insert(0, '0')
labels.insert(0, '0')

ax.set_yticklabels(labels, ha='left')
ax.tick_params(axis='y', which='major', pad=15)
ax.set_ylim(0, 0.65)

ax.xaxis.set_major_locator(ticker.MultipleLocator(480))
positions = (0, 480, 960, 1440)
labels = ("0", "8", "16", "24")
plt.xticks(positions, labels)
ax.set_xlim(-50, 1550)
ax.set_xlabel('Hours reacted', size=14)

ax.legend(fancybox=False, fontsize=14, loc='lower right')

fig_path = create_fig_path('bdas_reaction_nmr')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# bd07as03_bulk_ms vs. bd07as03_edb_ms: time reacted vs. pyrrolinone signal (in bulk and droplets)
expt_labels = ['bd07as03_bulk_ms', 'bd07as03_edb_ms']

ax_left, ax_right = plot_csv_data_with_break(experiments_dict=expts, experiment_names=expt_labels,
                                             x_data_col='MINS_ELAPSED', y_data_cols=['MZ84_MZ283'],
                                             series_labels=[['Bulk Liquid'], ['Droplet']],
                                             series_colors=[['Lightcoral'], ['Mistyrose']],
                                             series_markers=[['o'], ['o']],
                                             x_label='Hours elapsed', y_label=None,
                                             left_xlims=[-50, 350], right_xlims=[750, 1150],
                                             series_title='Reaction Medium')

positions = (0, 240)
ax_left.set_xticks(positions)
labels = (["0", "4"])
ax_left.set_xticklabels(labels)
labels = (["14", "18"])
positions = (840, 1080)
ax_right.set_xticks(positions)
ax_right.set_xticklabels(labels)

ax_left.set_ylim(0, 0.065)
ax_left.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax_left.set_yticklabels(ax_left.get_yticks())
labels = [item.get_text() for item in ax_left.get_yticklabels()]
labels[-2] = labels[-2][0:4] + r' counts \textbf{Pyrrolinone} per PEG-6'
labels[3] = str(round(float(labels[3]), 2))

ax_left.set_yticklabels(labels, ha='left')
ax_left.tick_params(axis='y', which='major', pad=30)

fig_path = create_fig_path('bdas_reaction_bulk_droplet')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# bd07as03_pr_edb_ms: time evaporated vs. butenedial and pyrrolinone signal (across solutions)
expt_label = 'bd07as03_pr_edb_ms'

processed_file_name = expts[expt_label]['paths']['processed_data']
df_proc_bd07as03_pr_ms = import_treated_csv_data(processed_file_name, expt_label)
df_proc_bd07as03_pr_ms.SOLUTION_ID.unique()
unique_solutions = df_proc_bd07as03_pr_ms.SOLUTION_ID.unique()
N_solutions = len(unique_solutions)

# y0s = []
ks_pr = []
ks_bd = []
ns_samples = []

count = 0
for tick in range(N_solutions):
    n_samples = len(df_proc_bd07as03_pr_ms[df_proc_bd07as03_pr_ms.SOLUTION_ID == unique_solutions[tick]])
    if n_samples > 2:

        t = df_proc_bd07as03_pr_ms.trapped[df_proc_bd07as03_pr_ms.SOLUTION_ID == unique_solutions[tick]].values.reshape(
            -1, 1)
        pr = df_proc_bd07as03_pr_ms.MZ84_MZ283[df_proc_bd07as03_pr_ms.SOLUTION_ID == unique_solutions[tick]].values
        bd = df_proc_bd07as03_pr_ms.MZ85_MZ283[df_proc_bd07as03_pr_ms.SOLUTION_ID == unique_solutions[tick]].values
        ln_pr = np.log(pr)
        ln_bd = np.log(bd)

        b0_pr, b1_pr, score_pr = perform_regression(t, ln_pr)
        b0_bd, b1_bd, score_bd = perform_regression(t, ln_bd)

        ks_pr.append(b1_pr[0])
        ks_bd.append(b1_bd[0])

        # y0s.append(b0)
        # xs, lnyhats = generate_linear_data(x, b0, b1)
        # yhats = np.exp(lnyhats)

        # ax[count].scatter(x, y, s=150, edgecolor='k', c=reds[tick], label='Observations (Sol #%.f)' % (count + 1))
        # ax[count].plot(xs, yhats, lw=3, alpha=0.8, color=reds[tick],
        #                label='OLS fit ($\\tau$ = %.1f hr)' % (-1 / (b1[0] * 60)))
        # ax[count].set_ylim(0.008, df_proc_bd07as03_pr_ms.MZ84_MZ283.max() * 1.3)
        # ax[count].set_xlim(-15, df_proc_bd07as03_pr_ms.trapped.max() * 1.1)
        # ax[count].legend(loc='upper right', edgecolor='k', framealpha=1, fancybox=False, prop={'size': 16})

        # count += 1
        # if (count + 1) == N_solutions:
        #     ax[count - 1].set_xlabel('Time evaporated (min)')
ts_bd = -1 / (np.asarray(ks_bd) * 60)
ts_pr = -1 / (np.asarray(ks_pr) * 60)

fig, ax = plt.subplots()

n_points = len(ks_bd)
ax.bar(np.arange(n_points), ts_bd, width=1, facecolor='Cornflowerblue')
ax.bar(np.arange(1+n_points, 2*n_points+1), ts_pr, width=1, facecolor='Lightcoral')
ax.xaxis.grid(False)
for tick in range(n_points):
    ax.text(tick, ts_bd[tick], str(round(ts_bd[tick], 1)), ha='center', va='bottom', color='Cornflowerblue')
    ax.text(n_points + tick + 1, ts_pr[tick], str(round(ts_pr[tick], 1)), ha='center', va='bottom', color='Lightcoral')

ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_yticklabels(ax.get_yticks())
labels = [str(round(float(item.get_text()))) for item in ax.get_yticklabels()]
labels[-1] = labels[-1] + ' hour lifetime in droplets'
ax.set_yticklabels(labels, ha='left')
ax.tick_params(axis='y', which='major', pad=15)
ax.set_ylim(0, 85)

ax.set_xticklabels(ax.get_xticks())
positions = (1.5, 6.5)
labels = (r'\textbf{Butenedial}', r'\textbf{Pyrrolinone}')
plt.xticks(positions, labels, ha='center')

fig_path = create_fig_path('pr_evap')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# bd10agxx_edb_ms: time reacted vs. pyrrolinone signal (across c_nh3)
expt_label = 'bd10agxx_edb_ms'
processed_file_name = expts[expt_label]['paths']['processed_data']
df_proc_bd10agxx = import_treated_csv_data(processed_file_name, expt_label)

mMs_ammonia = df_proc_bd10agxx.mM_AMMONIA_BUBBLER.unique()
n_tests = len(mMs_ammonia)
max_val = df_proc_bd10agxx.MZ84_MZ283.max()

mMs_ammonia = np.asarray([0, 0.58, 2.9, 145])
nh3g_factor = 30 / 2.9
ppms_ammonia = nh3g_factor * mMs_ammonia
cols_to_plot = ['MZ84_MZ283', 'MZ85_MZ283']

mz84_max = df_proc_bd10agxx.MZ84_MZ283.max()
mz85_max = df_proc_bd10agxx.MZ85_MZ283.max()

blues = ['White', 'Lightsteelblue', 'Cornflowerblue', 'Mediumblue']
reds = ['White', 'Mistyrose', 'Lightcoral', 'Red']

fig, ax = plt.subplots(2, len(mMs_ammonia), sharex=True, figsize=(10, 4.8))

for tick in range(len(mMs_ammonia)):
    mM_ammonia = mMs_ammonia[tick]
    df = df_proc_bd10agxx[df_proc_bd10agxx.mM_AMMONIA_BUBBLER == mM_ammonia]

    ax[0, tick].scatter(df['MINS_ELAPSED'], df['MZ85_MZ283'], s=100, facecolors=blues[tick], alpha=1, edgecolors='k')
    ax[1, tick].scatter(df['MINS_ELAPSED'], df['MZ84_MZ283'], s=100, facecolors=reds[tick], alpha=1, edgecolors='k')

    ax[0, tick].set_ylim(0, 0.12)
    ax[0, tick].set_xlim(-5, 50)
    ax[1, tick].set_ylim(0, 0.12)
    ax[1, tick].set_xlim(-5, 50)
    ax[1, tick].xaxis.set_major_locator(ticker.MultipleLocator(20))

    ax[1, tick].set_xticklabels(ax[1, tick].get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax[1, tick].get_xticklabels()]
    ax[1, tick].set_xticklabels(labels, fontsize=12)

    if tick != 0:
        ax[0, tick].tick_params(labelleft=False, length=0)
        ax[1, tick].tick_params(labelleft=False)

    ax[0, tick].tick_params(labelbottom=False)
    ax[0, tick].set_title('%.f ppm NH$_3$' % ppms_ammonia[tick], fontsize=14)

ax[0, 0].set_yticks(np.asarray([0, 0.05, 0.1]))
ax[0, 0].set_yticklabels(np.asarray(["0.00", "0.05", r"0.10 counts \textbf{Butenedial} per PEG-6"]), ha='left',
                         fontsize=12)
ax[0, 0].tick_params(axis='y', which='major', pad=30)
ax[1, 0].set_yticks(np.asarray([0, 0.05, 0.1]))
ax[1, 0].set_yticklabels(np.asarray(["0.00", "0.05", r"0.10 counts \textbf{Pyrrolinone} per PEG-6"]), ha='left',
                         fontsize=12)
ax[1, 0].tick_params(axis='y', which='major', pad=30)

fig.text(0.5, -0.05, 'Minutes elapsed', ha='center')

fig_path = create_fig_path('bdag_reaction')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 1h-nmr spectra plots: bd07as03 in d2o
dir = get_project_directory()
fn = '20200408_bd07as03rxn_d2o_1h_63min.csv'
path = os.path.join(dir, 'data_raw', 'nmrs', fn)
df_nmr = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_nmr.drop(columns=['x'], inplace=True)

df_nmr['SIG_100_MA'] = df_nmr.iloc[:, 1].rolling(window=100).mean()
fig, ax = plt.subplots()
ax.plot(df_nmr.PPM, df_nmr.SIG_100_MA, lw=2.5, c='0.5')
ax.set_xlim(10.1, -0.1)
ax.set_ylim(-2, 35)
ax.set_xlabel('Shift (ppm)')
ax.axes.get_yaxis().set_visible(False)

fig_path = create_fig_path('bdas_nmr_spectrum')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
