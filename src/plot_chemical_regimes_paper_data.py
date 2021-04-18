import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.integrate import odeint

from src.d00_utils.conf_utils import *
from src.d00_utils.data_utils import *
from src.d00_utils.nmr_utils import *
from src.d00_utils.plotting_utils import *
from src.d01_data.filter_ms_data import *
from src.d02_extraction.extract_least_sq_fit import perform_regression
from src.d03_modeling.perform_ols import generate_linear_data
from src.d03_modeling.perform_ols import block_coefficients_outside_confidence_interval
from src.d03_modeling.model_functions import bdasph9_odes
from src.d05_reporting.plot_csv_data import *

# NOTES ABOUT THIS SCRIPT
# Updated January 10, 2020 by Jack Hensley
# This script produces the plots used in the "chemical regimes" paper
# I move through each plot sequentially after the setup
# 1-4: main text
# 5- : supplemental

# 0: setup of sns for plotting
rc('text', usetex=True)

sns_style_dict = {'axes.spines.right': True, 'axes.spines.top': True, 'axes.grid': False, 'axes.edgecolor': '.25',
                  'ytick.color': '0.25', 'xtick.color': '0.25', 'ytick.left': True, 'xtick.bottom': True,
                  'axes.labelcolor': '0.25'}

sns.set_style("whitegrid", sns_style_dict)
sns.set_context("talk", font_scale=0.9)

compounds, water = load_compounds()
expts = load_experiments('chemical_regimes_experiments.yml')

# 1: nmr measurements butenedial + oh: ph=9, 10, 11, k_oh parametrization
# plot parameterization (model) as well as the data points from the measurements (pH vs. disproportionation rate)
expt_labels = ['bdph8_nmr', 'bdph9_nmr', 'bdph10_nmr', 'bdph11_nmr']

# obtain the measurement points for fitting
ks_avg = []
ks_se = []
phs_avg = []
phs_std = []
for expt_label in expt_labels:
    df_model_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)
    df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)
    ks_avg.append(df_model_params.k[0]/60)
    ks_se.append(df_model_params.k[1]/60)
    phs_avg.append(df_processed['pH'].mean())
    phs_std.append(df_processed['pH'].std())

ohs_avg = [10 ** (-14 + x) for x in phs_avg]
ohs = np.logspace(-5.4, -3.3, num=100)  # for plotting


# perform the fitting with confidence interval shown
def disproportionation(oh, ai, aii, aiii):
    """disproportionation rate law from Fratzke, 1986"""
    return (ai * oh + aii * oh * oh) / (1 + aiii * oh)

a, acov = curve_fit(disproportionation, ohs_avg, ks_avg, p0=(1, 1, 1), bounds=([0, 0, 0], [10000, 1000, 100000000]))
kds = disproportionation(ohs, a[0], a[1], a[2])  # best fit value

N = 4
ses = [np.sqrt(acov[0, 0] / N), np.sqrt(acov[1, 1] / N), np.sqrt(acov[2, 2] / N)]

df_coefs = pd.DataFrame()
for tick in range(len(a)):
    mean = a[tick]
    stderr = ses[tick]
    df_coefs['a_' + str(tick)] = np.random.normal(mean, stderr, 10000)

df_coefs = block_coefficients_outside_confidence_interval(df_coefs=df_coefs, ci=68)

# execute odes to get data for each row of synthesized coefficients, N_run iterations
solutions_array = np.empty([len(df_coefs), len(ohs)])

for tick in range(len(df_coefs)):
    a = df_coefs.iloc[tick].values[1:]
    solutions_array[tick, :] = disproportionation(ohs, a[0], a[1], a[2])

# report the average values for each Y_COL_NAME for list of ts
solutions_avg = np.median(solutions_array, 0)
solutions_min = np.min(solutions_array, 0)
solutions_max = np.max(solutions_array, 0)

phs = 14 + np.log10(ohs)

fig, ax = plt.subplots()

ax.plot(phs, kds, c='0.25', alpha=1, lw=3, label='Model Fit ($\pm$ 1$\sigma$)', zorder=2)
ax.plot(phs, solutions_min, c='0.6', alpha=0.5, lw=3, ls='--', zorder=1)
ax.plot(phs, solutions_max, c='0.6', alpha=0.5, lw=3, ls='--', zorder=1)
ax.errorbar(phs_avg, ks_avg, yerr=ks_se, marker='o', ms=7,
            ls='', c='0.25', label=r'Observation ($\pm$ 1$\sigma$)', zorder=3)
ax.set_xlabel('pH')
ax.legend(fancybox=False, loc='lower right')
ax.set_title(r'Butenedial/OH$^{-}$ rate constant, k$_1$ (s$^{-1}$)')

ax.set_ylim(0, 12e-4)
ax.set_yticklabels(ax.get_yticks())
labels = [str(int(float(item.get_text())*1e4)) + r'$\times$10$^{-4}$' for item in ax.get_yticklabels()]
ax.set_yticklabels(labels, ha='left')
ax.tick_params(axis='y', which='major', pad=55)

ax.set_xlim(8.5, 11)
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
ax.set_xticklabels(ax.get_xticks())
ax.set_xticklabels('')
labels = [str(round(float(item.get_text()), 1)) for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)

fig_path = create_fig_path('disproportionation')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 2: nmr measurements butenedial + ammonium sulfate: ph=4-8
expt_label = 'bdnhph5_nmr'
df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)
df_modeled = import_treated_csv_data(expts[expt_label]['paths']['modeled_data'], expt_label)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(9, 2.5))
axes = [ax0, ax1, ax2, ax3]
plt.tight_layout()

param_strs = ['pH', 'M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER']
titles = ['pH', '[BD] (M)', '[PR] (M)', '[BD-PR] (M)']
colors = ['0.25', '0.25', '0.25', '0.25']

for tick in range(4):
    ax = axes[tick]
    param = param_strs[tick]

    ax.plot(df_modeled['MINS_ELAPSED'], df_modeled[param], color=colors[tick], lw=3, label='Model Fit')
    ax.fill_between(df_modeled['MINS_ELAPSED'], df_modeled[param + '_MIN'], df_modeled[param + '_MAX'],
                    color='0.8', label='95\% Confidence Interval')
    ax.scatter(df_processed['MINS_ELAPSED'], df_processed[param], color=colors[tick], s=30, label='Observation')

    ax.set_title(titles[tick])

    # formatting
    bottom, top = ax.get_ylim()
    if param is 'pH':
        ax.set_ylim(ymin=bottom, ymax=top)
        ax.set_yticklabels(ax.get_yticks())
        ylabels = [str(round(float(item.get_text()))) for item in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels)
    else:
        ax.set_ylim(ymin=0, ymax=top*1.1)  # increase ymax of all other plots
        ax.set_yticklabels(ax.get_yticks())
        ylabels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels)

    ax.set_xlabel('mins')  # add xlabel only for the bottom plots

    ax.set_xlim(xmin=-10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50))
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

ax1.legend(fancybox=False, loc='upper center', ncol=3, bbox_to_anchor=(1.25, 1.5), fontsize=12)

fig_path = create_fig_path(expt_label)  # save path
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# 3: predictions of reactants and products in bd07as03 and bdahph9 experiments
# first bdnhph4
expt_label = 'bdnhph4_nmr'
df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)
df_predicted = import_treated_csv_data(expts[expt_label]['paths']['predicted_data'], expt_label)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2.5))
axes = [ax0, ax1]
plt.tight_layout()

param_strs = ['pH', 'M_BUTENEDIAL']
titles = ['pH', '[BD] (M)']

for tick in range(2):
    ax = axes[tick]
    param = param_strs[tick]

    ax.plot(df_predicted['MINS_ELAPSED'], df_predicted[param], color='0.25', lw=3, label='Model Output')
    ax.fill_between(df_predicted['MINS_ELAPSED'], df_predicted[param + '_MIN'], df_predicted[param + '_MAX'],
                    color='0.8', label='95\% Confidence Interval')
    ax.scatter(df_processed['MINS_ELAPSED'], df_processed[param], color='0.25', s=30, label='Observation')

    ax.set_title(titles[tick])

    # formatting
    bottom, top = ax.get_ylim()
    if param is 'pH':
        ax.set_ylim(ymin=3, ymax=4)
        ax.set_yticklabels(ax.get_yticks())
        ylabels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels)
    else:
        ax.set_ylim(ymin=0, ymax=top*1.1)  # increase ymax of all other plots
        ax.set_yticklabels(ax.get_yticks())
        ylabels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels)

    ax.set_xlabel('mins')  # add xlabel only for the bottom plots

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=500))
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

ax1.legend(fancybox=False, loc='center right', bbox_to_anchor=(2.7, 0.5), fontsize=12)

fig_path = create_fig_path(expt_label)  # save path
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# second bdnhph9
expt_label = 'bdnhph8_nmr'
df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)
df_predicted = import_treated_csv_data(expts[expt_label]['paths']['predicted_data'], expt_label)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2.5))
axes = [ax0, ax1]
plt.tight_layout()

param_strs = ['pH', 'M_BUTENEDIAL']
titles = ['pH', '[BD] (M)']

for tick in range(2):
    ax = axes[tick]
    param = param_strs[tick]

    ax.plot(df_predicted['MINS_ELAPSED'], df_predicted[param], color='0.25', lw=3, label='Model Output')
    ax.fill_between(df_predicted['MINS_ELAPSED'], df_predicted[param + '_MIN'], df_predicted[param + '_MAX'],
                    color='0.8', label='95\% Confidence Interval')
    ax.scatter(df_processed['MINS_ELAPSED'], df_processed[param], color='0.25', s=30, label='Observation')

    if tick == 1:
        ax.plot(df_predicted['MINS_ELAPSED'], df_predicted[param].max() - df_predicted['M_BD_OH'], ls='--',
                color='0.25', lw=1, alpha=0.3)

        ax.axhline(y=df_predicted[param].max(), xmin=0.75, xmax=0.8, lw=1, color='0.25', alpha=1)
        ax.axhline(y=0.76, xmin=0.75, xmax=0.8, lw=1, color='0.25', alpha=1)
        ax.axvline(x=25.6, ymin=0.85, ymax=0.875, lw=1, color='0.25', alpha=1)
        ax.text(10, 0.81, 'OH- reactive loss', fontsize=8, alpha=1)

    ax.set_title(titles[tick])

    # formatting
    bottom, top = ax.get_ylim()
    if param is 'pH':
        ax.set_ylim(ymin=8, ymax=9)
        ax.set_yticklabels(ax.get_yticks())
        ylabels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels)
    else:
        ax.set_ylim(ymin=0, ymax=top*1.1)  # increase ymax of all other plots
        ax.set_yticklabels(ax.get_yticks())
        ylabels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels)

    ax.set_xlabel('mins')  # add xlabel only for the bottom plots

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

ax1.legend(fancybox=False, loc='center right', bbox_to_anchor=(2.7, 0.5), fontsize=12)

fig_path = create_fig_path(expt_label)  # save path
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 5: plots of nmr: bd, bdnhph5, bdph11
fns = ['20210201_butenedial_dilute.csv', '20210201_butenedial_concentrated.csv']
d = get_project_directory()
path = os.path.join(d, 'data_raw', 'nmrs', fns[0])
df_0 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_0.drop(columns=['x'], inplace=True)

path = os.path.join(d, 'data_raw', 'nmrs', fns[1])
df_1 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_1.drop(columns=['x'], inplace=True)

dmso_sig_0 = df_0.loc[(df_0.PPM > 3) & (df_0.PPM < 3.2)].SIG.max()
dmso_sig_1 = df_1.loc[(df_1.PPM > 3) & (df_1.PPM < 3.2)].SIG.max()
df_0 = scale_nmr(df_0, 'SIG', dmso_sig_0)
df_1 = scale_nmr(df_1, 'SIG', dmso_sig_1)

fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
gs = GridSpec(2, 4)
gs.update(hspace=0.3)
gs.update(wspace=0.5)
ax0 = plt.subplot(gs[0:2, 0:2])
ax1 = plt.subplot(gs[0, 2:4])
ax2 = plt.subplot(gs[1, 2:4])

ax0.plot(df_1.PPM, df_1.SIG, lw=2, color='red', alpha=0.5, label='Butenedial')
ax0.plot(df_0.PPM, df_0.SIG, lw=2, color='blue', alpha=0.5, label='Butenedial diluted x11')

ax0.set_xlim([8.1, 0.1])
ymax = 40
ax0.set_ylim(bottom=ymax * -0.02, top=ymax)

ax1.plot(df_1.PPM, df_1.SIG, lw=2, color='red', alpha=0.5, label='Butenedial')
ax2.plot(df_0.PPM, df_0.SIG, lw=2, color='blue', alpha=0.5, label='Butenedial diluted x11')

ax1.set_xlim([6.4, 5.6])
ax2.set_xlim([6.4, 5.6])

ymax1 = choose_ymax_nmr_subplot(df_1, 'PPM', [5.5, 7.5])
ymax2 = choose_ymax_nmr_subplot(df_0, 'PPM', [5.5, 7.5])
ax1.set_ylim(bottom=-0.02 * ymax1 * 1.1, top=ymax1 * 1.1)
ax2.set_ylim(bottom=-0.02 * ymax2 * 1.1, top=ymax2 * 1.1)


rect = patches.Rectangle((5.6, ymax/20 * -0.02),
                         0.8, 30,
                         linewidth=1, edgecolor='0.25', facecolor='0.8', alpha=0.2)
ax0.add_patch(rect)

ax0.legend(loc='upper center', ncol=2, bbox_to_anchor=(1, 1.15), fontsize=12, frameon=False)

ax2.text(6.8, -1, 'Chemical Shift (ppm)', fontsize=16)
ax0.set_ylabel('Intensity', fontsize=16)
#
# add the relevant labels for species
# butenedial:
ax0.text(6.2, 27, r'\textbf{BD}', fontsize=10)

ax1.text(5.95, 25.5, r'\textbf{BD, a, cis}', fontsize=10)
ax1.text(6.25, 26, r'\textbf{BD, b, cis}', fontsize=10)
ax1.text(6.37, 13, r'\textbf{BD, a, trans}', fontsize=10)
ax1.text(6.34, 18, r'\textbf{BD, b, trans}', fontsize=10)
ax1.text(5.75, 1.5, 'Reaction\nimpurities', fontsize=10)
ax1.text(6.34, 1.5, 'Potential\nacetal\noligomers', fontsize=10)
ax1.text(6.02, 1.5, 'Potential\nacetal\noligomers', fontsize=10)

ax2.text(5.95, 2.3, r'\textbf{BD, a, cis}', fontsize=10)
ax2.text(6.25, 2.35, r'\textbf{BD, b, cis}', fontsize=10)
ax2.text(6.37, 1.2, r'\textbf{BD, a, trans}', fontsize=10)
ax2.text(6.34, 1.6, r'\textbf{BD, b, trans}', fontsize=10)
ax2.text(5.75, 0.2, 'Reaction\nimpurities', fontsize=10)
ax2.text(6.34, 0.2, 'Potential\nacetal\noligomers', fontsize=10)
ax2.text(6.02, 0.2, 'Potential\nacetal\noligomers', fontsize=10)

# # add in the leftover molecules
ax0.text(5.6, 38, 'HDO', fontsize=10)
ax0.text(3.9, 38, 'DMS', fontsize=10)
ax0.text(2.6, 21, 'HAc', fontsize=10)
ax0.text(4.1, 13, 'MeOH', fontsize=10)

fig_path = create_fig_path('bd_nmr_spectrum')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)

fns = ['20200724_bd_as_nmr.csv', '20210121_bd_as_nmr_2hrs.csv']
d = get_project_directory()
path = os.path.join(d, 'data_raw', 'nmrs', fns[0])
df_0 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_0.drop(columns=['x'], inplace=True)

path = os.path.join(d, 'data_raw', 'nmrs', fns[1])
df_1 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_1.drop(columns=['x'], inplace=True)

dmso_sig_0 = df_0.loc[(df_0.PPM > 3) & (df_0.PPM < 3.2)].SIG.max()
dmso_sig_1 = df_1.loc[(df_1.PPM > 3) & (df_1.PPM < 3.2)].SIG.max()
df_0 = scale_nmr(df_0, 'SIG', dmso_sig_0)
df_1 = scale_nmr(df_1, 'SIG', dmso_sig_1)

fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
gs = GridSpec(2, 4)
gs.update(hspace=0.5)
gs.update(wspace=0.5)
ax0 = plt.subplot(gs[0, :], )
ax1 = plt.subplot(gs[1, 0:3])
ax2 = plt.subplot(gs[1, 3:4])
axes = [ax0, ax1, ax2]

xranges = [[8.1, 1.1], [7, 5], [3.5, 3.25]]
xlims_for_ymax = [[3, 3.2], [3.2, 4.2], [5.5, 6]]

for tick in range(3):
    ax = axes[tick]
    ax.plot(df_0.PPM, df_0.SIG, lw=2, color='blue', alpha=0.5, label='10 min reacted')
    ax.plot(df_1.PPM, df_1.SIG, lw=2, color='red', alpha=0.5, label='120 min reacted')

    ymax = choose_ymax_nmr_subplot(df_0, 'PPM', xlims_for_ymax[tick])

    ax.set_xlim(xranges[tick][0], xranges[tick][1])
    ax.set_ylim(bottom=ymax * -0.02, top=ymax)

    if tick > 0:
        rect = patches.Rectangle((min(xranges[tick]), ymax * -0.02),
                                 max(xranges[tick]) - min(xranges[tick]), ymax,
                                 linewidth=1, edgecolor='0.25', facecolor='0.8', alpha=0.2)
        ax0.add_patch(rect)

ax0.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=12, frameon=False)
ax1.text(6, -15, 'Chemical Shift (ppm)', fontsize=16)
ax0.set_ylabel('Intensity', fontsize=16)
ax1.set_ylabel('Intensity', fontsize=16)

# add the relevant labels for species
# butenedial:
ax0.text(5.9, 60, r'\textbf{BD}', fontsize=10)
ax0.text(6.2, 90, r'\textbf{BD}', fontsize=10)

# pyrrolinone:
ax1.text(6.7, 7, r'\textbf{PR, h}', fontsize=10)
ax1.text(5.8, 20, 'PR, g', fontsize=10)
ax1.plot([5.9, 5.8], [8, 19], lw=1, color='0.25')
ax2.text(3.38, 25, 'PR,\nf', fontsize=10)
ax2.vlines(3.37, 16, 23, lw=1, color='0.25')

# butenedial-pyrrolinone dimer:
ax1.text(5.67, 14, r'\textbf{BD-PR,}', fontsize=10)
ax1.text(5.67, 11, r'\textbf{j}', fontsize=10)
ax1.text(5.45, 9, 'k', fontsize=10)
ax1.text(6.1, 18, 'BD-PR,', fontsize=10)
ax1.text(6., 15, 'n', fontsize=10)
ax1.text(6.08, 12, 'm', fontsize=10)
ax1.text(6.45, 15, 'BD-PR,\nl', fontsize=10)
ax1.vlines(6.3, 6, 16, lw=1, color='0.25')
ax1.vlines(6.05, 5, 11, lw=1, color='0.25')
ax1.vlines(5.97, 6, 14, lw=1, color='0.25')
ax2.text(3.49, 16, 'BD-PR,\ni', fontsize=10)

# add in the leftover molecules
ax0.text(5.15, 100, 'HDO', fontsize=10)
ax0.text(3.42, 100, 'DMS', fontsize=10)
ax0.text(2.1, 45, 'HAc', fontsize=10)
ax0.text(1.5, 15, 'MPA', fontsize=10)
ax2.text(3.38, 48, 'MeOH', fontsize=10)

fig_path = create_fig_path('bdnhx_nmr_spectrum')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)

fns = ['20210126_bdohph11_5min.csv', '20210126_bdohph11_25min.csv',
       '20210126_bdohph11_hr.csv']

path = os.path.join(d, 'data_raw', 'nmrs', fns[0])
df_0 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_0.drop(columns=['x'], inplace=True)

path = os.path.join(d, 'data_raw', 'nmrs', fns[1])
df_1 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_1.drop(columns=['x'], inplace=True)

path = os.path.join(d, 'data_raw', 'nmrs', fns[2])
df_2 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_2.drop(columns=['x'], inplace=True)

dmso_sig_0 = df_0.loc[(df_0.PPM > 3) & (df_0.PPM < 3.2)].SIG.max()
dmso_sig_1 = df_1.loc[(df_1.PPM > 3) & (df_1.PPM < 3.2)].SIG.max()
dmso_sig_2 = df_2.loc[(df_2.PPM > 3) & (df_2.PPM < 3.2)].SIG.max()

df_0 = scale_nmr(df_0, 'SIG', dmso_sig_0)
df_1 = scale_nmr(df_1, 'SIG', dmso_sig_1)
df_2 = scale_nmr(df_2, 'SIG', dmso_sig_2)

fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
gs = GridSpec(2, 4)
gs.update(hspace=0.3)
gs.update(wspace=0.5)
ax0 = plt.subplot(gs[0:2, 0:2])
ax1 = plt.subplot(gs[0, 2:4])
ax2 = plt.subplot(gs[1, 2:4])

axes = [ax0, ax1, ax2]

xranges = [[8.1, 0.1], [4.6, 3.6], [6.3, 5.3]]
xlims_for_ymax = [[5, 6.5], [4.6, 3.6], [5.5, 5]]

for tick in range(3):
    ax = axes[tick]

    ax.plot(df_0.PPM, df_0.SIG, lw=2, color='blue', alpha=0.5, label='5 min reacted')
    ax.plot(df_1.PPM, df_1.SIG, lw=2, color='red', alpha=0.5, label='25 min reacted')
    ax.plot(df_2.PPM, df_2.SIG, lw=2, color='brown', alpha=0.5, label='2 hr reacted')

    ymax = choose_ymax_nmr_subplot(df_0, 'PPM', xlims_for_ymax[tick])

    ax.set_xlim(xranges[tick][0], xranges[tick][1])
    ax.set_ylim(bottom=ymax * -0.02, top=ymax)

    if tick > 0:
        rect = patches.Rectangle((xranges[tick][0], -0.02),
                                  xranges[tick][1] - xranges[tick][0], ymax*1.2,
                                  linewidth=1, edgecolor='0.25', facecolor='0.8', alpha=0.2)
        ax0.add_patch(rect)


ax0.legend(loc='upper center', ncol=3, bbox_to_anchor=(1.1, 1.1), fontsize=12, frameon=False)
ax0.set_ylabel('Intensity')

# add the relevant labels for species
# butenedial:
ax0.text(6.05, 10, r'\textbf{BD, a}', fontsize=10)
ax0.plot([6, 6], [6, 9], lw=1, color='0.25')
ax0.text(7.1, 30, r'\textbf{BD, b}', fontsize=10)

# products:
# ax0.text(5.75, 4, 'accretion\nproducts', fontsize=10)

# add in the leftover molecules
ax0.text(5.6, 40, 'HDO', fontsize=10)
ax0.text(3.05, 40, 'DMS', fontsize=10)
ax0.text(2.15, 15, 'HAc', fontsize=10)
ax0.text(4.2, 18, 'MeOH', fontsize=10)

ax2.text(6.7, -1.3, 'Chemical Shift (ppm)', fontsize=16)
ax0.set_ylabel('Intensity', fontsize=16)

fig_path = create_fig_path('bdoh_nmr_spectrum')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)

# comparative plot of bd/nhx and bd/oh
fns = ['20200724_bd_as_nmr.csv', '20210126_bdohph11_5min.csv']
d = get_project_directory()
path = os.path.join(d, 'data_raw', 'nmrs', fns[0])
df_0 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_0.drop(columns=['x'], inplace=True)

path = os.path.join(d, 'data_raw', 'nmrs', fns[1])
df_1 = pd.read_csv(path, sep='\t', header=None, names=['PPM', 'SIG', 'x'])
df_1.drop(columns=['x'], inplace=True)

dmso_sig_0 = df_0.loc[(df_0.PPM > 3) & (df_0.PPM < 3.2)].SIG.max()
dmso_sig_1 = df_1.loc[(df_1.PPM > 3) & (df_1.PPM < 3.2)].SIG.max()
df_0 = scale_nmr(df_0, 'SIG', dmso_sig_0)
df_1 = scale_nmr(df_1, 'SIG', dmso_sig_1)

fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
gs = GridSpec(2, 4)
gs.update(hspace=0.5)
gs.update(wspace=0.5)
ax0 = plt.subplot(gs[0, :], )
ax1 = plt.subplot(gs[1, 0:3])
ax2 = plt.subplot(gs[1, 3:4])
axes = [ax0, ax1, ax2]

xranges = [[8.1, 1.1], [7, 5], [3.5, 3.25]]
xlims_for_ymax = [[3, 3.2], [3.2, 4.2], [5.5, 6]]

for tick in range(3):
    ax = axes[tick]
    ax.plot(df_0.PPM, df_0.SIG, lw=2, color='blue', alpha=0.5, label='10 min reacted')
    ax.plot(df_1.PPM, df_1.SIG, lw=2, color='red', alpha=0.5, label='120 min reacted')

    ymax = choose_ymax_nmr_subplot(df_0, 'PPM', xlims_for_ymax[tick])

    ax.set_xlim(xranges[tick][0], xranges[tick][1])
    ax.set_ylim(bottom=ymax * -0.02, top=ymax)

    if tick > 0:
        rect = patches.Rectangle((min(xranges[tick]), ymax * -0.02),
                                 max(xranges[tick]) - min(xranges[tick]), ymax,
                                 linewidth=1, edgecolor='0.25', facecolor='0.8', alpha=0.2)
        ax0.add_patch(rect)

ax0.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=12, frameon=False)
ax1.text(6, -15, 'Chemical Shift (ppm)', fontsize=16)
ax0.set_ylabel('Intensity', fontsize=16)
ax1.set_ylabel('Intensity', fontsize=16)

# add the relevant labels for species
# butenedial:
ax0.text(5.9, 60, r'\textbf{BD}', fontsize=10)
ax0.text(6.2, 90, r'\textbf{BD}', fontsize=10)

# pyrrolinone:
ax1.text(6.7, 7, r'\textbf{PR, h}', fontsize=10)
ax1.text(5.8, 20, 'PR, g', fontsize=10)
ax1.plot([5.9, 5.8], [8, 19], lw=1, color='0.25')
ax2.text(3.38, 25, 'PR,\nf', fontsize=10)
ax2.vlines(3.37, 16, 23, lw=1, color='0.25')

# butenedial-pyrrolinone dimer:
ax1.text(5.67, 14, r'\textbf{BD-PR,}', fontsize=10)
ax1.text(5.67, 11, r'\textbf{j}', fontsize=10)
ax1.text(5.45, 9, 'k', fontsize=10)
ax1.text(6.1, 18, 'BD-PR,', fontsize=10)
ax1.text(6., 15, 'n', fontsize=10)
ax1.text(6.08, 12, 'm', fontsize=10)
ax1.text(6.45, 15, 'BD-PR,\nl', fontsize=10)
ax1.vlines(6.3, 6, 16, lw=1, color='0.25')
ax1.vlines(6.05, 5, 11, lw=1, color='0.25')
ax1.vlines(5.97, 6, 14, lw=1, color='0.25')
ax2.text(3.49, 16, 'BD-PR,\ni', fontsize=10)

# add in the leftover molecules
ax0.text(5.15, 100, 'HDO', fontsize=10)
ax0.text(3.42, 100, 'DMS', fontsize=10)
ax0.text(2.1, 45, 'HAc', fontsize=10)
ax0.text(1.5, 15, 'MPA', fontsize=10)
ax2.text(3.38, 48, 'MeOH', fontsize=10)

fig_path = create_fig_path('bdnhx_bdoh_nmr_spectrum')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)

# 6: plots of the mass spec data

# load the data
project_dir = get_project_directory()
solution_file_name = '20210126_solution.txt'
background_file_name = '20210126_solution_background.txt'
data_path = os.path.join(project_dir, 'data_raw', 'ms_files', solution_file_name)
solution_df = pd.read_fwf(data_path, header=None)

data_path = os.path.join(project_dir, 'data_raw', 'ms_files', background_file_name)
background_df = pd.read_fwf(data_path, header=None)

# do some small data treatment
solution_df.columns = ['MZ', 'SIG']
background_df.columns = ['MZ', 'SIG']
solution_df.dropna(inplace=True)
background_df.dropna(inplace=True)
background_df.MZ = background_df.MZ.astype(float)
solution_df.MZ = solution_df.MZ.astype(float)
solution_df['BKGD_SUB_SIG'] = solution_df.SIG - background_df.SIG

solution_df = solution_df.round(0)
solution_df = solution_df.groupby(solution_df.MZ).sum().reset_index()  # group by integer mz units
peg6_fragments = [45, 89, 133, 151, 177, 195, 221, 239, 283, 301]

# plot of total spectrum
fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
gs = GridSpec(2, 6)
gs.update(hspace=0.3)
gs.update(wspace=1.2)
ax = plt.subplot(gs[0, :], )
ax1 = plt.subplot(gs[1, 0:2])
ax2 = plt.subplot(gs[1, 2:5])
ax3 = plt.subplot(gs[1, 5:6])
axes = [ax, ax1, ax2, ax3]

ax.stem(solution_df.MZ, solution_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True,
        basefmt='k')
ax.stem(solution_df.MZ, solution_df.BKGD_SUB_SIG, linefmt='0.3', markerfmt='None', use_line_collection=True,
        basefmt='k')
ax.plot([-1, -1], [-5, -5], c='0.8', label='Raw signal')
ax.plot([-1, -1], [-5, -5], c='0.3', label='Background subtracted signal')
ax.set_xlim(20, 400)
ax.set_ylim(-5000, 100000)
ax.set_xticklabels(ax.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
ax.set_yticklabels(ax.get_yticks())
labels = [str(int(float(item.get_text()))) for item in ax.get_yticklabels()]
ax.set_yticklabels(labels)
ax.set_ylabel('Intensity')

for peg6_fragment in peg6_fragments:
    if peg6_fragment is 283:
        ax.text(peg6_fragment - 3, 5000 + solution_df.SIG[solution_df.MZ == peg6_fragment], '* PEG-6', fontsize=14)
    else:
        ax.text(peg6_fragment - 3, 5000 + solution_df.SIG[solution_df.MZ == peg6_fragment], '*', fontsize=14)

ax.text(78, 80000, '(a)', color='0.5', fontsize=16)
ax.text(143, 80000, '(b)', color='0.5', fontsize=16)
ax.text(163, 80000, '(c)', color='0.5', fontsize=16)

ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=12, frameon=False)

ax1.stem(solution_df.MZ, solution_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True, basefmt='k')
ax1.stem(solution_df.MZ, solution_df.BKGD_SUB_SIG, linefmt='0.4', markerfmt='None', use_line_collection=True,
         basefmt='k')
ax1.set_xlim(83.5, 85.5)
ax1.set_ylim(-1000, 70000)
ax1.set_yticklabels(ax1.get_yticks())
labels = [str(int(float(item.get_text()))) for item in ax1.get_yticklabels()]
ax1.set_yticklabels(labels)
ax1.set_xticklabels(ax1.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax1.get_xticklabels()]
ax1.set_xticklabels(labels)
ax1.text(84.15, 55000, 'PR', fontsize=14)
ax1.text(84.9, 30000, 'BD', fontsize=14)
ax1.text(83.6, 60000, '(a)', color='0.5', fontsize=16)
ax1.set_ylabel('Intensity')

ax2.stem(solution_df.MZ, solution_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True, basefmt='k')
ax2.stem(solution_df.MZ, solution_df.BKGD_SUB_SIG, linefmt='0.4', markerfmt='None', use_line_collection=True,
         basefmt='k')
ax2.set_xlim(148.5, 151.5)
ax2.set_ylim(-500, 20000)
ax2.set_yticklabels(ax2.get_yticks())
labels = [str(int(float(item.get_text()))) for item in ax2.get_yticklabels()]
ax2.set_yticklabels(labels)
ax2.set_xticklabels(ax2.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax2.get_xticklabels()]
ax2.set_xticklabels(labels)
ax2.text(148.7, 5000, 'DZ', fontsize=14)
ax2.text(149.7, 13000, 'BD-PR', fontsize=14)
ax2.text(150.95, 10000, '*', fontsize=14)
ax2.text(148.6, 17000, '(b)', color='0.5', fontsize=16)

ax3.stem(solution_df.MZ, solution_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True, basefmt='k')
ax3.stem(solution_df.MZ, solution_df.BKGD_SUB_SIG, linefmt='0.4', markerfmt='None', use_line_collection=True,
         basefmt='k')
ax3.set_xlim(167.5, 168.5)
ax3.set_ylim(-500, 20000)
ax3.set_yticklabels(ax3.get_yticks())
labels = [str(int(float(item.get_text()))) for item in ax3.get_yticklabels()]
ax3.set_yticklabels(labels)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
ax3.set_xticklabels(ax3.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax3.get_xticklabels()]
ax3.set_xticklabels(labels)
ax3.text(167.55, 6000, 'BD-PR', fontsize=14)
ax3.text(167.6, 17000, '(c)', color='0.5', fontsize=16)

ax2.text(148.3, -8000, 'mass-to-charge ratio')  # xlabel

fig_path = create_fig_path('solution_mass_spec')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 7: plots of the bd degradation data
expt_labels = ['bdph8_nmr', 'bdph9_nmr', 'bdph10_nmr', 'bdph11_nmr']

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(9, 2.5))
axes = [ax0, ax1, ax2, ax3]
plt.tight_layout()

for tick in range(4):
    ax = axes[tick]
    expt_label = expt_labels[tick]
    fn = expts[expt_label]['paths']['processed_data']
    df_processed = import_treated_csv_data(fn, expt_label)

    fn = expts[expt_label]['paths']['modeled_data']
    df_modeled = import_treated_csv_data(fn, expt_label)
    df_model_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)

    title = round(np.mean(df_processed.pH), 1)
    if title >= 10:
        title = 'pH ' + str(title)[:4]
    elif title < 10:
        title = 'pH ' + str(title)[:3]

    ax.plot(df_modeled['MINS_ELAPSED'], df_modeled['M_BUTENEDIAL'], color='0.25', lw=3, label='Model Fit')
    ax.fill_between(df_modeled['MINS_ELAPSED'], df_modeled['M_BUTENEDIAL_MIN'], df_modeled['M_BUTENEDIAL_MAX'],
                    color='0.8', label='95\% Confidence Interval')
    ax.scatter(df_processed['MINS_ELAPSED'], df_processed['M_BUTENEDIAL'], color='0.25', s=30, label='Observation')

    k = 1000*df_model_params['k'][0]/60  # s
    ax.text(2, 0.02, 'k = ' + str(k)[0:4] + r'$\times$10$^{-3}$ s$^{-1}$', fontsize=10)

    ax.set_title(title)

    # formatting
    bottom, top = ax.get_ylim()
    ax.set_ylim(ymin=0, ymax=0.25)  # increase ymax of all other plots
    ax.set_yticklabels(ax.get_yticks())
    ylabels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)

    ax.set_xlabel('mins')  # add xlabel only for the bottom plots

    ax.set_xlim(0, 30)
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

ax0.set_ylabel('[BD] (M)')
ax1.legend(fancybox=False, loc='upper center', ncol=3, bbox_to_anchor=(1.25, 1.5), fontsize=12)

fig_path = create_fig_path('_'.join(expt_labels))  # save path
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


ks_avg = []
phs_avg = []
for expt_label in expt_labels:  # extract the fitted ks and phs from the experiments
    df_model_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)
    df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)
    ks_avg.append(df_model_params.k[0] / 60)  # convert to seconds
    phs_avg.append(df_processed['pH'].mean())

ohs_avg = [10 ** (-14 + x) for x in phs_avg]  # convert from ph to [oh-]


# produce the disproportionation empirical fitting (k = f(ph) from the modeled datasets
def disproportionation(oh, ai, aii, aiii):
    """disproportionation rate law from Fratzke, 1986"""
    return (ai * oh + aii * oh * oh) / (1 + aiii * oh)


a, acov = curve_fit(disproportionation, ohs_avg, ks_avg, p0=(1, 1, 1), bounds=([0, 0, 0], [10000, 1000, 100000000]))

# 8: summary figure plot of butenedial sinks
phs = np.linspace(3, 11.03, 300)
nhxs = np.logspace(-4, 2, 300)
x, y = np.meshgrid(nhxs, phs, sparse=True)

pr_rates = np.empty([len(nhxs), len(phs)])
hca_rates = np.empty([len(nhxs), len(phs)])
dep_rates = np.empty([len(nhxs), len(phs)])

expt_label = 'bdnhph5_nmr'
bdasnmr_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)

expt_label = 'bdph9_nmr'
bdohnmr_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)

# obtain estimates of first-order loss terms for comparison, plot with rgb
for tick in range(len(phs)):
    ph = phs[tick]
    hp = 10**-ph
    oh = 1e-14 / hp
    for tock in range(len(nhxs)):
        nhx = nhxs[tock]
        nh3 = nhx * (1 + hp / 10**(-9.25)) ** (-1)
        nh4 = nhx * (1 + 10**(-9.25) / hp) ** (-1)
        hca_rates[tick, tock] = disproportionation(oh, a[0], a[1], a[2]) * 60
        pr_rates[tick, tock] = bdasnmr_params.k6[0] * nh3
        dep_rates[tick, tock] = 1 / (7 * 24 * 60)  # deposition rate of 1 week converted to minutes


total_rates = hca_rates + pr_rates + dep_rates
hca_rel_rates = hca_rates / total_rates
pr_rel_rates = pr_rates / total_rates
dep_rates = dep_rates / total_rates
alphas = np.full(total_rates.shape, 0.4)
zeros = np.full(total_rates.shape, 0)

rgbs = np.array([pr_rel_rates, dep_rates, hca_rel_rates, alphas])
rgbs = np.moveaxis(rgbs, 0, -1)

ylabels = np.arange(min(phs), max(phs), 2).astype(int)
N_ylabels = len(ylabels)
ypositions = len(phs) * (ylabels - min(phs)) / (max(phs) - min(phs))
ypositions = ypositions.astype(int)
ylabels = ylabels.tolist()
ylabels[-1] = r'pH'

xlabels = np.arange(-4, 3).astype(int)
N_xlabels = len(xlabels)
xpositions = len(nhxs) * (xlabels - min(xlabels)) / (max(xlabels) - min(xlabels))
xlabelstrs = ['$10^{' + str(xlabel) + '}$' for xlabel in xlabels]
xpositions = xpositions.astype(int)

fig, ax = plt.subplots()
ax.imshow(rgbs, origin='lower')
ax.set_yticks(ypositions)
ax.set_yticklabels(ylabels)
ax.set_xticklabels(ax.get_xticks())
ax.set_xticks(xpositions)
ax.set_xticklabels(xlabelstrs)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('[NHx] (M)', fontsize=12)
ax.grid(False)
ax.text(150, 85, r'NH$_{X}$', size=14, c=(1, 0, 0.2, 1))
ax.text(10, 220, r'OH$^{-}$', size=14, c=(0, 0, 1, 1))
ax.text(50, 30, r'Wet Deposition', size=14, c='green')
ax.text(10, 100, r'Atmospheric Range', size=11, c='black')
ax.text(110, 275, r'Phase Separation' + '\n' + r'Observed', size=11, c='white')
ax.text(253, 240, r'AS' + '\n' + r'Solubility' + '\n' + r'Limit', size=11, c='white')

xlen = (len(nhxs) - 2) * (1.44 - min(xlabels)) / (max(xlabels) - min(xlabels))  # goes to 28 M
ylen = (len(phs) - 2) * (6 - min(phs)) / (max(phs) - min(phs))
aerosol = patches.Rectangle((2, 2), xlen, ylen, lw=1, edgecolor='k', fill=None, alpha=0.8)
ax.add_patch(aerosol)

x_ps = len(nhxs) * (0 - min(xlabels)) / (max(xlabels) - min(xlabels))
x_as = len(nhxs) * (1 - min(xlabels)) / (max(xlabels) - min(xlabels))
plt.axvline(x_ps, 0, len(phs), c='white', ls='--', lw=1)
plt.axvline(x_as, 0, len(phs), c='white', ls='--', lw=1)

fig_path = create_fig_path('chemical_regimes_summary')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 9: comparison of nmr and ms

expt_label_nmr = 'bdnhph5_nmr'
df_nmr_processed = import_treated_csv_data(expts[expt_label_nmr]['paths']['processed_data'], expt_label_nmr)
df_nmr_modeled = import_treated_csv_data(expts[expt_label_nmr]['paths']['modeled_data'], expt_label_nmr)

expt_label_ms = 'bdnhph5_ms'
df_ms_processed = import_treated_csv_data(expts[expt_label_ms]['paths']['processed_data'], expt_label_ms)
df_ms_clustered = import_treated_csv_data(expts[expt_label_ms]['paths']['clustered_data'], expt_label_ms)


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7, 2.5))
axes = [ax0, ax1, ax2]
plt.tight_layout()

nmr_param_strs = ['M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER']
ms_param_strs = ['MZ85_MZ283', 'MZ84_MZ283', 'MZ150_MZ283']
scaling = [1.85, 1.85, 1.8]
titles = ['BD', 'PR', 'BD-PR']
colors = ['cornflowerblue', 'coral', 'orchid']

for tick in range(3):
    ax = axes[tick]
    nmr_param = nmr_param_strs[tick]
    ms_param = ms_param_strs[tick]

    l1 = ax.scatter(df_nmr_processed['MINS_ELAPSED'], df_nmr_processed[nmr_param], color='0.25', s=30, label='NMR',
               zorder=5)
    l2 = ax.plot(df_nmr_modeled['MINS_ELAPSED'], df_nmr_modeled[nmr_param], color='0.25', lw=2, label='Model', zorder=4)
    l3 = ax.fill_between(df_nmr_modeled['MINS_ELAPSED'], df_nmr_modeled[nmr_param + '_MIN'],
                    df_nmr_modeled[nmr_param + '_MAX'], color='0.25', label='Model (95\% CI)',
                    alpha=0.3, linewidth=0, zorder=3)
    at = ax.twinx()
    l4 = at.scatter(df_ms_processed['MINS_ELAPSED'], df_ms_processed[ms_param], color=colors[tick], s=3, label='MS',
                    zorder=1)
    l5 = at.errorbar(df_ms_clustered['MINS_ELAPSED'], df_ms_clustered[ms_param], xerr=df_ms_clustered['MINS_ELAPSED_std'],
                yerr= df_ms_clustered[ms_param + '_std'], marker='o', ms=7, ls='', c='1',
                markeredgecolor=colors[tick], mew=2, ecolor=colors[tick],
                label=r'MS ($\pm$ 1$\sigma$)', zorder=0)

    ax.set_title(titles[tick])

    # formatting
    ymax_nmr = df_nmr_processed[nmr_param].max()
    ymax_ms = df_ms_clustered[ms_param].max()

    ax.set_ylim(ymin=0, ymax=ymax_nmr*scaling[tick])  # increase ymax of all other plots
    at.set_ylim(ymin=0, ymax=ymax_ms*2)  # increase ymax of all other plots

    ax.set_yticks([])
    at.set_yticks([])

    ax.set_xlabel('mins')  # add xlabel only for the bottom plots

    ax.set_xlim(xmin=0, xmax=100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50))
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    # ls = l1 + l2 + l3 + l4 + l5
    # labs = [l.get_label() for l in ls]

    if ax == ax1:
        ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.6, 1.6), fontsize=12)
        at.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.6, 1.45), fontsize=12)


# ax1.legend(fancybox=False, loc='upper center', ncol=3, bbox_to_anchor=(1.25, 1.5), fontsize=12)

fig_path = create_fig_path('ms_nmr_comparison')  # save path
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)