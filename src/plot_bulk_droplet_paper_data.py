import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sklearn.cluster as cluster
import statsmodels.api as sm

from src.d00_utils.conf_utils import *
from src.d00_utils.data_utils import *
from src.d00_utils.plotting_utils import *
from src.d01_data.filter_ms_data import *
from src.d02_extraction.extract_least_sq_fit import perform_regression
from src.d03_modeling.perform_ols import generate_linear_data
from src.d05_reporting.plot_csv_data import *

# activate latex text rendering
rc('text', usetex=True)

sns_style_dict = {'axes.spines.right': True, 'axes.spines.top': True, 'axes.grid': False, 'axes.edgecolor': '.25',
                  'ytick.color': '0.25', 'xtick.color': '0.25', 'ytick.left': True, 'xtick.bottom': True,
                  'axes.labelcolor': '0.25'}

sns.set_style("whitegrid", sns_style_dict)
sns.set_context("talk", font_scale=0.9)

compounds, water = load_compounds()
expts = load_experiments('bulk_droplet_experiments.yml')

# 1. bd07as03_bulk_ms vs. bd07as03_edb_ms: time reacted vs. pyrrolinone signal (in bulk and droplets)
expt_labels = ['bd07as03_bulk_ms', 'bd07as03_edb_ms']

ax_left, ax_right = plot_csv_data_with_break(experiments_dict=expts, experiment_names=expt_labels,
                                             x_data_col='MINS_ELAPSED', y_data_cols=['MZ84_MZ283'],
                                             series_labels=[['Bulk Liquid'], ['Levitated Particle']],
                                             series_colors=[['mistyrose'], ['coral']],
                                             series_markers=[['o'], ['o']],
                                             x_label='hours', y_label=None,
                                             left_xlims=[-50, 350], right_xlims=[750, 1150],
                                             fig_title=r'PR (n.s.)', series_title='Reaction Medium')

positions = (0, 240)
ax_left.set_xticks(positions)
labels = (["0", "4"])
ax_left.set_xticklabels(labels)
labels = (["14", "18"])
positions = (840, 1080)
ax_right.set_xticks(positions)
ax_right.set_xticklabels(labels)

ax_left.set_yticklabels(ax_left.get_yticks())
labels = [item.get_text() for item in ax_left.get_yticklabels()]
labels[3] = str(round(float(labels[3]), 2))

ax_left.set_yticklabels(labels, ha='left')
ax_left.tick_params(axis='y', which='major', pad=30)

fig_path = create_fig_path('bdas_reaction_bulk_droplet')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 2. bd10agxx_edb_ms: time reacted vs. pyrrolinone signal (across c_nh3)
expt_label = 'bd10agxx_edb_ms'
processed_file_name = expts[expt_label]['paths']['processed_data']
df_proc_bd10agxx = import_treated_csv_data(processed_file_name, expt_label)

ppms_ammonia = np.sort(df_proc_bd10agxx.PPM_AMMONIA.unique())
cols_to_plot = ['MZ85_MZ283', 'MZ84_MZ283']

mz84_max = df_proc_bd10agxx.MZ84_MZ283.max()
mz85_max = df_proc_bd10agxx.MZ85_MZ283.max()

fig, ax = plt.subplots(2, len(ppms_ammonia), sharex=True, figsize=(6, 3))
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)

for tick in range(len(ppms_ammonia)):
    ppm_ammonia = ppms_ammonia[tick]
    df = df_proc_bd10agxx[df_proc_bd10agxx.PPM_AMMONIA == ppm_ammonia]

    ax[0, tick].scatter(df['MINS_ELAPSED'], df.MZ85_MZ283, s=30, facecolors='cornflowerblue', edgecolor='0.25')
    ax[1, tick].scatter(df['MINS_ELAPSED'], df.MZ84_MZ283, s=30, facecolors='coral', edgecolor='0.25')
    ax[0, tick].tick_params('both', length=5, width=2, which='major')
    ax[1, tick].tick_params('both', length=5, width=2, which='major')

    ax[0, tick].set_xlim(-5, 85)
    ax[1, tick].set_xlim(-5, 85)
    ax[0, tick].yaxis.set_label_position("right")
    ax[0, tick].yaxis.tick_right()
    ax[1, tick].yaxis.set_label_position("right")
    ax[1, tick].yaxis.tick_right()

    ax[1, tick].xaxis.set_major_locator(ticker.MultipleLocator(40))
    ax[0, tick].yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax[1, tick].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax[0, tick].set_ylim(-mz85_max * 0.1, mz85_max * 1.3)
    ax[1, tick].set_ylim(-mz84_max * 0.1, mz84_max * 1.2)

    ax[1, tick].set_xticklabels(ax[1, tick].get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax[1, tick].get_xticklabels()]
    ax[1, tick].set_xticklabels(labels, fontsize=12)

    ax[0, tick].set_yticklabels(ax[0, tick].get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax[0, tick].get_yticklabels()]
    ax[0, tick].set_yticklabels(labels, fontsize=12)

    ax[1, tick].set_yticklabels(ax[1, tick].get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax[1, tick].get_yticklabels()]
    ax[1, tick].set_yticklabels(labels, fontsize=12)

    if tick != 3:
        ax[0, tick].tick_params(labelright=False)
        ax[1, tick].tick_params(labelright=False)

    ax[0, tick].tick_params(labelbottom=False)
    ax[1, tick].set_xlabel('mins')
    if 0 < ppm_ammonia < 10:
        title_str = str(round(ppm_ammonia, 1))[:3] + ' ppm NH$_3$'
    else:
        title_str = '{0:.0f} ppm NH$_3$'.format(ppm_ammonia)
    ax[0, tick].set_title(title_str, fontsize=14)

ax[0, 0].text(-40, 0.02, 'BD\n(n.s.)', multialignment='center', horizontalalignment='left',
              verticalalignment='center', fontsize=14)
ax[1, 0].text(-40, 0.05, 'PR\n(n.s.)', multialignment='center', horizontalalignment='left',
              verticalalignment='center', fontsize=14)

fig_path = create_fig_path('bdag_reaction')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# plot of mz150 vs. mz168
fig, ax = plt.subplots()
ax.scatter(df_proc_bd10agxx.MZ150_MZ283, df_proc_bd10agxx.MZ168_MZ283, s=30, color='0.25')
ax.set_xlabel('BD-PR (m/z 150) \n counts per PEG-6')
ax.set_ylabel('BD-PR (m/z 168) \n counts per PEG-6')

fig_path = create_fig_path('mz150vsmz168')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 3: calibrated bd10ag30_edb_ms data vs. model prediction
expt_label = 'bd10ag30_edb_ms'
df_proc_bd10ag30 = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)
df_clus_bd10ag30 = import_treated_csv_data(expts[expt_label]['paths']['clustered_data'], expt_label)
df_pred_bd10ag30 = import_treated_csv_data(expts[expt_label]['paths']['predicted_data'], expt_label)

cols_to_plot = ['M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER']
title_str = ['[BD] (M)', '[PR] (M)', '[BD-PR] (M)']

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7, 2.5))
axes = [ax0, ax1, ax2]
plt.tight_layout()

for tick in range(3):
    ax = axes[tick]
    col = cols_to_plot[tick]

    ax.fill_between(df_pred_bd10ag30.MINS_ELAPSED, df_pred_bd10ag30[col + '_MIN'], df_pred_bd10ag30[col + '_MAX'],
                    color='0.8', label='95\% Confidence Interval')
    ax.plot(df_pred_bd10ag30.MINS_ELAPSED, df_pred_bd10ag30[col],
                    color='0.25', label='Model Prediction')
    ax.scatter(df_proc_bd10ag30.MINS_ELAPSED, df_proc_bd10ag30[col], color='0.25', s=30, label='Observation')
    ax.errorbar(df_clus_bd10ag30.MINS_ELAPSED, df_clus_bd10ag30[col], color='0.25', s=30, label='Cluster')

    ax.set_xlim(-5, 95)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

    ax.set_ylim(-df_proc_bd10ag30[col].max() * 0.1,
                np.max([df_proc_bd10ag30[col].max(), df_pred_bd10ag30[col + '_MAX'].max()]) * 1.2)

    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_yticklabels(ax.get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel('mins')
    ax.set_title(title_str[tick])

ax1.legend(fancybox=False, loc='upper center', ncol=3, bbox_to_anchor=(0.4, 1.5), fontsize=12)

fig_path = create_fig_path(expt_label)
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 4: plot of sinks as a function of C_nh3

ppb_nh3s = np.logspace(-1, 5, num=100)
H_nh3 = 61  # M atm-1
M_nh3s = ppb_nh3s * H_nh3 * 1e-9

# import the modeling mechanism (from bdasph9 and the bdoh fittings)
expt_label = 'bdnhph5_nmr'
file_name = expts[expt_label]['paths']['model_parameters_data']
model_parameters = import_treated_csv_data(file_name, expt_label)
# add parameters: k[0] = mean
k6 = model_parameters['k6'][0]
k7 = model_parameters['k7'][0]
k8 = model_parameters['k8'][0]
k9 = model_parameters['k9'][0]
k10 = model_parameters['k10'][0]
ke = 1 / (2.6 * 60)  # take k = tau_evap ** -1 and convert to min

# reporting the disproportionation empirical fitting (k = f(ph)) from make_chemical_regimes_paper_data.py
ai = 15.5
aii = 64.6
aiii = 1.61e4


K_S2 = 1.2e-2
K_N = 10**(-9.25)

def butenedial_branching_ode(ts, coefs):
    def odes(y, t):
        bdg, bd, pr, dm = y

        # solution characteristics
        S_T = coefs[5]  # total sulfur, moles

        def calculate_hplus(H):
            
            # from sulfate equilibrium, assuming all h2so4 dissociated since strong acid
            HSO4 = S_T * (1 + (K_S2 / H))**(-1)
            SO4 = 2 * S_T * (1 + (H / K_S2))**(-1)
            
            # from ammonia equilibrium, where coefs[4] is nh3(particle)
            NH4 = H * coefs[4] / K_N
            
            return HSO4 + 2 * SO4 - NH4 - H  # charge balance

        hp = fsolve(calculate_hplus, 1e-5)

        oh = 1e-14 / hp
        nh3 = coefs[4]

        # rate constants
        k1 = ((ai * oh + aii * oh * oh) / (1 + aiii * oh)) * 60

        # rate laws
        dydt = [ke * bd,  # need bdg to have gas-phase butenedial accounted for
                -k6 * bd * nh3 - k1 * bd - ke * bd - k7 * bd * pr * oh - k8 * bd * dm,
                k6 * bd * nh3 - k7 * bd * pr * oh - k9 * pr * dm,
                k7 * bd * pr * oh - k10 * dm * dm]

        return dydt

    solution = odeint(odes, y0=coefs[0:4], t=ts)

    return solution


butenedial_branching_low_sulfur = np.empty([len(M_nh3s)])
butenedial_branching_high_sulfur = np.empty([len(M_nh3s)])
ts = np.arange(0, 1000)
for tick in range(len(M_nh3s)):
    M_nh3 = M_nh3s[tick]
    S_low = 0.1
    S_high = 5
    bd0 = 1
    pr0 = dm0 = bdg0 = 0
    coef = [bdg0, bd0, pr0, dm0, M_nh3, S_low]
    solution = butenedial_branching_ode(ts, coef)
    butenedial_branching_low_sulfur[tick] = solution[-1, 0]

    coef = [bdg0, bd0, pr0, dm0, M_nh3, S_high]
    solution = butenedial_branching_ode(ts, coef)
    butenedial_branching_high_sulfur[tick] = solution[-1, 0]

fig, ax = plt.subplots()

ax.fill_between(ppb_nh3s, butenedial_branching_low_sulfur, butenedial_branching_high_sulfur, color='gray', alpha=0.2)
ax.fill_between(ppb_nh3s, 1 - butenedial_branching_low_sulfur, 1 - butenedial_branching_high_sulfur, color='chocolate', alpha=0.4)
ax.plot(ppb_nh3s, butenedial_branching_low_sulfur, color='gray', lw=1, alpha=0.8)
ax.plot(ppb_nh3s, butenedial_branching_high_sulfur, color='gray', lw=1, alpha=0.8)
ax.plot(ppb_nh3s, 1 - butenedial_branching_low_sulfur, color='chocolate', lw=1, alpha=0.8)
ax.plot(ppb_nh3s, 1 - butenedial_branching_high_sulfur, color='chocolate', lw=1, alpha=0.8)
ax.set_xscale('log')
ax.set_xlabel('NH$_3$ mixing ratio')
ax.set_xticklabels(ax.get_xticks())
labels = [float(item.get_text()) for item in ax.get_xticklabels()]
for tick in range(len(labels)):
    if labels[tick] < 1:
        labels[tick] = str(labels[tick])[:3] + ' ppb'
    elif (labels[tick] >= 1) and (labels[tick] < 1000):
        labels[tick] = str(int(labels[tick])) + ' ppb'
    elif labels[tick] >= 1000:
        labels[tick] = str(int(labels[tick]))[:-3] + ' ppm'
ax.set_xticklabels(labels)

ax.set_ylim(-0.1, 1.02)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.set_yticklabels(ax.get_yticks())
labels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
ax.set_yticklabels(labels)

ax.set_title('Butenedial branching ratio')
ax.text(1.2, 0.8, r'\textbf{Evaporation}', c='gray', size=14)
ax.text(10000, 0.6, r'\textbf{Reaction}', c='chocolate', size=14)
ax.text(700, 0.87, r'5 M S(VI)', c='gray', size=10) #, rotation=-55)
ax.text(50, 0.6, r'0.1 M S(VI)', c='gray', size=10) #, rotation=-65)
ax.text(300, 0.03, r'5 M S(VI)', c='chocolate', size=10) #, rotation=45)
ax.text(25, 0.2, r'0.1 M S(VI)', c='chocolate', size=10) #, rotation=70)

ax.axvline(x=0.1, ymin=0, ymax=0.09, c='k', ls='--', lw=0.5)
ax.axvline(x=10, ymin=0, ymax=0.09, c='k', ls='--', lw=0.5)
ax.axvline(x=100, ymin=0, ymax=0.09, c='k', ls='--', lw=0.5)
ax.axhline(y=0, xmin=0, xmax=1, c='k', lw=0.5)
ax.text(0.1, 0.05, 'UT/Remote', backgroundcolor='1', size=8)
ax.plot([0.07, 0.09], [-0.02, 0.03], color='k', lw=0.5)
ax.text(1, -0.06, 'LT', size=8)
ax.text(17, -0.06, 'Polluted', size=8)
ax.text(2000, -0.06, 'Plumes', size=8)

fig_path = create_fig_path('reaction_vs_evaporation')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)


# 4: nmr and ms measurements butenedial + ammonium sulfate: ph=4-8, conversion factors
expt_labels = ['bdnhph5_nmr', 'bdnhph5_ms']
df_nmr_clustered = import_treated_csv_data(expts[expt_labels[0]]['paths']['clustered_data'], expt_labels[0])
df_ms_clustered = import_treated_csv_data(expts[expt_labels[1]]['paths']['clustered_data'], expt_labels[1])

nmr_cols = ['M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER']
ms_cols = ['MZ85_MZ283', 'MZ84_MZ283', 'MZ150_MZ283']

WTFRAC_PEG6_MS = expts[expt_labels[1]]['experimental_conditions']['solution_weight_fractions']['PEG-6']
MW_PEG6 = compounds['hexaethylene_glycol']['mw']
M_PEG6_MS = WTFRAC_PEG6_MS / MW_PEG6


def perform_regression(x, y):
    x = x.reshape(-1, 1)
    ols = sm.OLS(y, x)
    ols_result = ols.fit()

    return ols_result


cfs = []
cf_ses = []
for tick in range(3):
    ms_signals = df_ms_clustered[ms_cols[tick]][:5]  # taking to 5 to remove the na
    nmr_Ms = df_nmr_clustered[nmr_cols[tick]][:5]  # taking to 5 to remove the na

    ols_results = perform_regression(np.array(ms_signals * M_PEG6_MS), np.array(nmr_Ms))
    cf = ols_results.params[0]
    se = ols_results.bse[0]
    cfs.append(cf)
    cf_ses.append(se)

title_str = ['[BD] (M)', '[PR] (M)', '[BD-PR] (M)']

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7, 2.5))
axes = [ax0, ax1, ax2]
plt.tight_layout()

for tick in range(3):
    ax = axes[tick]
    col = cols_to_plot[tick]

    xs = np.linspace(0, np.nanmax(df_ms_clustered[ms_cols[tick]])*1.5, num=10)

    ax.plot(xs, xs * M_PEG6_MS * cfs[tick], color='0.25', label='Best Fit',
            zorder=2)
    ax.fill_between(xs, xs * M_PEG6_MS * (cfs[tick] - 2 * cf_ses[tick]),
                    xs * M_PEG6_MS * (cfs[tick] + 2 * cf_ses[tick]),
                    color='0.8', zorder=1, label=r'95\% Confidence')
    ax.errorbar(df_ms_clustered[ms_cols[tick]][:5], df_nmr_clustered[nmr_cols[tick]][:5],
                xerr=df_ms_clustered[ms_cols[tick] + '_std'][:5],
                yerr=df_nmr_clustered[nmr_cols[tick] + '_std'][:5],
                color='0.25', marker='o', ms=7, ls='', label='Observation', zorder=3)

    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_yticklabels(ax.get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=12)

    molecule_str = title_str[tick].split(']')[0][1:]
    ax.set_xlabel('counts ' + molecule_str + '\n per PEG-6')
    ax.set_title(title_str[tick])
    ax.text(np.nanmax(df_ms_clustered[ms_cols[tick]])*0.3, 0,
            'b = ' + str(cfs[tick])[:4] + ' $\pm$ ' + str(cf_ses[tick])[:4], fontsize=10)

ax1.legend(fancybox=False, loc='upper center', ncol=3, bbox_to_anchor=(0.4, 1.5), fontsize=12)

fig_path = create_fig_path('_'.join(expt_labels) + '_calibration')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)


# 5: plot mass spec of droplet
# load the data
project_dir = get_project_directory()
particle_file_name = '20200722_droplet.txt'
background_file_name = '20200722_droplet_background.txt'
data_path = os.path.join(project_dir, 'data_raw', 'ms_files', particle_file_name)
particle_df = pd.read_fwf(data_path, header=None)

data_path = os.path.join(project_dir, 'data_raw', 'ms_files', background_file_name)
background_df = pd.read_fwf(data_path, header=None)

# do some small data treatment
particle_df.columns = ['MZ', 'SIG']
background_df.columns = ['MZ', 'SIG']
particle_df.dropna(inplace=True)
background_df.dropna(inplace=True)
background_df.MZ = background_df.MZ.astype(float)
particle_df.MZ = particle_df.MZ.astype(float)
particle_df['BKGD_SUB_SIG'] = particle_df.SIG - background_df.SIG

particle_df = particle_df.round(0)
particle_df = particle_df.groupby(particle_df.MZ).sum().reset_index()  # group by integer mz units
peg6_fragments = [45, 89, 133, 177, 283, 300]

# plot of total spectrum
fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
gs = GridSpec(2, 6)
gs.update(hspace=0.3)
gs.update(wspace=0.5)
ax = plt.subplot(gs[0, :], )
ax1 = plt.subplot(gs[1, 0:2])
ax2 = plt.subplot(gs[1, 2:5])
ax3 = plt.subplot(gs[1, 5:6])
axes = [ax, ax1, ax2, ax3]

ax.stem(particle_df.MZ, particle_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True,
        basefmt='k')
ax.stem(particle_df.MZ, particle_df.BKGD_SUB_SIG, linefmt='0.3', markerfmt='None', use_line_collection=True,
        basefmt='k')
ax.plot([-1, -1], [-5, -5], c='0.8', label='Raw signal')
ax.plot([-1, -1], [-5, -5], c='0.3', label='Background subtracted signal')
ax.set_xlim(20, 400)
ax.set_ylim(-50000, 1200000)
ax.set_xticklabels(ax.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
ax.set_yticklabels(ax.get_yticks())
labels = [str(int(float(item.get_text())/1e5)) + r'$\times$10$^{5}$' for item in ax.get_yticklabels()]
ax.set_yticklabels(labels)
ax.set_ylabel('Intensity')

for peg6_fragment in peg6_fragments:
    if peg6_fragment is 283:
        ax.text(peg6_fragment - 3, 5000 + particle_df.SIG[particle_df.MZ == peg6_fragment], '* PEG-6', fontsize=14)
    else:
        ax.text(peg6_fragment - 2, 50000 + particle_df.SIG[particle_df.MZ == peg6_fragment], '*', fontsize=14)

ax.text(78, 500000, '(a)', color='0.5', fontsize=16)
ax.text(143, 500000, '(b)', color='0.5', fontsize=16)
ax.text(163, 500000, '(c)', color='0.5', fontsize=16)

ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=12, frameon=False)

ax1.stem(particle_df.MZ, particle_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True, basefmt='k')
ax1.stem(particle_df.MZ, particle_df.BKGD_SUB_SIG, linefmt='0.4', markerfmt='None', use_line_collection=True,
         basefmt='k')
ax1.set_xlim(83.5, 85.5)
ax1.set_ylim(-1000, 30000)
ax1.set_yticklabels(ax1.get_yticks())
labels = [str(int(float(item.get_text())))[0] + r'$\times$10$^{4}$' for item in ax1.get_yticklabels()]
ax1.set_yticklabels(labels)
ax1.set_xticklabels(ax1.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax1.get_xticklabels()]
ax1.set_xticklabels(labels)
ax1.text(83.9, 21000, 'PR', fontsize=14)
ax1.text(84.9, 19000, 'BD', fontsize=14)
ax1.text(83.6, 25000, '(a)', color='0.5', fontsize=16)
ax1.set_ylabel('Intensity')

ax2.stem(particle_df.MZ, particle_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True, basefmt='k')
ax2.stem(particle_df.MZ, particle_df.BKGD_SUB_SIG, linefmt='0.4', markerfmt='None', use_line_collection=True,
         basefmt='k')
ax2.set_xlim(148.5, 151.5)
ax2.set_ylim(-500, 30000)
ax2.set_yticklabels(ax2.get_yticks())
labels = [str(int(float(item.get_text())))[0] + r'$\times$10$^{4}$' for item in ax2.get_yticklabels()]

ax2.set_yticklabels([])  # same axis as ax1
ax2.set_xticklabels(ax2.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax2.get_xticklabels()]
ax2.set_xticklabels(labels)
ax2.text(148.9, 10000, 'DZ', fontsize=14)
ax2.text(149.8, 23000, 'BD-PR', fontsize=14)
ax2.text(150.95, 15000, '*', fontsize=14)
ax2.text(148.6, 25000, '(b)', color='0.5', fontsize=16)

ax3.stem(particle_df.MZ, particle_df.SIG, linefmt='0.8', markerfmt='None', use_line_collection=True, basefmt='k')
ax3.stem(particle_df.MZ, particle_df.BKGD_SUB_SIG, linefmt='0.4', markerfmt='None', use_line_collection=True,
         basefmt='k')
ax3.set_xlim(167.5, 168.5)
ax3.set_ylim(-500, 30000)
ax3.set_yticklabels(ax3.get_yticks())
labels = [str(int(float(item.get_text())))[0] + r'$\times$10$^{4}$' for item in ax3.get_yticklabels()]

ax3.set_yticklabels([])  # same axis as ax1
ax3.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
ax3.set_xticklabels(ax3.get_xticks())
labels = [str(int(float(item.get_text()))) for item in ax3.get_xticklabels()]
ax3.set_xticklabels(labels)
ax3.text(167.65, 14000, 'BD-PR', fontsize=14)
ax3.text(167.6, 25000, '(c)', color='0.5', fontsize=16)

ax2.text(148.3, -12000, 'mass-to-charge ratio')  # xlabel

fig_path = create_fig_path('droplet_mass_spec')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# 6. chromatographs of the mass spec data
project_dir = get_project_directory()
particle_file_name = '20210126_droplet_chromatograph.csv'
solution_file_name = '20210126_solution_chromatograph.csv'
data_path = os.path.join(project_dir, 'data_raw', 'ms_files', particle_file_name)
df_particle_chrom = pd.read_csv(data_path)

data_path = os.path.join(project_dir, 'data_raw', 'ms_files', solution_file_name)
df_solution_chrom = pd.read_csv(data_path)

fig, axes = plt.subplots(2, 3, figsize=(7, 5))
plt.tight_layout()

param_strs = ['COUNTS_MZ84', 'COUNTS_MZ85', 'COUNTS_MZ150', 'COUNTS_MZ168', 'COUNTS_MZ149', 'COUNTS_MZ283']
titles = ['PR', 'BD', 'BD-PR (m/z 150)', 'BD-PR (m/z 168)', 'DZ', 'PEG-6']

for tick in range(6):
    row = int(np.floor(tick/3))
    col = tick - row * 3
    ax = axes[row, col]
    param = param_strs[tick]
    df = df_particle_chrom.query('SECONDS >= 3.2 and SECONDS <= 3.8')  # temporal location of where the peak is

    ax.plot(df['SECONDS'], df[param], color='0.25', lw=3, alpha=0.5)

    ax.set_title(titles[tick])

    # formatting
    # bottom, top = ax.get_ylim()
    ax.set_ylim(ymin=-500, ymax=6000)
    if tick is 5:
        ax.set_ylim(ymin=-25000, ymax=300000)
    ax.set_yticklabels(ax.get_yticks())

    if tick is not 5:
        ylabels = [str(round(float(item.get_text()), 2))[0] + r'$\times$10$^{3}$' for item in ax.get_yticklabels()]

    elif tick is 5:
        ylabels = [str(round(float(item.get_text()), 2))[0] + r'$\times$10$^{5}$' for item in ax.get_yticklabels()]

    ax.set_yticklabels(ylabels)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()), 1))[:3] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    if row is 0:
        ax.set_xticklabels([''])

axes[0, 0].set_ylabel('Intensity')
axes[1, 0].set_ylabel('Intensity')
axes[1, 1].set_xlabel('Seconds')

fig_path = create_fig_path('droplet_chromatograph')  # save path
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# 7. bd10ss10_edb_ms vs. bd_edb_ms
expt_label = 'bd10ss10_edb_ms'
processed_file_name = expts[expt_label]['paths']['processed_data']
modeled_file_name = expts[expt_label]['paths']['modeled_data']
model_params_file_name = expts[expt_label]['paths']['model_parameters_data']
df_proc_bd10ss10 = import_treated_csv_data(processed_file_name, expt_label)
df_modeled_bd10ss10 = import_treated_csv_data(modeled_file_name, expt_label)
df_model_params_bd10ss10 = import_treated_csv_data(model_params_file_name, expt_label)

expt_label = 'bd_edb_ms'
processed_file_name = expts[expt_label]['paths']['processed_data']
modeled_file_name = expts[expt_label]['paths']['modeled_data']
model_params_file_name = expts[expt_label]['paths']['model_parameters_data']
df_proc_bd10 = import_treated_csv_data(processed_file_name, expt_label)
df_modeled_bd10 = import_treated_csv_data(modeled_file_name, expt_label)
df_model_params_bd10 = import_treated_csv_data(model_params_file_name, expt_label)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2.5))
plt.tight_layout()

axes = [ax0, ax1]
dfs = [[df_proc_bd10, df_modeled_bd10, df_model_params_bd10],
       [df_proc_bd10ss10, df_modeled_bd10ss10, df_model_params_bd10ss10]]
title_str = ['1.6 M BD', '1.6 M BD/1.8 M SS']

for tick in range(2):
    ax = axes[tick]
    df_list = dfs[tick]
    ax.plot(df_list[1].MINS_ELAPSED, df_list[1].MZ85_MZ283, color='0.25', lw=3, zorder=2,
            label='Best Model Fit')
    ax.fill_between(df_list[1].MINS_ELAPSED, df_list[1].MZ85_MZ283_MIN, df_list[1].MZ85_MZ283_MAX,
                    color='0.8', zorder=1, label='95\% Confidence')
    ax.scatter(df_list[0].MINS_ELAPSED, df_list[0].MZ85_MZ283, s=30, color='0.25', zorder=3,
               label='Observation')

    ax.set_yticklabels(ax.get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    bottom, top = ax.get_ylim()
    ax.set_ylim(ymin=0, ymax=top*1.2)

    k = df_list[2].k[0]
    k_se = df_list[2].k[1]
    annotation_str = "$\\tau_{BD}$ = %.1f $\pm$ %.1f hr" % (1 / (60 * k), k_se / (60 * k * k))
    ax.text(20, top, annotation_str, color='0.25', fontsize=10)

    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, ha='left')
    ax.set_title(title_str[tick])

ax0.set_ylabel('counts BD\n per PEG-6')
ax1.text(-100, -0.04, 'Minutes')
ax1.legend(fancybox=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.5, 1.5), fontsize=12)

fig_path = create_fig_path('butenedial_evaporation')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# 8. bd07as03_pr_edb_ms: time evaporated vs. butenedial and pyrrolinone signal (across solutions)
expt_label = 'bd07as03_pr_edb_ms'

processed_file_name = expts[expt_label]['paths']['processed_data']
df_processed = import_treated_csv_data(processed_file_name, expt_label)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2.5))
plt.tight_layout()

axes = [ax0, ax1]
mz_str = ['MZ85_MZ283', 'MZ84_MZ283']
molecule_str = ['BD', 'PR']

for tick in range(2):
    ax = axes[tick]

    ax.scatter(df_processed.trapped, df_processed[mz_str[tick]], s=30, c='0.25')
    ax.set_title('counts ' + molecule_str[tick] + '\nper PEG-6')

    ax.set_yticklabels(ax.get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    ax.set_ylim(ymin=0)

    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, ha='left')

ax1.text(-200, -0.015, 'Minutes in EDB')

fig_path = create_fig_path('aged_droplet_loss')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)

# 9. diazepine comparison
expt_label = 'bd10ag30_edb_ms'
df_droplet = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)

expt_label = 'bdnhph5_ms'
df_solution = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)

WTFRAC_PEG6_SOLUTION = expts[expt_label]['experimental_conditions']['solution_weight_fractions']['PEG-6']
MW_PEG6 = compounds['hexaethylene_glycol']['mw']
M_PEG6_SOLUTION = WTFRAC_PEG6_MS / MW_PEG6
M_PEG6_DROPLET = 1.4  # from aiomfac (see main text)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2.5))
plt.tight_layout()

dfs = [df_droplet, df_solution]
axes = [ax0, ax1]
title_strs = ['1.6 M BD/3.5 ppm NH$_{3}$', '0.9 M BD/0.45 M AS']
mpeg6 = [M_PEG6_DROPLET, M_PEG6_SOLUTION]

for tick in range(2):
    ax = axes[tick]
    df = dfs[tick]
    ax.scatter(df.MINS_ELAPSED, df.MZ149_MZ283 * mpeg6[tick], s=30, c='0.25')
    ax.set_title(title_strs[tick])

    ax.set_ylim(ymin=-0.001, ymax=0.021)
    ax.set_yticklabels(ax.get_yticks())
    labels = [str(round(float(item.get_text()), 2)) for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)

    if ax is ax1:
        ax.set_yticklabels([''])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=25))
    ax.set_xlim(xmin=-2)
    ax.set_xticklabels(ax.get_xticks())
    labels = [str(round(float(item.get_text()))) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

ax0.set_ylabel('DZ')
ax1.text(-75, -0.009, 'Minutes')

fig_path = create_fig_path('diazepine')
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)