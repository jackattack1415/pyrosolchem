import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import statsmodels.api as sm

from src.d00_utils.conf_utils import *
from src.d01_data.filter_ms_data import *
from src.d01_data.process_ms_data import *
from src.d01_data.cluster_ms_data import *
from src.d03_modeling.model_functions import *
from src.d03_modeling.toy_model import *
from src.d03_modeling.perform_ols import *


compounds, water = load_compounds()
expts_file_name = 'bulk_droplet_experiments.yml'
expts = load_experiments(expts_file_name)
experiment_labels = [*expts]

# Create the filtered and processed data in a for loop for all experiments identified in expts

filter_raw_data_for_experiments(expts, save_filtered_data=True)
process_data_for_experiments(expts, compounds, save_processed_data=True)
cluster_data_for_experiments(expts, save_clustered_data=True)

# Perform the modeling (as needed) for each experiment

# 1. create the calibration factors to convert bdasph9_ms to M, using the bdasph9_nmr
expts_file_name = 'chemical_regimes_experiments.yml'
expts = load_experiments(expts_file_name)
expt_labels = ['bdasph9_nmr', 'bdasph9_ms']
df_nmr_processed = import_treated_csv_data(expts[expt_labels[0]]['paths']['processed_data'], expt_labels[0])
df_ms_processed = import_treated_csv_data(expts[expt_labels[1]]['paths']['processed_data'], expt_labels[1])

nmr_cols = ['M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER']
ms_cols = ['MZ85_MZ283', 'MZ84_MZ283', 'MZ150_MZ283']

WTFRAC_PEG6_MS = expts[expt_labels[1]]['experimental_conditions']['solution_weight_fractions']['PEG-6']
MW_PEG6 = compounds['hexaethylene_glycol']['mw']
M_PEG6_MS = WTFRAC_PEG6_MS / MW_PEG6

# make the clustered data for the calibration to be performed on, for both nmr and ms data
N_clusters = 6
df_with_clusters = add_clusters_to_dataframe(df_nmr_processed, N_clusters=N_clusters)
df_means = df_with_clusters.groupby('clusters', as_index=False).mean()
df_stds = df_with_clusters.groupby('clusters', as_index=False).std()
df_nmr_clustered = pd.merge(df_means, df_stds, on=None,
                            suffixes=('', '_std'), left_index=True, right_index=True, how='outer')
df_nmr_clustered = df_nmr_clustered.drop(columns=['clusters', 'clusters_std'])
df_nmr_clustered.sort_values(by='MINS_ELAPSED', ascending=True, inplace=True)

df_ms_clustered = pd.DataFrame()
for tick in range(N_clusters):
    t = df_nmr_clustered.MINS_ELAPSED[tick]
    subset = df_ms_processed.iloc[(df_ms_processed.MINS_ELAPSED-t).abs().argsort()[:5]]
    subset.clusters = tick
    subset.drop(subset[(subset.MINS_ELAPSED >= 1.2*t) | (subset.MINS_ELAPSED <= 0.8*t)
                      & (subset.clusters == tick)].index, inplace=True)
    mean = subset.mean(axis=0).rename(str(tick)).to_frame().transpose()
    std = subset.std(axis=0).rename(str(tick)).to_frame().transpose()
    combine = pd.merge(mean, std, on=None,
                       suffixes=('', '_std'), left_index=True, right_index=True, how='outer')
    df_ms_clustered = df_ms_clustered.append(combine)

df_ms_clustered.sort_values(by='MINS_ELAPSED', ascending=True, inplace=True)


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

    # create new columns with the calibrated data in the df_clustered files
    df_ms_clustered[nmr_cols[tick]] = df_ms_clustered[ms_cols[tick]] * cf * M_PEG6_MS
    df_ms_clustered[nmr_cols[tick] + '_std'] = df_ms_clustered[ms_cols[tick] + '_std'] * cf * M_PEG6_MS

save_data_frame(df_to_save=df_nmr_clustered, experiment_label=expt_labels[0], level_of_treatment='CLUSTERED')
save_data_frame(df_to_save=df_ms_clustered, experiment_label=expt_labels[1], level_of_treatment='CLUSTERED')

# 2. model butenedial particle + nh3 gas: bd10ag30_edb_ms experiment

# produce modeled data
# import the modeling mechanism (from bdasph9 and the bdoh fittings)
expts_file_name = 'chemical_regimes_experiments.yml'
expts = load_experiments(expts_file_name)
expt_label = 'bdasph9_nmr'
file_name = expts[expt_label]['paths']['model_parameters_data']
model_parameters = import_treated_csv_data(file_name, expt_label)

# add parameters: k[0] = mean, k[1] = se
k6 = model_parameters['k6']
k7 = model_parameters['k7']
k8 = model_parameters['k8']
k9 = model_parameters['k9']
k10 = model_parameters['k10']
ke = [1 / (2.6 * 60), (0.8 / 2.6) * 1 / (2.6 * 60)]  # take k = tau_evap ** -1 and convert to min

# reporting the disproportionation empirical fitting (k = f(ph)) from make_chemical_regimes_paper_data.py
ai = 15.5
aii = 64.6
aiii = 1.61e4

# import and calibrate bd10ag30_edb_ms
expts_file_name = 'bulk_droplet_experiments.yml'
expts = load_experiments(expts_file_name)
expt_label = 'bd10ag30_edb_ms'
df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)

nmr_cols = ['M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER']
ms_cols = ['MZ85_MZ283', 'MZ84_MZ283', 'MZ150_MZ283']

# add column for M_PEG6 since there are different values for the solutions used
M_PEG6_PARTICLE = 1.4  # from aiomfac for particles of interest
M_BUTENEDIAL0_PARTICLE = 1.6  # also from aiomfac

for tick in range(3):
    df_processed[nmr_cols[tick]] = df_processed[ms_cols[tick]] * cfs[tick] * M_PEG6_PARTICLE
    df_processed[nmr_cols[tick] + '_std'] = df_processed[ms_cols[tick]] * cf_ses[tick] * M_PEG6_PARTICLE

save_data_frame(df_to_save=df_processed, experiment_label=expt_label, level_of_treatment='PROCESSED')

# make the modeled data: take the best fit ph=6.9 (see 20200118_predicting_butenedial_nh3g_droplets.ipynb)
t_max = df_processed.MINS_ELAPSED.max()
ts = np.arange(0, 90)
pH = 6.9

coefs = [[M_BUTENEDIAL0_PARTICLE, 0], [0, 0], [0, 0], [df_processed.M_AMMONIA.unique()[0], 0], [pH, 0],
         ke, k6, k7, k8, k9, k10, [ai, 0], [aii, 0], [aiii, 0]]  # report here as lists: [mean, se]

predict_data_with_odes(bd10ag30_edb_ms, ts, coefs, experiment=expt_label,
                       x_col_name='MINS_ELAPSED', y_col_names=['M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER'],
                       confidence_interval=True, save_data=True)

file_name = expts[expt_label]['paths']['predicted_data']
df_predicted = import_treated_csv_data(file_name, expt_label)
df_predicted['pH'] = pH
df_predicted['M_AMMONIA'] = df_processed.M_AMMONIA.unique()[0]
save_data_frame(df_to_save=df_predicted, experiment_label=expt_label, level_of_treatment='PREDICTED')


# 3. first-order butenedial evaporation
# bd10ss10_edb_ms
expt_label = 'bd10ss10_edb_ms'
file_name = expts[expt_label]['paths']['processed_data']
df_processed_bd10ss10 = import_treated_csv_data(file_name, expt_label)
X_0 = np.mean(df_processed_bd10ss10.MZ85_MZ283[df_processed_bd10ss10.MINS_ELAPSED < 10].values[0])

params = Parameters()
params.add('X_0', value=X_0, vary=True)
params.add('k', value=0.001, min=0, max=10.)
t_max = int(np.max(df_processed_bd10ss10.MINS_ELAPSED) + 5)
ts = np.linspace(0, t_max, t_max + 1)

model_data_with_odes(f_function=first_order, residuals_function=first_order_residuals,
                     solve_ode_function=first_order_ode, params=params,
                     experiments_dict=expts, experiment=expt_label,
                     x_col_name='MINS_ELAPSED', y_col_names=['MZ85_MZ283'], ts=ts,
                     vars_init=X_0, confidence_interval=True, save_data=True)

# bd_edb_ms
expt_label = 'bd_edb_ms'
file_name = expts[expt_label]['paths']['processed_data']
df_processed_bd = import_treated_csv_data(file_name, expt_label)
X_0 = np.mean(df_processed_bd.MZ85_MZ283[df_processed_bd.MINS_ELAPSED < 10].values[0])

params = Parameters()
params.add('X_0', value=X_0, vary=True)
params.add('k', value=0.001, min=0, max=10.)
t_max = int(np.max(df_processed_bd.MINS_ELAPSED) + 5)
ts = np.linspace(0, t_max, t_max + 1)

model_data_with_odes(f_function=first_order, residuals_function=first_order_residuals,
                     solve_ode_function=first_order_ode, params=params,
                     experiments_dict=expts, experiment=expt_label,
                     x_col_name='MINS_ELAPSED', y_col_names=['MZ85_MZ283'], ts=ts,
                     vars_init=X_0, confidence_interval=True, save_data=True)
