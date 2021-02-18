import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from src.d00_utils.conf_utils import *
from src.d00_utils.data_utils import *
from src.d01_data.filter_ms_data import *
from src.d01_data.process_ms_data import *
from src.d01_data.cluster_ms_data import *
from src.d02_extraction.extract_least_sq_fit import block_coefficients_outside_confidence_interval
from src.d03_modeling.model_functions import *
from src.d03_modeling.toy_model import *
from src.d03_modeling.perform_ols import *

# NOTES ABOUT THIS SCRIPT
# Updated January 10, 2020 by Jack Hensley
# This script produces the datasets used in the "chemical regimes" paper from the "raw" NMR data
# Filtered, processed, and clustered data are operated on the observations
# I move through each experiment and derive "modeled" data
# These data are used for the plotting performed in plot_chemical_regimes_paper_data.py


# Load the compounds, water, and all the experiments of interest, given in the bulk_droplet_experiments.yml file
compounds, water = load_compounds()
expts_file_name = 'chemical_regimes_experiments.yml'
expts = load_experiments(expts_file_name)
experiment_labels = [*expts]

# Create the filtered, processed, and clustered data in a for loop for all experiments identified in expts
filter_raw_data_for_experiments(expts, save_filtered_data=True)
process_data_for_experiments(expts, compounds, save_processed_data=True)
cluster_data_for_experiments(expts, save_clustered_data=True)

# Perform the modeling (as needed) per experiment to obtain modeled data

# 1. bdph9, bdph10, bdph11, bdoh
# produce modeled data from first order kinetic fits of butenedial nmr data in each experiment
expt_labels = ['bdph9_nmr', 'bdph10_nmr', 'bdph11_nmr', 'bdoh_nmr']

for expt_label in expt_labels:
    file_name = expts[expt_label]['paths']['processed_data']
    df_processed = import_treated_csv_data(file_name, expt_label)

    t_min = df_processed.MINS_ELAPSED.min()  # start at t_min instead of 0 (no initial constraint)
    t_max = df_processed.MINS_ELAPSED.max()
    ts = np.arange(t_min, t_max)

    # initialize
    bd0 = df_processed.M_BUTENEDIAL[df_processed.MINS_ELAPSED == t_min].values[0]
    params = Parameters()  # initialize
    params.add('X_0', value=bd0, vary=True)
    params.add('k', value=0.001, min=0, max=100.)

    # produce modeled data sets
    model_data_with_odes(f_function=first_order, residuals_function=first_order_residuals,
                         solve_ode_function=first_order_ode, params=params,
                         experiments_dict=expts, experiment=expt_label,
                         x_col_name='MINS_ELAPSED',
                         y_col_names=['M_BUTENEDIAL'],
                         ts=ts, vars_init=[bd0],
                         confidence_interval=True, save_data=True)

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

# 2. bdasph9_nmr
# produce modeled data from system of odes that explains butenedial/nh3 chemical kinetics
expt_label = 'bdasph9_nmr'
file_name = expts[expt_label]['paths']['processed_data']
df_processed = import_treated_csv_data(file_name, expt_label)

t_min = df_processed['MINS_ELAPSED'].min()  # model starts at t_min rather than 0 mins
t_max = df_processed['MINS_ELAPSED'].max()
ts = np.arange(t_min, t_max)


# estimate ph; bd and nh initial conditions
def inverse(x, i0, i1, i2):
    """inverse function used for estimation of ph(t) and bd_0"""
    y = i0 / (x + i1) + i2
    return y


ph_inv_coefs, pcov = curve_fit(inverse, df_processed['MINS_ELAPSED'], df_processed['pH'],
                               [df_processed['pH'].max(), -1, df_processed['pH'].max()])
bd_inv_coefs, bcov = curve_fit(inverse, df_processed['MINS_ELAPSED'], df_processed['M_BUTENEDIAL'],
                               [df_processed['M_BUTENEDIAL'].max(), -1, df_processed['M_BUTENEDIAL'].max()])

ph0 = inverse(t_min, ph_inv_coefs[0], ph_inv_coefs[1], ph_inv_coefs[2])
bd0 = inverse(t_min, bd_inv_coefs[0], bd_inv_coefs[1], bd_inv_coefs[2])  # estimate of butenedial at t_min

# calculate nh0 (at t=t_min, not t=0!) from the loss of butenedial. assume butenedial and nh loss are 1:1.
nhi = expts[expt_label]['experimental_conditions']['solution_weight_fractions']['NH42SO4'] * 2 / 0.132
bdi = inverse(0, bd_inv_coefs[0], bd_inv_coefs[1], bd_inv_coefs[2])  # use the fit (more robust than the assumed comp)
nh0 = nhi - (bdi - bd0)

# initialize the parameters ("add" them and give them initial "values" as guesses, as well as constraints)
params = Parameters()
params.add('ph0', value=ph0, vary=False)
params.add('bd0', value=bd0, vary=True)
params.add('pr0', value=0, vary=True)
params.add('dm0', value=0, vary=True)
params.add('i0', value=ph_inv_coefs[0], vary=False)
params.add('i1', value=ph_inv_coefs[1], vary=False)
params.add('i2', value=ph_inv_coefs[2], vary=False)
params.add('k6', value=24.094253490238014, min=0, vary=True)
params.add('k7', value=1e8, min=0, vary=True)
params.add('k8', value=10, min=0, vary=True)
params.add('k9', value=0.45, min=0, vary=True)
params.add('k10', value=0.16296, min=0, vary=True)
params.add('initial_butenedial', value=bd0, vary=False)
params.add('initial_nhx', value=nh0, vary=False)
params.add('ai', value=a[0], vary=False)  # no fitting of disproportionation constants: variability insig. at this ph
params.add('aii', value=a[1], vary=False)
params.add('aiii', value=a[2], vary=False)

# custom-built function produces the modeled data with scipy
model_data_with_odes(f_function=bdasph9_f, residuals_function=bdasph9_residuals,
                     solve_ode_function=bdasph9_odes, params=params,
                     experiments_dict=expts, experiment=expt_label,
                     x_col_name='MINS_ELAPSED',
                     y_col_names=['pH', 'M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER'],
                     ts=ts, vars_init=[ph0, bd0, 0, 0],
                     confidence_interval=True, save_data=True)

# add estimate of NHx into the modeled data
file_name = expts[expt_label]['paths']['modeled_data']
df_modeled = import_treated_csv_data(file_name, expt_label)
df_modeled['M_NH'] = nhi - (bdi - df_modeled['M_BUTENEDIAL'])

# add pH uncertainty to the modeled data
covs = [pcov[0, 0], pcov[1, 1], pcov[2, 2]]
stds = np.sqrt(covs)
ses = stds / np.sqrt(len(df_processed))

df_coefs = pd.DataFrame()

for tick in range(3):
    df_coefs[tick] = np.random.normal(ph_inv_coefs[tick], ses[tick], 10000)

df_coefs = block_coefficients_outside_confidence_interval(df_coefs=df_coefs, ci=95)
solutions_array = np.empty([len(df_coefs), len(ts)])

# execute odes to get data for each row of synthesized coefficients, N_run iterations
for tick in range(len(df_coefs)):
    coefficients = df_coefs.iloc[tick].values[1:]
    solutions = inverse(ts, coefficients[0], coefficients[1], coefficients[2])
    solutions_array[tick, :] = solutions

# report the average values for each Y_COL_NAME for list of ts
solutions_max = np.max(solutions_array, 0)
solutions_min = np.min(solutions_array, 0)

df_modeled['pH_MIN'] = solutions_min
df_modeled['pH_MAX'] = solutions_max

save_data_frame(df_to_save=df_modeled, experiment_label=expt_label, level_of_treatment='LMFIT')

# 3. predictions of bdnhxph4 and bdnhxph8
# load the model params from the fitting
expt_label = 'bdasph9_nmr'
df_model_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)

coefs = []  # put into list so can be input into model for uncertainty: [[mean, std]....]
for col in df_model_params.columns:
    if col != 'METRIC':
        coefs.append([df_model_params[col][0], df_model_params[col][1]])

# perform fitting on bdnhxph4
expts = load_experiments('chemical_regimes_experiments.yml')
expt_label = 'bd07as03_bulk_nmr'
df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)

# initialize ph and initial conditions of experiment
ph0 = np.median(df_processed['pH'])
coefs_new = coefs
coefs_new[0] = [ph0, 0]
coefs_new[1] = coefs_new[12] = bd0 = [expts[expt_label]['experimental_conditions']['solution_weight_fractions'][
                                         'butenedial'] / 0.12, 0]
coefs_new[2] = coefs_new[3] = [0, 0]  # no starting content of products
coefs_new[4] = [0, 0]  # no pH change through experiment
coefs_new[13] = nh0 = [expts[expt_label]['experimental_conditions']['solution_weight_fractions']['NH42SO4'] * 2 / 0.132,
                       0]

ts = np.arange(0, 1500)  # df_processed.MINS_ELAPSED.max())

# custom-built function produces the modeled data with scipy
predict_data_with_odes(solve_ode_function=bdasph9_odes, ts=ts, coefs=coefs_new, experiment=expt_label,
                       x_col_name='MINS_ELAPSED',
                       y_col_names=['pH', 'M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER'],
                       confidence_interval=True, save_data=True)

# add estimate of NHx into the modeled data
file_name = expts[expt_label]['paths']['predicted_data']
df_predicted = import_treated_csv_data(file_name, expt_label)
df_predicted['M_NH'] = nh0[0] - (bd0[0] - df_predicted['M_BUTENEDIAL'])

save_data_frame(df_to_save=df_predicted, experiment_label=expt_label, level_of_treatment='PREDICTED')

# load the model params from the fitting
expt_label = 'bdasph9_nmr'
df_model_params = import_treated_csv_data(expts[expt_label]['paths']['model_parameters_data'], expt_label)

coefs = []  # put into list so can be input into model for uncertainty: [[mean, std]....]
for col in df_model_params.columns:
    if col != 'METRIC':
        coefs.append([df_model_params[col][0], df_model_params[col][1]])

# perform fitting on bdnhxph8
expts = load_experiments('chemical_regimes_experiments.yml')
expt_label = 'bdahph9_nmr'
df_processed = import_treated_csv_data(expts[expt_label]['paths']['processed_data'], expt_label)

# initialize ph and initial conditions of experiment
guess = [df_processed['pH'].max(), np.median(df_processed['MINS_ELAPSED']),
         np.median(df_processed['MINS_ELAPSED'])]
ph_coef, pcov = curve_fit(inverse, df_processed['MINS_ELAPSED'],
                           df_processed['pH'], guess)
ph0 = inverse(0, ph_coef[0], ph_coef[1], ph_coef[2])

coefs_new = coefs  # reset
coefs_new[0] = [round(ph0, 2), 0]  # dont know why but it fails if not rounded....
coefs_new[1] = coefs_new[12] = bd0 = [expts[expt_label]['experimental_conditions'][
                                          'solution_weight_fractions']['butenedial'] / 0.12, 0]
coefs_new[2] = coefs_new[3] = [0, 0]
coefs_new[4] = [ph_coef[0], 0]
coefs_new[5] = [ph_coef[1], 0]
coefs_new[6] = [ph_coef[2], 0]
coefs_new[13] = nh0 = [expts[expt_label]['experimental_conditions']['solution_weight_fractions']['NH4OH'] / 0.035, 0]
coefs_new.insert(4, [0, 0])


ts = np.arange(0, df_processed.MINS_ELAPSED.max() + 5, 0.1)
predict_data_with_odes(solve_ode_function=bdahph9_odes, ts=ts, coefs=coefs_new, experiment=expt_label,
                       x_col_name='MINS_ELAPSED',
                       y_col_names=['pH', 'M_BUTENEDIAL', 'M_C4H5NO', 'M_DIMER', 'M_BD_OH'],
                       confidence_interval=True, save_data=True)

# add estimate of NHx into the modeled data
file_name = expts[expt_label]['paths']['predicted_data']
df_predicted = import_treated_csv_data(file_name, expt_label)
df_predicted['M_NH'] = nh0[0] - (bd0[0] - df_predicted['M_BUTENEDIAL'])

# add pH uncertainty to the modeled data
covs = [pcov[0, 0], pcov[1, 1], pcov[2, 2]]
stds = np.sqrt(covs)
ses = stds / np.sqrt(len(df_processed))

df_coefs = pd.DataFrame()

for tick in range(3):
    df_coefs[tick] = np.random.normal(ph_coef[tick], ses[tick], 10000)

df_coefs = block_coefficients_outside_confidence_interval(df_coefs=df_coefs, ci=95)
solutions_array = np.empty([len(df_coefs), len(ts)])

# execute odes to get data for each row of synthesized coefficients, N_run iterations
for tick in range(len(df_coefs)):
    coefficients = df_coefs.iloc[tick].values[1:]
    solutions = inverse(ts, coefficients[0], coefficients[1], coefficients[2])
    solutions_array[tick, :] = solutions

# report the average values for each Y_COL_NAME for list of ts
solutions_max = np.max(solutions_array, 0)
solutions_min = np.min(solutions_array, 0)

df_predicted['pH_MIN'] = solutions_min
df_predicted['pH_MAX'] = solutions_max

save_data_frame(df_to_save=df_predicted, experiment_label=expt_label, level_of_treatment='PREDICTED')
