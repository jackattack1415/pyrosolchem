import pandas as pd
import numpy as np

from src.d00_utils.conf_utils import *
from src.d01_data.filter_ms_data import *
from src.d01_data.process_ms_data import *
from src.d01_data.cluster_ms_data import *
from src.d03_modeling.model_functions import *
from src.d03_modeling.toy_model import *
from src.d03_modeling.perform_ols import *

# Load the compounds, water, and all the experiments of interest, given in the bulk_droplet_experiments.yml file

compounds, water = load_compounds()
expts_file_name = 'bulk_droplet_experiments.yml'
expts = load_experiments(expts_file_name)
experiment_labels = [*expts]

# Create the filtered and processed data in a for loop for all experiments identified in expts

filter_raw_data_for_experiments(expts, save_filtered_data=True)
process_data_for_experiments(expts, compounds, save_processed_data=True)
# cluster_data_for_experiments(expts, save_clustered_data=True)

# Perform the modeling (as needed) for each experiment

# # bd07as03_nmr_bulk
# expt_label = 'bd07as03_bulk_nmr'
# file_name = expts[expt_label]['paths']['processed_data']
# df_processed_bd07as03_bulk_nmr = import_treated_csv_data(file_name, expt_label)
# X_0 = np.mean(df_processed_bd07as03_bulk_nmr.M_BUTENEDIAL[df_processed_bd07as03_bulk_nmr.MINS_ELAPSED < 25].values[0])
#
# params = Parameters()
# params.add('X_0', value=X_0, vary=True)
# params.add('k', value=0.001, min=0, max=10.)
# t_max = int(np.max(df_processed_bd07as03_bulk_nmr.MINS_ELAPSED) + 5)
# ts = np.linspace(0, t_max, t_max + 1)
#
# model_data_with_odes(f_function=first_order, residuals_function=first_order_residuals,
#                      solve_ode_function=first_order_ode, params=params,
#                      experiments_dict=expts, experiment=expt_label,
#                      x_col_name='MINS_ELAPSED', y_col_names=['M_BUTENEDIAL'], ts=ts,
#                      vars_init=X_0, confidence_interval=True, save_data=True)


# bd07as03_pr_edb_ms
expt_label = 'bd07as03_pr_edb_ms'
file_name = expts[expt_label]['paths']['processed_data']
df_processed_pr_evap = import_treated_csv_data(file_name, expt_label)
X_0 = np.mean(df_processed_pr_evap.MZ84_MZ283[df_processed_pr_evap.trapped < 7].values[0])

params = Parameters()
params.add('X_0', value=X_0, vary=True)
params.add('k', value=0.001, min=0, max=10.)
t_max = int(np.max(df_processed_pr_evap.trapped) + 5)
ts = np.linspace(0, t_max, t_max + 1)

model_data_with_odes(f_function=first_order, residuals_function=first_order_residuals,
                     solve_ode_function=first_order_ode, params=params,
                     experiments_dict=expts, experiment=expt_label,
                     x_col_name='trapped', y_col_names=['MZ84_MZ283'], ts=ts,
                     vars_init=[X_0], confidence_interval=True, save_data=True)



# # bd10ss10_edb_ms
# expt_label = 'bd10ss10_edb_ms'
# file_name = expts[expt_label]['paths']['processed_data']
# df_processed_bd10ss10 = import_treated_csv_data(file_name, expt_label)
# X_0 = np.mean(df_processed_bd10ss10.MZ85_MZ283[df_processed_bd10ss10.MINS_ELAPSED < 10].values[0])
#
# params = Parameters()
# params.add('X_0', value=X_0, vary=True)
# params.add('k', value=0.001, min=0, max=10.)
# t_max = int(np.max(df_processed_bd10ss10.MINS_ELAPSED) + 5)
# ts = np.linspace(0, t_max, t_max + 1)
#
# model_data_with_odes(f_function=first_order, residuals_function=first_order_residuals,
#                      solve_ode_function=first_order_ode, params=params,
#                      experiments_dict=expts, experiment=expt_label,
#                      x_col_name='MINS_ELAPSED', y_col_names=['MZ85_MZ283'], ts=ts,
#                      vars_init=X_0, confidence_interval=True, save_data=True)
