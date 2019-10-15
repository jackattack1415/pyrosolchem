from lmfit import Parameters

from src.d00_utils.conf_utils import *
from src.d01_data.clean_ms_data import filter_and_clean_data_in_experiment
from src.d01_data.cluster_ms_data import create_clustered_statistics_dataframe
from src.d01_data.process_ms_data import *
from src.d01_data.perform_ols import *
from src.d02_extraction.extract_least_sq_fit import model_data_with_odes, g_bd_pyr_dia, f_residual

compounds, water = load_compounds()

# make data for pyrrolinone evaporation
pyr_evap_expt_label = 'pyr_evap_nh42so4'
pyr_evap_params = load_experiments([pyr_evap_expt_label])

raw_pyr_evap_file_name = pyr_evap_params[pyr_evap_expt_label]['paths']['raw_data']
df_cleaned_pyr_evap = filter_and_clean_data_in_experiment(ms_file_name=raw_pyr_evap_file_name,
                                                          experiments_dict=pyr_evap_params,
                                                          save_cleaned_data=True)

cleaned_pyr_evap_file_name = pyr_evap_params[pyr_evap_expt_label]['paths']['cleaned_data']
df_processed_pyr_evap = process_ms_data_in_pyrrolinone_evap_experiments(cleaned_ms_file_name=cleaned_pyr_evap_file_name,
                                                                        experiments_dict=pyr_evap_params,
                                                                        save_processed_data=True)

processed_pyr_evap_file_name = pyr_evap_params[pyr_evap_expt_label]['paths']['processed_data']
df_clustered_pyr_evap = create_clustered_statistics_dataframe(experiment_dict=pyr_evap_params,
                                                              col_to_cluster='hrs',
                                                              y_cols_to_keep=['mz84_mz283'],
                                                              max_points_per_cluster=6,
                                                              save_clustered_data=True)

df_ols_pyr_evap = create_ordinary_least_squares_data(processed_ms_file_name=processed_pyr_evap_file_name,
                                                     experiments_dict=pyr_evap_params,
                                                     x_col_name='hrs', y_col_name='mz84_mz283',
                                                     take_log=True, save_data=True, experiment=None)

# make data for droplet vs. vial experiments
bd_as_expt_labels = ['bd_rxn_nh42so4_vial', 'bd_rxn_nh42so4_droplet']
bd_as_params = load_experiments(bd_as_expt_labels)

for label in bd_as_expt_labels:
    params = {k: v for k, v in bd_as_params.items() if k in label}
    raw_bd_as_file_name = bd_as_params[label]['paths']['raw_data']
    df_cleaned_bd_as = filter_and_clean_data_in_experiment(ms_file_name=raw_bd_as_file_name,
                                                           experiments_dict=params,
                                                           save_cleaned_data=True)

    cleaned_bd_as_file_name = bd_as_params[label]['paths']['cleaned_data']
    df_processed_bd_as = process_ms_data_in_droplet_vs_vial_experiments(cleaned_ms_file_name=cleaned_bd_as_file_name,
                                                                        experiments_dict=params,
                                                                        save_processed_data=True)


# make data for butenedial evaporation plots to use
bd_evap_labels = ['bd_evap_wet', 'bd_evap_low_na2so4', 'bd_evap_high_na2so4']
max_points_to_cluster = [5, 3, 3]
bd_evap_params = load_experiments(bd_evap_labels)

count = 0
for label in bd_evap_labels:
    params = {k: v for k, v in bd_evap_params.items() if k in label}
    raw_bd_evap_file_name = bd_evap_params[label]['paths']['raw_data']
    df_cleaned_bd_evap = filter_and_clean_data_in_experiment(ms_file_name=raw_bd_evap_file_name,
                                                             experiments_dict=params,
                                                             save_cleaned_data=True)

    cleaned_bd_evap_file_name = bd_evap_params[label]['paths']['cleaned_data']
    df_processed_bd_evap = process_ms_data_in_evap_experiments(cleaned_ms_file_name=cleaned_bd_evap_file_name,
                                                               experiments_dict=params,
                                                               save_processed_data=True)

    processed_bd_evap_file_name = bd_evap_params[label]['paths']['processed_data']
    df_clustered_bd_evap = create_clustered_statistics_dataframe(experiment_dict=params,
                                                                  col_to_cluster='hrs',
                                                                  y_cols_to_keep=['mol85_mol283'],
                                                                  max_points_per_cluster=max_points_to_cluster[count],
                                                                  save_clustered_data=True)

    df_ols_bd_evap = create_ordinary_least_squares_data(processed_ms_file_name=processed_bd_evap_file_name,
                                                        experiments_dict=params,
                                                        x_col_name='hrs', y_col_name='mol85_mol283',
                                                        take_log=True, save_data=True, experiment=None)

    count += 1


# make data for butenedial with ammonia bubbled over
bd_rxn_nh3g_label = 'bd_rxn_nh3g_droplet'
bd_rxn_nh3g_params = load_experiments([bd_rxn_nh3g_label])

raw_bd_rxn_nh3g_file_name = bd_rxn_nh3g_params[bd_rxn_nh3g_label]['paths']['raw_data']
df_cleaned_bd_rxn_nh3g = filter_and_clean_data_in_experiment(ms_file_name=raw_bd_rxn_nh3g_file_name,
                                                             experiments_dict=bd_rxn_nh3g_params,
                                                             save_cleaned_data=True)

cleaned_bd_rxn_nh3g_file_name = bd_rxn_nh3g_params[bd_rxn_nh3g_label]['paths']['cleaned_data']
df_processed_bd_rxn_nh3g = process_ms_data_in_nh3g_experiments(cleaned_ms_file_name=cleaned_bd_rxn_nh3g_file_name,
                                                               experiments_dict=bd_rxn_nh3g_params,
                                                               save_processed_data=True)

processed_bd_rxn_nh3g_file_name = bd_rxn_nh3g_params[bd_rxn_nh3g_label]['paths']['processed_data']
df_clustered_bd_rxn_nh3g = create_clustered_statistics_dataframe(experiment_dict=bd_rxn_nh3g_params,
                                                                 col_to_cluster='mins',
                                                                 y_cols_to_keep=[
                                                                     'mz84_mz283', 'mz85_mz283', 'mz149_mz283'],
                                                                 max_points_per_cluster=3,
                                                                 save_clustered_data=True)




B_0 = 0.033
P_0 = D_0 = 0
X_0 = [P_0, B_0, D_0]

params = Parameters()
params.add('B_0', value=B_0, vary=False)
params.add('P_0', value=P_0, vary=False)
params.add('D_0', value=D_0, vary=False)
params.add('k0', value=5, min=0, max=10.)
params.add('k1', value=5, min=0, max=10.)
params.add('k_LD', value=2, min=0, max=2.)
params.add('a', value=5, min=0.0001, max=20.)
params.add('b', value=3.5, min=0.0001, max=20.)

t_max = int(np.nanmax(df_processed_bd_rxn_nh3g.mins.values) + 5)
ts = np.linspace(0, t_max, t_max + 1)
df_modeled_bd_rxn_nh3g = model_data_with_odes(g_function=g_bd_pyr_dia, residual_function=f_residual, params=params,
                                              experiments_dict=bd_rxn_nh3g_params, x_col_name='mins',
                                              y_col_names=['mz84_mz283', 'mz85_mz283', 'mz149_mz283'], ts=ts,
                                              vars_init=X_0, save_data=True)
