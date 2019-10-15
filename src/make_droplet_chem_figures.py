from src.d00_utils.conf_utils import *
from src.d00_utils.data_utils import import_ms_data
from src.d05_reporting.plot_ms_data import (plot_ms_data_across_experiments, plot_ms_data_with_break, \
    subplot_ms_data_across_experiments)


compounds, water = load_compounds()

# make plot for pyrrolinone evaporation
pyr_evap_expt_label = 'pyr_evap_nh42so4'
pyr_evap_params = load_experiments([pyr_evap_expt_label])

pyr_evap_modeled_data_path = pyr_evap_params[pyr_evap_expt_label]['paths']['modeled_data']
df_modeled_pyr_evap = import_ms_data(pyr_evap_modeled_data_path, subdirectory=pyr_evap_expt_label)
tau_pyr_evap = df_modeled_pyr_evap.tau[0]

plot_ms_data_across_experiments(experiments_dict=pyr_evap_params, x_col_name='hrs', y_col_names=['mz84_mz283'],
                                series_labels=[['Pyrrolinone']], series_colors=[['green']],
                                x_label='Droplet residence time (hrs)', y_label='Normalized signal',
                                series_title=None, save_fig=True, ax=None,
                                add_clusters=True, add_modeled=True, y_model_col_names=['mz84_mz283'],
                                model_label='first order loss ($\\tau = %.1f$ hrs)' % tau_pyr_evap)


# make plot for butenedial and ammonium sulfate droplet vs. vial reactions
bd_as_expt_labels = ['bd_rxn_nh42so4_vial', 'bd_rxn_nh42so4_droplet']
bd_as_vial_label = ['bd_rxn_nh42so4_vial']
bd_as_params = load_experiments(bd_as_expt_labels)
bd_as_vial_params = load_experiments(bd_as_vial_label)
series_labels = [['Bulk'], ['Droplet']]
series_colors = [['darkgreen'], ['lightgreen']]

plot_ms_data_with_break(experiments_dict=bd_as_params, x_col_name='hrs', y_col_names=['mz84_mz283'],
                        series_labels=series_labels, series_colors=series_colors,
                        x_label='Reaction time (hrs)', y_label='Pyrrolinone \n (normalized signal)',
                        left_xlims=[-0.5, 5.5], right_xlims=[13, 19],
                        series_title='Reaction Medium', save_fig=True)

plot_ms_data_with_break(experiments_dict=bd_as_vial_params, x_col_name='hrs', y_col_names=['mz84_mz283'],
                        series_labels=[['Bulk']], series_colors=[['darkgreen']],
                        x_label='Reaction time (hrs)', y_label='Pyrrolinone \n (normalized signal)',
                        left_xlims=[-0.5, 5.5], right_xlims=[13, 19],
                        series_title='Reaction Medium', save_fig=True)


# make plots for butenedial evaporation
bd_evap_labels = ['bd_evap_wet', 'bd_evap_low_na2so4', 'bd_evap_high_na2so4']
bd_evap_params = load_experiments(bd_evap_labels)
bd_series_colors = ['red', 'hotpink', 'pink']

count = 0
for label in bd_evap_labels:
    params = {k: v for k, v in bd_evap_params.items() if k in label}
    bd_evap_modeled_data_path = params[label]['paths']['modeled_data']
    df_modeled_bd_evap = import_ms_data(bd_evap_modeled_data_path, subdirectory=label)
    tau_bd_evap = df_modeled_bd_evap.tau[0]

    plot_ms_data_across_experiments(experiments_dict=params, x_col_name='hrs', y_col_names=['mol85_mol283'],
                                    series_labels=[['Butenedial']], series_colors=[[bd_series_colors[count]]],
                                    x_label='Droplet residence time (hrs)', y_label='n$_{BD}$ / n$_{PEG6}$',
                                    series_title=None, save_fig=True, ax=None,
                                    add_clusters=True, add_modeled=True, y_model_col_names=['mol85_mol283'],
                                    model_label='first order loss ($\\tau = %.1f$ hrs)' % tau_bd_evap)

    count += 1


# make plots for butenedial + nh3g reaction experiments
bd_rxn_nh3g_label = 'bd_rxn_nh3g_droplet'
bd_rxn_nh3g_params = load_experiments([bd_rxn_nh3g_label])

subplot_ms_data_across_experiments(bd_rxn_nh3g_params, x_col_name='mins',
                                   y_col_names=['mz85_mz283', 'mz84_mz283', 'mz149_mz283'],
                                   series_labels=[['Butenedial', 'Pyrrolinone', 'Diazepine']],
                                   series_colors=[['red', 'green', 'blue']],
                                   big_x_label='Droplet residence time (mins)', big_y_label='Normalized signal',
                                   save_fig=True, legend=True,
                                   add_clusters=True, add_modeled=True,
                                   y_model_col_names=['mz85_mz283', 'mz84_mz283', 'mz149_mz283'],
                                   model_label='model fit')

plot_ms_data_across_experiments(experiments_dict=bd_rxn_nh3g_params, x_col_name='mins',
                                y_col_names=['mz84_mz283', 'mz149_mz283'],
                                series_labels=[['Pyrrolinone', 'Diazepine']],
                                series_colors=[['green', 'blue']],
                                x_label='Droplet residence time (mins)', y_label='Normalized signal',
                                series_title=None, save_fig=True, ax=None,
                                add_clusters=True, add_modeled=False)