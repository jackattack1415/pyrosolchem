import pandas as pd

from src.d00_utils.processing_utils import get_bootstrapped_statistics
from src.d00_utils.data_utils import extract_calibration_data, save_data_frame, import_ms_data


def process_ms_data_in_evap_experiments(cleaned_ms_file_name, experiments_dict, save_processed_data=False):
    """
    """

    experiment_name = [*experiments_dict].pop()
    df_cleaned = import_ms_data(file_name=cleaned_ms_file_name,
                                directory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)
    df_processed = add_calibrated_ms_data_columns(df=df_processed,
                                                  experiments_dict=experiments_dict,
                                                  analyte='Butenedial',
                                                  internal_standard='PEG-6')

    df_processed = df_processed.rename(columns={'mins': 'hrs'})
    df_processed.hrs = df_processed.hrs / 60

    if save_processed_data:
        save_data_frame(df_to_save=df_processed,
                        experiment_label=experiment_name,
                        level_of_cleaning='PROCESSED')

    return df_processed


def process_ms_data_in_droplet_vs_vial_experiments(cleaned_ms_file_name, experiments_dict, save_processed_data=False):
    """
    """

    experiment_name = [*experiments_dict].pop()
    df_cleaned = import_ms_data(file_name=cleaned_ms_file_name,
                                directory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)
    df_processed['hrs'] = (df_processed.mins + df_processed.vial) / 60
    df_processed = df_processed.drop(['vial', 'mins'], axis=1)

    if save_processed_data:
        save_data_frame(df_to_save=df_processed,
                        experiment_label=experiment_name,
                        level_of_cleaning='PROCESSED')

    return df_processed


def process_ms_data_in_pyrrolinone_evap_experiments(cleaned_ms_file_name, experiments_dict, save_processed_data=False):
    """
    """

    experiment_name = [*experiments_dict].pop()
    df_cleaned = import_ms_data(file_name=cleaned_ms_file_name,
                                directory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)

    df_processed = df_processed.rename(columns={'mins': 'hrs', 'vial': 'hrs_in_vial'})
    df_processed.hrs = df_processed.hrs / 60
    df_processed.hrs_in_vial = df_processed.hrs_in_vial / 60
    df_processed['experiment'] = "pyr_nh42so4"
    df_processed = df_processed[df_processed.hrs < 3]

    if save_processed_data:
        save_data_frame(df_to_save=df_processed,
                        experiment_label=experiment_name,
                        level_of_cleaning='PROCESSED')

    return df_processed


def process_ms_data_in_nh3g_experiments(cleaned_ms_file_name, experiments_dict, save_processed_data=False):
    """
    """
    experiment_name = [*experiments_dict].pop()
    df_cleaned = import_ms_data(file_name=cleaned_ms_file_name,
                                directory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)
    df_processed['experiment'] = 'bd_nh3g_' + df_processed.nominal_nh3_mM.astype(str).str.replace('.','')

    # df_processed['mM_nhx']  add this to the droplet definitions soon

    if save_processed_data:
        save_data_frame(df_to_save=df_processed,
                        experiment_label=experiment_name,
                        level_of_cleaning='PROCESSED')

    return df_processed


def add_calibrated_ms_data_columns(df, experiments_dict, analyte, internal_standard='PEG-6'):
    """"""

    df_calibrated = pd.DataFrame()
    for experiment_name, experiment_defs in experiments_dict.items():

        df_experiment = df[df.experiment == experiment_name]
        ms_signal_inits = extract_calibration_data(df=df_experiment,
                                                   t_init_cutoff=experiment_defs['cal_data_time'],
                                                   cal_data_col=experiment_defs['y_col'])
        ms_signal_inits_avg, ms_signal_inits_rel_std = get_bootstrapped_statistics(ms_signal_inits)

        for compound_name, compound_mol_frac in experiment_defs['composition'].items():
            internal_standard_mol_frac = experiment_defs['composition'][internal_standard]

            if compound_name == analyte:
                rel_molar_abundance_in_solution = compound_mol_frac / internal_standard_mol_frac
                cal_factor_avg = rel_molar_abundance_in_solution / ms_signal_inits_avg
                cal_factor_std = ms_signal_inits_rel_std

                cal_data_col = experiment_defs['y_col'].replace('mz', 'mol')

                df_experiment[cal_data_col] = df_experiment[experiment_defs['y_col']] * cal_factor_avg
                df_experiment[cal_data_col + '_std'] = df_experiment[cal_data_col] * cal_factor_std

                decay_data_col = cal_data_col.split('/')[0] + '/' + cal_data_col.split('/')[0] + '_0'
                df_experiment[decay_data_col] = df_experiment[cal_data_col] / \
                                                rel_molar_abundance_in_solution

        df_calibrated = df_calibrated.append(df_experiment)

    return df_calibrated
