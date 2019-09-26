import pandas as pd

from src.d00_utils.processing_utils import get_bootstrapped_statistics
from src.d00_utils.data_utils import extract_calibration_data


def process_ms_data_in_evap_experiments(df_cleaned, experiments):
    """
    """

    df_processed = df_cleaned.copy(deep=True)
    df_processed = add_calibrated_ms_data_columns(df=df_processed,
                                                  experiments=experiments,
                                                  analyte='Butenedial',
                                                  internal_standard='PEG-6')

    df_processed.rename(columns={'mins': 'hrs'})
    df_processed.hrs = df_cleaned.hrs / 60

    return df_processed


def process_ms_data_in_droplet_vs_vial_experiments(df_cleaned):
    """
    """

    df_processed = df_cleaned.copy(deep=True)
    df_processed['hrs'] = (df_processed.mins + df_processed.vial) / 60
    df_processed.drop(['vial', 'mins'], axis=1)

    return df_processed


def process_ms_data_in_nh3g_experiments(df_cleaned):
    """
    """

    df_processed = df_cleaned.copy(deep=True)
    df_processed['experiment'] = 'bd_nh3g_' + df_processed.nh3_bubbler.astype(str).str.replace('.','')

    # df_processed['mM_nhx']  add this to the droplet definitions soon

    return df_processed



def add_calibrated_ms_data_columns(df, experiments, analyte, internal_standard='PEG-6'):
    """"""

    df_calibrated = pd.DataFrame()
    for experiment_name, experiment_defs in experiments.items():

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
