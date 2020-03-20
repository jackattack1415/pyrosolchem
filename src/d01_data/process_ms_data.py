import pandas as pd
import numpy as np

from src.d00_utils.processing_utils import get_bootstrapped_statistics
from src.d00_utils.calc_utils import calculate_molarity_from_weight_fraction
from src.d00_utils.data_utils import *


def process_data_for_experiments(experiments_dict, compounds, save_processed_data=False):
    """ Adds columns (processing) while removing unnecessary columns (cleaning) from the dataframe based on
    function input strings and columns_to_keep, given in the experiments_dict.

    :param experiments_dict: dict. Contains the information necessary for the filtering and the paths.
    :param compounds: dict. Contains dictionary of compounds in experiments. Required for any calibration.
    :param save_processed_data: Boolean. Tells you whether to save data in the experiments into the
    subdirectories of the treated_data directory.
    """

    experiment_labels = [*experiments_dict]

    for experiment in experiment_labels:
        filtered_file_name = experiments_dict[experiment]['paths']['filtered_data']
        df_imported = import_treated_csv_data(file_name=filtered_file_name,
                                              experiment_label=experiment)

        processing_functions = experiments_dict[experiment]['data_treatment']['processing_functions']
        df_processed = apply_processing_functions_in_experiment(df_unprocessed=df_imported,
                                                                experiments_dict=experiments_dict,
                                                                compounds=compounds,
                                                                experiment=experiment,
                                                                processing_function_strings=processing_functions)

        columns_to_keep = experiments_dict[experiment]['data_treatment']['columns_to_keep']
        df_processed_and_cleaned = df_processed[columns_to_keep]

        if save_processed_data:
            save_data_frame(df_to_save=df_processed_and_cleaned,
                            experiment_label=experiment,
                            level_of_treatment='PROCESSED')

    return


def apply_processing_functions_in_experiment(df_unprocessed, experiments_dict, compounds,
                                             experiment, processing_function_strings):
    """"""

    for function_string in processing_function_strings:

        if function_string == 'normalize':
            df_unprocessed = add_normalized_intensity_column(df=df_unprocessed,
                                                             internal_std_col='mz283')

        if function_string == 'calibrate_from_initial_conditions':
            calibration_params = experiments_dict[experiment]['data_treatment']['calibration_parameters']
            solution_weight_fractions = experiments_dict[experiment]['experimental_conditions']['solution_' +
                                                                                                'weight_fractions']
            df_unprocessed = add_calibrated_data_column(df=df_unprocessed,
                                                        calibration_params=calibration_params,
                                                        compounds=compounds,
                                                        initial_composition=solution_weight_fractions)

    return df_unprocessed


def add_calibrated_data_column(df, calibration_params, compounds, initial_composition):
    """"""

    df_with_calibrated_data = df

    calibration_query = calibration_params['calibration_query']
    df_calibrate = df.query(calibration_query)

    vars_to_calibrate = calibration_params['cols_to_calibrate']
    N_calibrations = len(vars_to_calibrate)
    for tick in range(N_calibrations):
        var = vars_to_calibrate[tick]
        analyte = calibration_params['analyte'][tick]
        method = calibration_params['method']

        if method == 'weight_fraction_to_molarity':
            signal_avg = np.mean(df_calibrate[var])
            molarity = calculate_molarity_from_weight_fraction(analyte=analyte,
                                                               compounds=compounds,
                                                               solution_comp=initial_composition)

            calibration_factor = molarity / signal_avg

        y_out_col = calibration_params['ys_out'][tick]
        df_with_calibrated_data[y_out_col] = calibration_factor * df_with_calibrated_data[var]

    return df_with_calibrated_data



def process_ms_data_in_evap_experiments(cleaned_ms_file_name, experiments_dict, save_processed_data=False):
    """
    """

    experiment_name = [*experiments_dict].pop()
    processing_params = experiments_dict[experiment_name]['processing']
    composition = experiments_dict[experiment_name]['experimental']['composition']
    df_cleaned = import_ms_data(file_name=cleaned_ms_file_name,
                                subdirectory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)


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
                                subdirectory=experiment_name)
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
                                subdirectory=experiment_name)
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
                                subdirectory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)
    df_processed['experiment'] = 'bd_nh3g_' + df_processed.nominal_nh3_mM.astype(str).str.replace('.','')

    # df_processed['mM_nhx']  add this to the droplet definitions soon

    if save_processed_data:
        save_data_frame(df_to_save=df_processed,
                        experiment_label=experiment_name,
                        level_of_cleaning='PROCESSED')

    return df_processed


def add_calibrated_ms_data_columns(df, processing_params, composition, analyte, internal_standard='PEG-6'):
    """"""

    df_calibrated = df.copy()
    ms_signal_inits = extract_calibration_data(df=df,
                                               t_init_cutoff=processing_params['cal_data_time'],
                                               cal_data_col=processing_params['y_col'])
    ms_signal_inits_avg, ms_signal_inits_rel_std = get_bootstrapped_statistics(ms_signal_inits)

    for compound_name, compound_mol_frac in composition.items():
        internal_standard_mol_frac = composition[internal_standard]

        if compound_name == analyte:
            rel_molar_abundance_in_solution = compound_mol_frac / internal_standard_mol_frac
            cal_factor_avg = rel_molar_abundance_in_solution / ms_signal_inits_avg
            cal_factor_std = ms_signal_inits_rel_std

            cal_data_col = processing_params['y_col'].replace('mz', 'mol')

            df_calibrated[cal_data_col] = df_calibrated[processing_params['y_col']] * cal_factor_avg
            df_calibrated[cal_data_col + '_std'] = df_calibrated[cal_data_col] * cal_factor_std

            decay_data_col = cal_data_col.split('/')[0] + '/' + cal_data_col.split('/')[0] + '_0'
            df_calibrated[decay_data_col] = df_calibrated[cal_data_col] / rel_molar_abundance_in_solution

    return df_calibrated


def add_normalized_intensity_column(df, internal_std_col='p283'):
    """Add "mz###_mz###" columns to DataFrame of normalized peak intensities.
    """

    p_cols = [col for col in df.columns if col[0] == 'p']  # first column of mass spec peaks is p

    for tick, p_col in enumerate(p_cols):
        df['mz' + p_col[1:] + '_mz' + internal_std_col[1:]] = df[p_col] / df[internal_std_col]

    return df
