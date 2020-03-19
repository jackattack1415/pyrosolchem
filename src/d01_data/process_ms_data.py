import pandas as pd

from src.d00_utils.processing_utils import get_bootstrapped_statistics
from src.d00_utils.data_utils import extract_calibration_data, save_data_frame, import_ms_data


def process_data_in_experiment(ms_file_name, experiments_dict, save_processed_data=False):
    """"""

    experiment_name = [*experiments_dict].pop()
    df_imported = import_ms_data(file_name=ms_file_name)

    df_filtered = filter_ms_data_in_experiment(df=df_imported,
                                               processing_parameters=experiments_dict[experiment_name]['processing'])

    df_cleaned = clean_ms_data_in_experiment(df=df_filtered,
                                             processing_parameters=experiments_dict[experiment_name]['processing'])

    if save_cleaned_data:
        save_data_frame(df_to_save=df_cleaned,
                        experiment_label=experiment_name,
                        level_of_cleaning='CLEANED')

    return df_cleaned


def process_ms_data_in_evap_experiments(cleaned_ms_file_name, experiments_dict, save_processed_data=False):
    """
    """

    experiment_name = [*experiments_dict].pop()
    processing_params = experiments_dict[experiment_name]['processing']
    composition = experiments_dict[experiment_name]['experimental']['composition']
    df_cleaned = import_ms_data(file_name=cleaned_ms_file_name,
                                subdirectory=experiment_name)
    df_processed = df_cleaned.copy(deep=True)
    df_processed = add_calibrated_ms_data_columns(df=df_processed,
                                                  processing_params=processing_params,
                                                  composition=composition,
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


def clean_ms_data_in_experiment(df_uncleaned, columns_to_keep):
    """ Removes columns from the dataframe based on 'columns_to_keep' list in the experiments dictionary from
    which processing parameters comes.

    :param df_uncleaned: df. Contains the unfiltered dataset.
    :param columns_to_keep: list. List of the columns to keep, from the experimental parameters.
    :return: df. Cleaned dataset.
    """


    df = add_normalized_intensity_column(df)  # make this more general eventually to non p283...

    df_cleaned = pd.DataFrame()

    experiment_cleaned = df[processing_parameters['columns_to_keep']]
    df_cleaned = df_cleaned.append(experiment_cleaned)

    df_cleaned = df_cleaned.rename(columns={'trapped': 'mins', 'comp': 'solution_name'})

    return df_cleaned


def add_normalized_intensity_column(df, internal_std_col='p283'):
    """Add "mz###_mz###" columns to DataFrame of normalized peak intensities.
    """

    p_cols = [col for col in df.columns if col[0] == 'p']  # first column of mass spec peaks is p

    for tick, p_col in enumerate(p_cols):
        df['mz' + p_col[1:] + '_mz' + internal_std_col[1:]] = df[p_col] / df[internal_std_col]

    return df
