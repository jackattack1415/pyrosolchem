import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.d00_utils.data_utils import *
from src.d00_utils.processing_utils import *


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
    """ Performs processing functions, which are listed in the processing_function_strings. For loop goes through
    list of processing function strings and performs corresponding processing function. Nested within the
    process_data_for_experiments function.

    :param df_unprocessed: dataframe. Unprocessed input dataframe.
    :param experiments_dict: dict. Contains the information necessary for the filtering and the paths.
    :param compounds: dict. Contains dictionary of compounds in experiments. Required for any calibration.
    :param experiment: str. Experiment within experiments_dict from which to extract processing function inputs.
    :param processing_function_strings: list of str. Strings corresponding to function names in processing.
    subdirectories of the treated_data directory.

    :return df_processed: dataframe. Processed output dataframe.
    """

    for function_string in processing_function_strings:  # applies functions according to list from experiments yml

        solution_weight_fractions = experiments_dict[experiment]['experimental_conditions']['solution_' +
                                                                                            'weight_fractions']

        if function_string == 'normalize':
            df_unprocessed = add_normalized_intensity_column(df=df_unprocessed,
                                                             internal_std_col='MZ283')
        if function_string == 'calculate_pH':
            indicator = experiments_dict[experiment]['data_treatment']['pH_indicator']
            df_unprocessed = add_estimated_ph_column(df=df_unprocessed,
                                                     indicator=indicator)

        if function_string == 'calibrate_from_initial_conditions':
            calibration_params = experiments_dict[experiment]['data_treatment']['calibration_parameters']
            df_unprocessed = add_calibrated_data_column(df=df_unprocessed,
                                                        calibration_params=calibration_params,
                                                        compounds=compounds,
                                                        initial_composition=solution_weight_fractions)

        if function_string == 'calibrate_with_internal_standard':
            calibration_params = experiments_dict[experiment]['data_treatment']['calibration_parameters']
            df_unprocessed = add_calibrated_data_column(df=df_unprocessed,
                                                        calibration_params=calibration_params,
                                                        compounds=compounds,
                                                        initial_composition=solution_weight_fractions)

        if function_string == 'calculate_thca_nmer':
            df_unprocessed['M_THCA_NMER'] = df_unprocessed['M_THCA'] - df_unprocessed['M_THCA_MONOMER']

        if function_string == 'calculate_bd-pr-dz_nmer':
            df_unprocessed['M_NMER'] = df_unprocessed['M_POLYMER'] - df_unprocessed['M_DIMER']

        if function_string == 'calculate_nh':
            nh_source = experiments_dict[experiment]['data_treatment']['nh_source']
            df_unprocessed = add_estimated_nh_column(df=df_unprocessed,
                                                     initial_composition=solution_weight_fractions,
                                                     nh_source=nh_source,
                                                     compounds=compounds)

        if function_string == 'calculate_ammonia_from_bubbler':
            df_unprocessed = add_estimated_nh3_column(df=df_unprocessed,
                                                      rh=experiments_dict[experiment]['experimental_conditions'][
                                                          'x_water'])

        if function_string == 'merge_c4h5no_compounds':
            df_unprocessed['M_C4H5NO'] = df_unprocessed['M_BUTENIMINE'] + df_unprocessed['M_PYRROLINONE']

        if function_string == 'average_dimer':
            df_unprocessed['M_DIMER'] = (df_unprocessed['M_DIMER_1'] + df_unprocessed['M_DIMER_2']) / 2

    return df_unprocessed
