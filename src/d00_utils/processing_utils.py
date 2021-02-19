import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.d00_utils.calc_utils import calculate_molarity_from_weight_fraction


def get_bootstrap_sample(dataset):
    bootstrap_sample = np.random.choice(dataset, size=len(dataset))

    return bootstrap_sample


def perform_bootstrap(dataset):
    n = 10000
    samples = np.empty(shape=(n, len(dataset)))

    for tick in range(n):
        samples[tick] = get_bootstrap_sample(dataset)

    return samples


def get_bootstrapped_statistics(data):
    """"""

    samples = perform_bootstrap(data)
    sample_means = np.mean(samples, axis=0)
    sample_avg = np.mean(sample_means)
    sample_std = np.std(sample_means)

    sample_rel_std = sample_std / sample_avg

    return sample_avg, sample_rel_std


def add_estimated_nh_column(df, initial_composition, nh_source, compounds):
    """ Estimates total ammonium (NH3 + NH4) from initial concentration and a 1:1 loss of butenedial. Adds
    column with estimated NH concentration.

    :param df: dataframe. Requires butenedial in a column.
    :param initial_composition: dict. Contains the starting solution, with compound, wt-frac as key, value pairs.
    :param nh_source: str. States the source of NH to the system. Used to ID molar ratio (e.g., ammonium sulfate -> 2).
    :param compounds: dict. Contains dictionary of compounds in experiments. Required for any calibration.

    :return df: dataframe. Dataframe now with column of total ammonium.
    """

    mw_nh = compounds[nh_source]['mw']
    nh_shorthand_name = compounds[nh_source]['name']
    mw_bd = compounds['butenedial']['mw']

    nh_init = initial_composition[nh_shorthand_name]
    bd_init = initial_composition['butenedial']  # assumes for now this is butenedial...

    if nh_shorthand_name[0:4] == 'NH42':
        N_nh = 2
    else:
        N_nh = 1

    n_nh_init = N_nh * nh_init / mw_nh
    n_bd_init = bd_init / mw_bd

    molar_ratio = n_nh_init / n_bd_init

    df['M_NH'] = molar_ratio * df['M_BUTENEDIAL']

    return df


def add_calibrated_data_column(df, calibration_params, compounds, initial_composition):
    """ Adds calibrated columns to dataframe, calibrated according to specified method, from uncalibrated signal data.

    :param df: dataframe. Requires butenedial in a column.
    :param calibration_params: list of str. ID the calibration procedures to use.
    :param initial_composition: dict. Contains the starting solution, with compound, wt-frac as key, value pairs.
    :param compounds: dict. Contains dictionary of compounds in experiments. Required for any calibration.

    :return df_with_calibrated_data: dataframe. Dataframe now with column of calibrated data.
    """

    df_with_calibrated_data = df

    if 'calibration_query' in calibration_params.keys():
        calibration_query = calibration_params['calibration_query']
        df_calibrate = df.query(calibration_query)

    else:
        df_calibrate = df.copy(deep=True)

    vars_to_calibrate = calibration_params['cols_to_calibrate']
    N_calibrations = len(vars_to_calibrate)
    for tick in range(N_calibrations):
        var = vars_to_calibrate[tick]
        method = calibration_params['method']

        # uses the initial conditions to calculate the molarity (MS)
        if method == 'weight_fraction_to_molarity':
            analyte = calibration_params['analyte'][tick]
            signal_avg = np.mean(df_calibrate[var])
            molarity = calculate_molarity_from_weight_fraction(analyte=analyte,
                                                               compounds=compounds,
                                                               solution_comp=initial_composition)

            calibration_factor = molarity / signal_avg

        # uses the unchanging internal standard itself to calculate molarity (NMR)
        if method == 'signal_to_molarity':
            internal_standard = calibration_params['internal_standard']
            internal_standard_shorthand = compounds[internal_standard]['name']
            internal_standard_wt_frac = initial_composition[internal_standard_shorthand]
            internal_standard_molecular_weight = compounds[internal_standard]['mw']
            internal_standard_proton_count = compounds[internal_standard]['protons_in_nmr_peak']

            calibration_factor = internal_standard_wt_frac * internal_standard_proton_count / \
                                 internal_standard_molecular_weight

        y_out_col = calibration_params['ys_out'][tick]
        df_with_calibrated_data[y_out_col] = calibration_factor * df_with_calibrated_data[var]

    return df_with_calibrated_data


def add_estimated_ph_column(df, indicator):
    """ Adds pH column to dataframe, according to chemical shift, in ppm, quantities in prescribed indicator column.

    :param df: dataframe. Requires butenedial in a column.
    :param indicator. str. Can either be acetic_acid, methyl_phosphonic_acid, or tri_methyl_phenol, and requires
    the PPM_INTERNAL_STD or PPM_ACETIC_ACID column to be in the dataframe.

    :return df_with_calibrated_data: dataframe. Dataframe now with column of estimated pH data
    """

    df_with_pH = df.copy(deep=True)

    if indicator == 'acetic_acid':  # tynkkynen et al., 2009; wallace, 2018
        shift_base = 1.900  # (wallace_shift_base: 1.906)
        shift_acid = 2.096  # wallace_shift_acid: 2.0830
        pka = 4.58  # wallace. 2018
        df_with_pH['f_acid'] = (shift_base - df['PPM_ACETIC_ACID']) / (shift_base - shift_acid)
        df_with_pH['f_base'] = 1 - df_with_pH['f_acid']

    elif indicator == 'methyl_phosphonic_acid':  # wallace, 2015 (error < 0.1)
        shift_base = 1.066  # wallace: shift_base: 1.0711
        shift_acid = 1.294  # seen in low ph scenarios (wallace: shift_acid = 1.2819)
        pka = 7.75

        df_with_pH['f_acid'] = (shift_base - df['PPM_INTERNAL_STD']) / (shift_base - shift_acid)
        df_with_pH['f_base'] = 1 - df_with_pH['f_acid']

    elif indicator == 'tri_methyl_phenol':
        shift_base = 6.355
        shift_acid = 6.615
        pka = 10.88
        df_with_pH['f_acid'] = (shift_base - df['PPM_INTERNAL_STD']) / (shift_base - shift_acid)
        df_with_pH['f_base'] = 1 - df_with_pH['f_acid']

    df_with_pH['pH'] = pka + np.log10(df_with_pH['f_base'] / df_with_pH['f_acid'])
    df_with_pH['H_PLUS'] = 10 ** (-df_with_pH['pH'])

    return df_with_pH


def add_normalized_intensity_column(df, internal_std_col='MZ283'):
    """Add "MZ###_MZ###" columns to DataFrame of normalized peak intensities.

    :param df: dataframe. Requires internal_std_col in a column.
    :param internal_std_col: str. The title of column with internal standard used for normalizing the data.

    :return df: dataframe. Dataframe now with normalized peak intensity data
    """

    mz_cols = [col for col in df.columns if col[:2] == 'MZ']  # first column of mass spec peaks is 'MZ'

    for tick, mz_col in enumerate(mz_cols):
        col_str = 'MZ' + mz_col[2:] + '_MZ' + internal_std_col[2:]
        df[col_str] = df[mz_col] / df[internal_std_col]

    return df


def add_estimated_nh3_column(df, rh):
    """Add gas-phase and droplet estimated ammonia concentrations based on concentration in bubbler and Henry's law.

    :param df: dataframe. Requires mM_AMMONIA_BUBBLER in a column.
    :param rh: float. The relative humidity in the EDB.

    :return df_with_nh3: dataframe. Dataframe now with PPM_AMMONIA and M_AMMONIA columns.
    """

    H = 62
    df_with_nh3 = df.copy(deep=True)
    df_with_nh3['PPM_AMMONIA'] = df_with_nh3['mM_AMMONIA_BUBBLER'] * rh * (1 / H) * 1e3
    df_with_nh3['M_AMMONIA'] = df_with_nh3['mM_AMMONIA_BUBBLER'] * 1e-3 * rh

    return df_with_nh3