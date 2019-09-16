import numpy as np
from scipy.constants import pi

from src.d00_utils.calc_utils import (convert_moles_to_volume, convert_volume_to_radius,
                                      convert_molar_abundances_to_mole_fractions, calculate_vp_from_reference)


def gas_particle_partition(t, n_cmpds, partition_dict):
    """ Construct differential equations to describe change in n.

    Parameters:
    -----------
    t : float
    time (set to zero?).
    n_cmpds : ndarray
    1D-array of moles of compounds.
    partition_dict : dict
    dictionary of partitioning params (from pack_partition_dict) containing c_infs,
    D_gs, vps, T, compounds, water, x_water.

    Outputs
    -------
    dn : ndarray
    1D-array of dn for all compounds (excluding water).
    """

    def unpack(c_infs, D_gs, vps, T, compounds, water, x_water):
        return c_infs, D_gs, vps, T, compounds, water, x_water

    c_infs, D_gs, vps, T, compounds, water, x_water = unpack(**partition_dict)

    try:
        x_water, compounds, water, vps, T, D_gs, c_infs
    except NameError:
        print('evaporate_params dictionary is missing variables')

    n_water = x_water * n_cmpds.sum() / (1 - x_water)

    # add water back to compounds and ns for radius calculation
    ns = np.append(n_cmpds, n_water)
    cmpds = {**compounds, **{'water': water}}

    V = convert_moles_to_volume(compounds=cmpds,
                                ns=ns)
    r = convert_volume_to_radius(V=V)
    xs = convert_molar_abundances_to_mole_fractions(composition=n_cmpds,
                                                    x_water=x_water)
    c_sats = xs * vps / T  # assume ideal mixing to calculate saturation concentration at surface

    # maxwellian flux
    dns = 4 * pi * r * (D_gs * (c_infs - c_sats))

    return dns


def pack_partition_dict(compounds, water, T, x_water=0):
    """ Packs relevant partitioning parameters into a dictionary.

    Parameters
    ----------
    compounds : dict(dict)
    dictionary of compounds and their definitions.
    water : dict
    dictionary of water definitions.
    T : float
    temperature, K.
    x_water : float
    water mole fraction.

    Returns
    -------
    partition_dict : dict
    dictionary of parameters required for the forward run of gas particle partitioning flux function.
    """

    c_infs = np.array([defs['c_inf'] for name, defs in compounds.items()])
    D_gs = np.array([defs['D_g'] for name, defs in compounds.items()])

    vp_refs = np.array([defs['vp'] for name, defs in compounds.items()])
    dHs = np.array([defs['dH'] for name, defs in compounds.items()])
    T_refs = np.array([defs['T_vp'] for name, defs in compounds.items()])

    vps = np.empty(len(vp_refs))
    for tick in range(len(vps)):
        vps[tick] = calculate_vp_from_reference(vp_ref=vp_refs[tick],
                                                dH=dHs[tick],
                                                T_ref=T_refs[tick],
                                                T_desired=T)

    partition_dict = {'c_infs': c_infs, 'D_gs': D_gs, 'vps': vps,
                      'T': T, 'compounds': compounds, 'water': water, 'x_water': x_water}

    return partition_dict
