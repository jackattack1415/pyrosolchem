import numpy as np
from scipy.constants import pi

from src.d00_utils.calc_utils import (calculate_partial_volumes_from_moles, calculate_radius_from_volume, \
    calculate_mole_fractions_from_molar_abundances)


def dn_gas_particle_partitioning(ns_cmpd, c_infs, vps, D_gs, T, compounds, water, x_water=0):
    """Construct differential equations to describe change in n.

    Parameters:
    -----------
    ns_cmpd : ndarray
    1D-array of moles of compounds.
    c_infs: ndarray
    1D-array of gas-phase background concentrations of compounds, mol m^-3.
    vps : ndarray
    1D-array of saturation vapor pressures at T, Pa.
    D_gs : ndarray
    1D-array of gas-phase diffusivities, m^2 s^-1.
    T : float
    temperature, K.
    compounds : dict(dict)
    dictionary of compounds and their definitions.
    water : dict
    dictionary of water definitions.
    x_water : float
    water mole fraction.

    Outputs
    -------
    dn : ndarray
    1D-array of dn for all compounds (excluding water).
    """

    n_water = x_water * ns_cmpd.sum() / (1 - x_water)

    # add water back to compounds and ns for radius calculation
    ns = np.append(ns_cmpd, n_water)
    cmpds = {**compounds, **{'water': water}}

    Vs = calculate_partial_volumes_from_moles(compounds=cmpds,
                                              ns=ns)
    V = Vs.sum()
    r = calculate_radius_from_volume(V)

    xs = calculate_mole_fractions_from_molar_abundances(ns=ns_cmpd,
                                                        x_water=x_water)
    c_sats = xs * vps / T  # assume ideal mixing to calculate saturation concentration at surface

    # maxwellian flux
    dns = 4 * pi * r * (D_gs * (c_infs - c_sats))

    return dns