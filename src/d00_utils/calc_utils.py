# code modified from: https://github.com/awbirdsall/pyvap

from __future__ import division
import numpy as np
from scipy.constants import pi, R, N_A


def calculate_volume_from_radius(radius):
    """ Calculate volume from radius.

    Parameters
    ----------
    radius : float
    radius of droplet in m^3.

    Returns
    -------
    volume : float
    volume of droplet in m^3.
    """

    volume = 4. / 3. * pi * radius ** 3

    return volume


def calculate_moles_from_volume(compounds, composition, v_total, x_h2o=None):
    """ Calculate volume from the molar composition of a list of compounds in solution.

    Parameters
    ----------
    components : list
    List of dicts for each component.

    ns : numpy.array
    2D numpy array of molar amounts of material. First index is entry number
    (e.g., each timestep), second index is index of component in `components`.

    x_h2o : float (optional, default None)
    Mole fraction of water added to particle in calculating value.

    Returns
    -------
    vtot : numpy.array
    1D array of total volumes calculated for each row in `ns`, with possible
    addition of water, in m^3.
    """

    if x_h2o:
        composition.append(x_h2o * composition.sum)
        compounds.append(water)

    rho_avg = np.average([compound['rho'] for compound in compounds],
                         weights=composition)  # kg m^-3

    m_total = v_total * rho_avg

    mw_avg = np.average([compound['M'] for compound in compounds],
                        weights=composition)  # kg mole^-1

    n_total = m_total / mw_avg  # mole

    composition_normalized = composition / composition.sum()
    n_compound = composition_normalized * n_total

    return n_compound





    moles = np.zeros_like(len(components))
    for moles, component in enumerate(components):
        # convert number of molecules in ns, using molecular/molar mass Ma/M
        # (units kg (molec or mol)^-1) and density rho (kg m^-3), to volume in
        # m^3
        v_component = n_components[:, moles] * component['M'] / component['rho'] / N_A
        v_total = v_total + v_component
    if x_h2o:
        n_h2o = x_h2o / (1 - x_h2o) * n_components.sum(axis=1)
        v_h2o = n_h2o * MA_H2O / RHO_H2O
        v_total = v_total + v_h2o
    return v_total


def calculate_volume_from_moles(components, n_components, constants, x_h2o=None):
    """ Calculate volume from the molar composition of a list of components in solution.

    Parameters
    ----------
    components : list
    List of dicts for each component.

    ns : numpy.array
    2D numpy array of molar amounts of material. First index is entry number
    (e.g., each timestep), second index is index of component in `components`.

    xh2o : float (optional, default None)
    Fixed mole fraction of water added to particle in calculating value.

    Returns
    -------
    vtot : numpy.array
    1D array of total volumes calculated for each row in `ns`, with possible
    addition of water, in m^3.
    """

    v_total = np.zeros_like(n_components.shape[0])
    for moles, component in enumerate(components):
        # convert number of molecules in ns, using molecular/molar mass Ma/M
        # (units kg (molec or mol)^-1) and density rho (kg m^-3), to volume in
        # m^3
        v_component = n_components[:, moles] * component['M'] / component['rho'] / N_A
        v_total = v_total + v_component
    if x_h2o:
        n_h2o = x_h2o / (1 - x_h2o) * n_components.sum(axis=1)
        v_h2o = n_h2o * MA_H2O / RHO_H2O
        v_total = v_total + v_h2o
    return v_total


def moles_to_radius(components, n_components, x_h2o=None):
    '''Given array of n values in time and list of components, calculate radii.

    Parameters
    ----------
    components : list
    List of dicts for each component.

    ns : numpy.array
    2D numpy array of molar amounts of material. First index is entry number
    (e.g., each timestep), second index is index of component in `components`.

    has_water : Boolean (optional, default False)
    Whether implicit water is added in addition to `components`.

    xh2o : float (optional, default None)
    Fixed mole fraction of water added to particle in calculating value. Only
    considered if has_water is True.

    Returns
    -------
    r : numpy.array
    Array of radii, in m, for each row of components given in `ns`.

    '''
    vtot = calcv(components, ns, has_water, xh2o)
    r = (3 * vtot / (4 * pi)) ** (1 / 3)
    return r


def convert_p0_enth_a_b(p0, del_enth, t0):
    '''Convert p0 and delta enthalpy to vapor pressure temp dependence params.

    Parameters
    ----------
    p0 : float or ndarray
    Vapor pressure at reference temperature, Pa.
    del_enth : float or ndarray
    Enthalpy of vaporization (or sublimation), J mol^-1.
    t0 : float or ndarray
    Reference temperature for p0 value, K.

    Returns
    -------
    p0_a, p0_b : float
    a (intercept, Pa) and b (slope, 1000/K) linear regression parameters for
    temperature dependence of log10(vapor pressure).

    '''
    p0_a = 1 / np.log(10) * ((del_enth / (R * t0)) + np.log(p0))
    p0_b = -del_enth / (1000 * np.log(10) * R)
    return p0_a, p0_b


def calcp0(a, b, temp):
    '''given regression line parameters, calculate vapor pressure at given
    temperature.'''
    log_p0 = a + b * (1000. / temp)
    p0 = pow(10, log_p0)
    return p0