# code modified from: https://github.com/awbirdsall/pyvap

from __future__ import division
import numpy as np
from scipy.constants import pi, R

from src.d00_utils.misc_utils import normalize


def calculate_mole_fractions_from_molar_abundances(composition, x_water=0):
    """ Calculates mole fractions of compounds in composition knowing the water mole fraction.

    Parameters
    ----------
    composition : dict

    x_water : float
    water mole fraction.

    Returns
    -------
    xs_cmpd : array
    array of mole fractions for compounds in composition.


    """

    if type(composition) is dict:
        composition = np.array(list(composition.values()))

    composition = normalize(composition)
    xs_cmpd = composition * (1 - x_water)

    return xs_cmpd


def calculate_volume_from_radius(r):
    """ Calculate volume from radius.

    Parameters
    ----------
    r : float
    radius of droplet in m.

    Returns
    -------
    V : float
    volume of droplet in m^3.
    """

    V = 4. / 3. * pi * r ** 3

    return V


def calculate_radius_from_volume(V):
    """ Calculate radius from volume.

    Parameters
    ----------
    V : float
    volume of droplet in m^3.

    Returns
    -------
    r : float
    radius of droplet in m.
    """

    r = (3 * V / (4 * pi)) ** (1/3)

    return r


def calculate_moles_from_volume(V_total, compounds, water, xs_cmpd, x_water=0):
    """ Calculate volume from the mole fractions of a list of compounds in solution.

    Parameters
    ----------
    V_total : float
    Volume of solution in m^3.

    compounds : dict(dict)
    Dict of dicts for each component.

    water : dict
    Dict of values describing water.

    xs_cmpd : numpy.array
    array of mole fractions of compounds in solution.

    x_water : float
    mole fraction of water in solution.

    Returns
    -------
    ns_cmpd : numpy.array
    1D array of moles of compounds according to composition and droplet size.
    """

    # add water to the compounds for the purposes of averaging within the droplet
    xs = np.append(xs_cmpd, x_water)
    cmpds = {**compounds, **{'water': water}}

    mw_avg = np.average([defs['mw'] for name, defs in cmpds.items()],
                        weights=xs)

    rho_avg = np.average([defs['rho'] for name, defs in cmpds.items()],
                         weights=xs)

    m_total = V_total * rho_avg
    n_total = m_total / mw_avg
    ns_cmpd = xs_cmpd * n_total

    return ns_cmpd


def calculate_volume_from_moles(compounds, ns):
    mw_avg = np.average([defs['mw'] for name, defs in compounds.items()],
                        weights=ns)

    rho_avg = np.average([defs['rho'] for name, defs in compounds.items()],
                         weights=ns)

    n_total = ns.sum()
    V = n_total * mw_avg / rho_avg

    return V


def calculate_vp_from_reference(vp_ref, dH, T_ref, T_desired):
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

    a = 1 / np.log(10) * ((dH / (R * T_ref)) + np.log(vp_ref))
    b = -dH / (1000 * np.log(10) * R)

    log_vp_desired = a + b * (1000. / T_desired)
    vp_desired = pow(10, log_vp_desired)

    return vp_desired
