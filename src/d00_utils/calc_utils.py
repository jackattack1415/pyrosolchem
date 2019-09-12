# code modified from: https://github.com/awbirdsall/pyvap

from __future__ import division
import numpy as np
from scipy.constants import pi, R


def add_water_to_droplet_composition(composition, X_h2o=None):
    """ Updates dry droplet dict with water with X_h2o (typically specified in "compounds.yml").

    Parameters
    ----------
    composition : dict(float)
    dict of floats representing the molar composition of (dry) droplet

    X_h2o : float or None
    mole fraction of water in the droplet.

    """

    dry_composition_moles = list(composition.values())
    water_content = X_h2o * np.sum(dry_composition_moles)
    composition.update([('water', water_content)])


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

    r = float(r)
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


def calculate_moles_from_volume(compounds, composition, V_total):
    """ Calculate volume from the relative molar composition of a list of compounds in solution.

    Parameters
    ----------
    compounds : list
    List of dicts for each component.

    composition : numpy.array
    2D numpy array of molar amounts of material. First index is entry number
    (e.g., each timestep), second index is index of component in `components`.

    V_total : float
    Volume of solution in m^3.

    Returns
    -------
    ns_compound : numpy.array
    1D array of moles of compounds according to composition and droplet size.
    """

    mw_avg = np.average([compound['mw'] for compound in compounds],
                        weights=composition)  # kg mole^-1

    rho_avg = np.average([compound['rho'] for compound in compounds],
                         weights=composition)  # kg m^-3

    m_total = V_total * rho_avg  # kg
    n_total = m_total / mw_avg  # mole
    ns = composition * n_total

    return ns


def calculate_partial_volumes_from_moles(compounds, ns):

    mws = np.array([compound['mw'] for compound in compounds])
    rhos = np.array([compound['rho'] for compound in compounds])

    Vs = ns * mws / rhos  # array of partial volumes, m^3

    return Vs


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
