from src.d00_utils.calc_utils import *
from src.d00_utils.flux_utils import (gas_particle_partition, pack_partition_dict)
from src.d02_processing.numerical_model import differentiate


def evaporate(compounds, water, params):
    """ unifying function for evaporation of compounds in a droplet, as described in params.

    :param compounds: (dict) dictionary of definitions of compounds in solution.
    :param water:  (dict) dictionary of definitions about water.
    :param params: (dict) dictionary of defintions about the system.
    :return: ns: (ndarray(floats)) 2D array of floats of moles by compound and time in simulation.
    :return: rs: (ndarray(floats)) 1D array of floats of radii in time in simulation.
    :return: ts: (ndarray(floats)) 1D array of floats of times in simulation.
    """

    x_cmpds = convert_molar_abundances_to_mole_fractions(composition=params['composition'],
                                                         x_water=params['x_water'])

    v_init = convert_radius_to_volume(r=params['r_init'])

    n_inits = convert_volume_to_moles(V=v_init,
                                      compounds=compounds,
                                      water=water,
                                      x_cmpds=x_cmpds,
                                      x_water=params['x_water'])

    gp_dict = pack_partition_dict(compounds=compounds,
                                  water=water,
                                  T=params['T'],
                                  x_water=params['x_water'])

    ns_dry = differentiate(function=gas_particle_partition,
                           function_params_dict=gp_dict,
                           n_inits=n_inits,
                           N_step=params['number_of_steps'],
                           step=params['step_size'])

    # inclusion of water is necessary to get the radius out
    wet_cmpds, ns_all = add_water_to_droplet(compounds=compounds,
                                             water=water,
                                             ns_dry=ns_dry,
                                             x_water=0)

    rs = np.empty(len(ns_all))
    for tick in range(len(ns_all)):
        V = convert_moles_to_volume(compounds=wet_cmpds,
                                    ns=ns_all[tick])
        rs[tick] = convert_volume_to_radius(V=V)

    ts = np.linspace(0, params['step_size']*params['number_of_steps'], params['number_of_steps']+1,
                     endpoint=True)

    return ns_dry, rs, ts


def add_water_to_droplet(compounds, water, ns_dry, x_water=0):
    """ Adds water to a dry droplet via changing compounds and moles.

    :param compounds: (dict) dictionary of definitions of compounds in solution.
    :param water:  (dict) dictionary of definitions about water.
    :param ns_dry: (ndarray(floats)) 2D array of floats of moles by compound (without water) and time in simulation.
    :param x_water: (float) water mole fraction in solution.
    :return: wet_compounds: (dict) dictionary of definitions of compounds and water.
    :return: ns_all: (ndarray(floats)) 2D array of floats of moles by compound (with water) and time in simulation.
    """

    n_waters = convert_water_mole_fraction_to_moles(ns_dry, x_water=x_water)

    ns_all = np.concatenate((ns_dry, n_waters.reshape(len(n_waters), 1)),
                            axis=1)  # add water ns to n_total

    wet_compounds = {**compounds, **{'water': water}}

    return wet_compounds, ns_all
