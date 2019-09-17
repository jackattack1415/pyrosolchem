from src.d00_utils.conf_utils import *
from src.d00_utils.calc_utils import *
from src.d00_utils.flux_utils import *
from src.d03_modeling.numerical_model import differentiate

cmpds, water = load_compounds()
params = load_parameters()

x_cmpds = convert_molar_abundances_to_mole_fractions(params['composition'],
                                                     x_water=params['x_h2o'])

v_init = convert_radius_to_volume(params['r_init'])

n_inits = convert_volume_to_moles(v_init,
                                  cmpds,
                                  water,
                                  x_cmpds,
                                  x_water=params['x_h2o'])

gp_dict = pack_partition_dict(cmpds,
                              water,
                              T=params['T'],
                              x_water=params['x_h2o'])

ns = differentiate(function=gas_particle_partition,
                   function_params_dict=gp_dict,
                   n_inits=n_inits,
                   step_count=100,
                   step=1)

n_waters = convert_water_mole_fraction_to_moles(ns, x_water=params['X_h2o'])
n_totals = np.concatenate((ns, n_waters.reshape(len(n_waters), 1)),
                          axis=1)  # add water ns to n_total

wet_cmpds = {**cmpds, **{'water': water}}

Vs = np.empty(len(ns))
rs = np.empty(len(ns))
for tick in range(len(n_totals)):
    Vs[tick] = convert_moles_to_volume(wet_cmpds, n_totals[tick])
    rs[tick] = convert_volume_to_radius(Vs[tick])
