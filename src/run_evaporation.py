from src.d00_utils.conf_utils import *
from src.d00_utils.calc_utils import *
from src.d03_modeling.evaporate import evaporate
from src.d05_reporting.plot_time_series import plot_composition_evolution


cmpds, water = load_compounds()
params = load_parameters()

ns_droplet, rs_droplet, Vs_droplet, ts_droplet = evaporate(cmpds, water, params)
Ms_droplet = convert_moles_to_molarity(n_cmpds=ns_droplet,
                                       V=Vs_droplet)

plot_composition_evolution(compounds=cmpds,
                           ts=ts_droplet,
                           ys=Ms_droplet,
                           rs=rs_droplet)