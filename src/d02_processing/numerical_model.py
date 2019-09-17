import numpy as np
from scipy.integrate import ode


def differentiate(function, function_params_dict, n_inits, N_step, step=1):
    """ Returns n(t) at time step t using scipy's ODE package.

    :param function: (function) the dn/dt function to be integrated
    :param function_params_dict: (dict) dictionary of parameters that are inputs to above function.
    :param n_inits: (list(floats) list of initial ns by compound, i.e., n(0).
    :param N_steps: (float or int) number of steps on which to perform ode.
    :param step: (float or int) size of step, in seconds.
    :return: output: (ndarray) 2d array of ns by compound and by time step.
    """

    output = np.empty((int(N_step), len(n_inits)))
    output[0, :] = n_inits

    r = ode(function)
    r.set_integrator('lsoda', with_jacobian=False,)
    r.set_initial_value(n_inits, t=0)
    r.set_f_params(function_params_dict)

    entry = 0
    while r.successful() and entry < N_step - 1:
        entry = int(round(r.t / step)) + 1
        next_step = r.integrate(r.t + step)
        output[entry, :] = next_step

    return output
