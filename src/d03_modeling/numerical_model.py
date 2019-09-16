import numpy as np
from scipy.integrate import ode


def differentiate(function, function_params_dict, n_inits, step_count, step=1):
    """"""

    output = np.empty((int(step_count), len(n_inits)))
    output[0, :] = n_inits

    r = ode(function)
    r.set_integrator('lsoda', with_jacobian=False,)
    r.set_initial_value(n_inits, t=0)
    r.set_f_params(function_params_dict)

    entry = 0
    while r.successful() and entry < step_count - 1:
        entry = int(round(r.t / step)) + 1
        next_step = r.integrate(r.t + step)
        output[entry, :] = next_step

    return output
