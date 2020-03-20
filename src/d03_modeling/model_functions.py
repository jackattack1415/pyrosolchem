from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit

from src.d02_extraction.extract_least_sq_fit import *


def g(t, y0, coefs, f):
    """
    Solution to the ODE y'(t) = f(t,y,k) with initial condition y0
    """

    y = odeint(f, y0, t, args=(coefs,))

    return y


# BD07AS03_NMR FUNCTIONS FOR LMFIT AND UNCERTAINTY ANALYSIS #
def f_bd07as03_nmr(y, t, coefs):

    BD = y[0]
    k = coefs['k'].value
    dBDdt = - k * BD * BD

    return [dBDdt]


def residuals_bd07as03_nmr(coefs, t, data):

    y0 = coefs['BD_0'].value
    model = g(t, y0, coefs, f_bd07as03_nmr).T

    return (model - data).ravel()


def odes_bd07as03_nmr(ts, coefs):

    def odes(y, t):
        bd = y
        dydt = -coefs[1] * bd * bd

        return dydt

    solution = odeint(odes, y0=coefs[0], t=ts).reshape(-1)

    return solution
# BD07AS03_NMR FUNCTIONS FOR LMFIT AND UNCERTAINTY ANALYSIS #