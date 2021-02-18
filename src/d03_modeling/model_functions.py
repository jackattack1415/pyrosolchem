import numpy as np
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit

from src.d00_utils.conf_utils import *
from src.d02_extraction.extract_least_sq_fit import *


def g(t, y0, coefs, f):
    """
    Solution to the ODE y'(t) = f(t,y,k) with initial condition y0
    """

    y = odeint(f, y0, t, args=(coefs,))

    return y


# SKELETON FOR BUILDING THE TOY MODEL
def differentials(y, t, coefs):
    """ Load the coefficients and build the differential equations. Coefs will be fit to the data. """

    # load all variables -- y[0], y[1]...
    X1 = y[0]
    X2 = y[1]

    # load all coefficients -- these are defined and params.add() outside of this fn.
    k = coefs['k'].value

    # create list of differential equations
    dX1dt = - k * X1
    dX2dt = - k * t

    return [dX1dt, dX2dt]


def residuals(coefs, t, data):
    """ Calculates the residuals of the function. """
    # load all values defined outside of this fn for each variable at t_start. y0 should match order in y[] above.
    y0 = [coefs['X1_0'].value, coefs['X2_0'].value]

    # calculates solution to odes, which will be compared against data in the return to get residuals.
    model = g(t, y0, coefs, differentials).T

    return (model - data).ravel()


def odes(ts, coefs):
    """ Recreates the odes above to produce the best fit solution. """
    def odes(y, t):
        X1, X2 = y
        # coefs numbering starts at # of initial conditions.
        dydt = [-coefs[2] * X1, -coefs[3] * X2]

        return dydt

    solution = odeint(odes, y0=coefs[0:2], t=ts).reshape(-1)

    return solution


# FIRST ORDER EQUATIONS #
def first_order(y, t, coefs):

    X = y[0]
    k = coefs['k'].value
    dXdt = - k * X

    return [dXdt]


def first_order_residuals(coefs, t, data):

    y0 = coefs['X_0'].value
    model = g(t, y0, coefs, first_order).T

    return (model - data).ravel()


def first_order_ode(ts, coefs):

    def odes(y, t):
        X = y
        dydt = -coefs[1] * X

        return dydt

    solution = odeint(odes, y0=coefs[0], t=ts).reshape(-1)

    return solution


# BDASPH9 FUNCTIONS FOR NMR FIT
def bdasph9_f(y, t, coefs):

    # initialize the odes
    ph = y[0]
    bd = y[1]
    pr = y[2]
    dm = y[3]
    i0 = coefs['i0'].value
    i1 = coefs['i1'].value
    i2 = coefs['i2'].value
    k6 = coefs['k6'].value
    k7 = coefs['k7'].value
    k8 = coefs['k8'].value
    k9 = coefs['k9'].value
    k10 = coefs['k10'].value
    bd0 = coefs['initial_butenedial'].value
    nh0 = coefs['initial_nhx'].value
    ai = coefs['ai'].value
    aii = coefs['aii'].value
    aiii = coefs['aiii'].value

    # calculate h_plus from empirical ph fit
    hp = 10 ** (-ph)
    nh = nh0 - (bd0 - bd)  # nh estimated from the 1:1 decay of butendial and total ammonium during reaction

    # acid-base equilibrium: nh3
    ka = 10 ** (-9.25)
    nh3 = nh * (1 + hp / ka) ** (-1)
    oh = 1e-14 / hp

    # disproportionation constant
    k1 = ((ai * oh + aii * oh * oh) / (1 + aiii * oh)) * 60

    # rate laws
    dphdt = -i0 * ph / ((i1 + t) * (i1 * i2 + i0 + i2 * t))
    dbddt = -k1 * bd - k6 * bd * nh3 - k7 * bd * pr * oh - k8 * bd * dm
    dprdt = k6 * bd * nh3 - k7 * bd * pr * oh - k9 * pr * dm
    ddmdt = k7 * bd * pr * oh - k10 * dm * dm

    return [dphdt, dbddt, dprdt, ddmdt]


def bdasph9_residuals(coefs, t, data):

    # calculate model prediction
    y0 = coefs['ph0'].value, coefs['bd0'].value, coefs['pr0'].value, coefs['dm0'].value
    model = g(t, y0, coefs, bdasph9_f).T

    # produce residuals between model and data, weighted with the averages of the data
    avgs = np.mean(data, axis=1)
    weight_factors = np.dstack([avgs] * data.shape[1])
    weight_factors = weight_factors.reshape(len(avgs), data.shape[1])

    weighted_residuals = (np.abs(model - data) / weight_factors).ravel()

    return weighted_residuals


def bdasph9_odes(ts, coefs):
    def odes(y, t):
        ph, bd, pr, dm = y

        i0 = coefs[4]
        i1 = coefs[5]
        i2 = coefs[6]
        k6 = coefs[7]
        k7 = coefs[8]
        k8 = coefs[9]
        k9 = coefs[10]
        k10 = coefs[11]
        bd0 = coefs[12]
        nh0 = coefs[13]
        ai = coefs[14]
        aii = coefs[15]
        aiii = coefs[16]

        # acid-base equilibrium
        hp = 10 ** (-ph)
        ka = 10 ** (-9.25)
        nh = max(0, nh0 - (bd0 - bd))
        nh3 = nh * (1 + hp / ka) ** (-1)
        oh = 1e-14 / hp

        # disproportionation
        k1 = ((ai * oh + aii * oh * oh) / (1 + aiii * oh)) * 60

        # rate laws
        dydt = [-i0 * ph / ((i1 + t) * (i1 * i2 + i0 + i2 * t)),
                -k1 * bd - k6 * bd * nh3 - k7 * bd * pr * oh - k8 * bd * dm,
                k6 * bd * nh3 - k7 * bd * pr * oh - k9 * pr * dm,
                k7 * bd * pr * oh - k10 * dm * dm]

        return dydt

    solution = odeint(odes, y0=coefs[0:4], t=ts)

    return solution


def bdahph9_odes(ts, coefs):
    def odes(y, t):
        ph, bd, pr, dm, ghc = y

        i0 = coefs[5]
        i1 = coefs[6]
        i2 = coefs[7]
        k6 = coefs[8]
        k7 = coefs[9]
        k8 = coefs[10]
        k9 = coefs[11]
        k10 = coefs[12]
        bd0 = coefs[13]
        nh0 = coefs[14]
        ai = coefs[15]
        aii = coefs[16]
        aiii = coefs[17]

        # acid-base equilibrium
        hp = 10 ** (-ph)
        ka = 10 ** (-9.25)
        nh = max(0, nh0 - (bd0 - bd))
        nh3 = nh * (1 + hp / ka) ** (-1)
        oh = 1e-14 / hp

        # disproportionation
        k1 = ((ai * oh + aii * oh * oh) / (1 + aiii * oh)) * 60

        # rate laws
        dydt = [-i0 * ph / ((i1 + t) * (i1 * i2 + i0 + i2 * t)),
                -k1 * bd - k6 * bd * nh3 - k7 * bd * pr * oh - k8 * bd * dm,
                k6 * bd * nh3 - k7 * bd * pr * oh - k9 * pr * dm,
                k7 * bd * pr * oh - k10 * dm * dm,
                k1 * bd]

        return dydt

    solution = odeint(odes, y0=coefs[0:5], t=ts)

    return solution


def bd10ag30_edb_ms(ts, coefs):
    def odes(y, t):
        bd, pr, dm = y

        # solution characteristics
        ph = coefs[4]
        hp = 10 ** (-ph)
        oh = 1e-14 / hp
        nh3 = coefs[3]

        # rate constants
        ke = coefs[5]
        k6 = coefs[6]
        k7 = coefs[7]
        k8 = coefs[8]
        k9 = coefs[9]
        k10 = coefs[10]
        ai = coefs[11]
        aii = coefs[12]
        aiii = coefs[13]
        k1 = ((ai * oh + aii * oh * oh) / (1 + aiii * oh)) * 60

        # rate laws
        dydt = [-k6 * bd * nh3 - k1 * bd - ke * bd - k7 * bd * pr * oh - k8 * bd * dm,
                k6 * bd * nh3 - k7 * bd * pr * oh - k9 * pr * dm,
                k7 * bd * pr * oh - k10 * dm * dm]

        return dydt

    solution = odeint(odes, y0=coefs[0:3], t=ts)

    return solution