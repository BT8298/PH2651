# error propagation libraries
# import chaospy # via polynomial chaos expansion
# from uncertainties import ufloat  # via linear error propagation theory
# import soerp3 # via second order error propagation theory
# import mcerp3 # via Monte-Carlo something

import numpy
import scipy.constants as const
import scipy.optimize
import sympy
from lib import quadratic_regression

# CODATA electron charge to mass ratio
cmr = abs(const.physical_constants["electron charge to mass quotient"][0])

# prelab
mu_0, N, I, R, V, B = sympy.symbols("mu_0 N I R V B")
sp1 = sympy.sqrt(2 * 2000 * cmr)
sp2 = sympy.sqrt(2 * 2500 * cmr)
spt = sympy.sqrt(3 * const.k * 300 / const.m_e)
sph = sympy.sqrt(2 * (13.6 * 1.602176634e-19) / const.m_e)
B1 = (const.m_e * sp1) / (const.e * 20e-2)
I1 = ((B1 * R) / (const.mu_0 * N) * (5 / 4) ** (3 / 2)).subs({R: 6e-2, N: 320})

PRELAB_RESULTS = f"""
(1) electron charge to mass ratio is {cmr}.

(2) 2000 Volts: {sp1} m/s
    2500 Volts: {sp2} m/s

(3) {B1} T

(4) Thermal electron: {spt} m/s
    Hydrogenic electron: {sph} m/s

(5) {I1} A
"""

# DATA
"""
width of front coil by caliper on black plastic: 8.25, 10, ~0.0
width of back coil by caliper on black plastic: 8.25, 10, ~0.0
outer coil separation: 9.2cm (uncertainty to 0.1 cm)
quarter circumference of front coil by tape measure: 2*pi*r/4 = 4 3/4 in., ~1/8 in. markings
number of turns for both: N = 320
deflection plate separation: d = 54 mm (from manual) X (from measurement by grid)
coil current steps (amps): 0.00, 0.04, 0.08, 0.12, 0.16, 0.20

vars:
    arc: quarter circumference of coil
    R: radius of coil (calculated)
    s: outer coil separation
"""

measurements = {
    # "arc": ufloat(
    #    4 + 3 / 4,
    # ),
    # should be sequence of yz points with uncertanties
    "E_1z": numpy.array((1, 2, 3, 4)),
    "E_1y": numpy.array((0.25, 0.5, 1.5, 4)),
    "E_2": numpy.array(()),
    "E_3": numpy.array(()),
    "B_1": numpy.array(()),
    "B_2": numpy.array(()),
    "B_3": numpy.array(()),
}

arc, R = sympy.symbols("arc R")
R = 2 * arc / sympy.pi

# with magnetic field ONLY

cmr_experimental_expression = (2 * V) / (R * B) ** 2
# propagate()

# value_and_covariance(cmr_experimental_expression)

# with electric field ONLY
# fit is Ax^2 + Bx + C, where ABC are the parameters so it is linear model in
# the parameters

results = {
    "E_1": quadratic_regression(measurements["E_1y"], measurements["E_1z"]),
    "E_2": None,
    "E_3": None,
    "B_1": None,
    "B_2": None,
    "B_3": None,
    "EB_1": None,
    "EB_2": None,
    "EB_3": None,
}


# with balanced electric and magnetic forces

if __name__ == "__main__":
    print(PRELAB_RESULTS)
    print(results["E_1"].fit().params)

    # check if the statsmodels regression agrees with scipy
    popt, pcov = scipy.optimize.curve_fit(
        lambda x, a, b, c: a * x**2 + b * x + c,
        measurements["E_1z"],
        measurements["E_1y"],
        p0=(0, 0, 0),
    )
    print(popt)
