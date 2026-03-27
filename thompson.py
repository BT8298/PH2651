import scipy.constants as const
from sympy import symbols, sqrt

# CODATA electron charge to mass ratio
cmr = abs(const.physical_constants["electron charge to mass quotient"][0])

# prelab
mu_0, N, I, R, V, B = symbols("mu_0 N I R V B")
sp1 = sqrt(2 * 2000 * cmr)
sp2 = sqrt(2 * 2500 * cmr)
spt = sqrt(3 * const.k * 300 / const.m_e)
sph = sqrt(2 * (13.6 * 1.602176634e-19) / const.m_e)
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
quarter circumference of front coil by tape measure: 2*pi*r/4 = 4 3/4 in.
number of turns for both: N = 320
coil current steps (amps): 0.00, 0.04, 0.08, 0.12, 0.16, 0.20
"""

# with magnetic field ONLY

cmr_experimental_expression = (2 * V) / (R * B) ** 2
#propagate()

# value_and_covariance(cmr_experimental_expression)

# with electric field ONLY

# with balanced electric and magnetic forces

if __name__ == "__main__":
    print(PRELAB_RESULTS)
#    x = Symbol('x', real=True)
#    prop = propagate(x**2, 'x')
