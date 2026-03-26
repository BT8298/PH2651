import scipy.constants as const
from collections.abc import Sequence
from sympy import parse_expr, pretty_print, symbols, sqrt, Symbol, Expr

def propagate(f: str | Expr, vars: str | Sequence[str]):
    """Calculate the expression for propagation of uncertainty.

    Assumes that all variables are independent.

    Args:
        f: sympy expression for the function to propagate the uncertainty
            through
        vars: sympy-style variables declaration, for example "x y z" or 
            ("x", "y", "z")
    
    Returns:
        The quadrature sum of terms of the form (∂f/∂x)² sₓ², a common
        approximation for the standard deviation of a function when the
        variables are independent.
    """
    if not isinstance(f, Expr):
        f = parse_expr(f)
    syms = symbols(vars)
    # if only one symbol is entered, syms is not iterable, so we make it
    if isinstance(syms, Symbol):
        syms = (syms,)
    s_syms = symbols(["s_" + var for var in vars], positive=True)
    return sqrt(sum([f.diff(sym) ** 2 * s_sym**2 for sym, s_sym in zip(syms, s_syms)]))

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
