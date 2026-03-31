from collections.abc import Sequence
from sympy import parse_expr, symbols, sqrt, Symbol, Expr
from statsmodels.regression.linear_model import OLS
import numpy


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


def quadratic_regression(endog, exog):
    """Linear regression for the model Ax^2+Bx+C.

    The model is linear in the parameters ABC.
    """
    return OLS(
        endog, numpy.column_stack([numpy.square(exog), exog, numpy.ones(len(exog))])
    )
