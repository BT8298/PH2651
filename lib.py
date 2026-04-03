from collections.abc import Sequence
from dataclasses import dataclass, field
from statsmodels.regression.linear_model import OLS
from sympy import parse_expr, symbols, sqrt, Symbol, Expr
from pint import Quantity
from uncertainties.core import Variable
from numpy import ndarray
import numpy

@dataclass
class ElectronBeamData:
    """Measurements for a deflected electron beam.

    Attributes:
        horizontal_beam_points:
            measured horizontal coordinates of points on the beam.
        vertical_beam_points:
            measured vertical coordinates of points on the beam.
    """

    horizontal_beam_points: ndarray[Quantity[Variable]]
    vertical_beam_points: ndarray[Quantity[Variable]]


@dataclass
class EFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by an electric field.

    Attributes:
        deflection_voltage:
            electric potential difference between two parallel metal plates
            above and below the flourescent screen.
    """

    deflection_voltage: Quantity[Variable]


@dataclass
class BFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by a magnetic field.

    Attributes:
        current:
            current as read from the power supply sent to the Helmholtz coils.
    """

    current: Quantity[Variable]


@dataclass
class EBCancellation(BFieldOnly, EFieldOnly):
    """Measurements for a null deflection electron beam.

    The electron beam data here is for when only the E field is active. The
    current is recorded when the B field cancels the beam's deflection.
    """
    pass


@dataclass
class ChargeMassRatioMeasurements:
    """Container for all measurements in this electron beam experiment.

    Parameters assumed to be fixed for the duration of the experiment belong to
    this class, such as the number of turns in a Helmholtz coil.

    Attributes:
        accel_voltage:
            potential difference through which electrons in the beam
            accelerate.
        coil_turns:
            number of turns in the Helmholtz coils.
        coil_radius:
            radius of Helmholtz coils.
        coil_separation:
            distance between the centers of the two coils.
        electric_field_trials:
        magnetic_field_trials:
    """

    accel_voltage: Quantity[Variable]
    coil_turns: int
    coil_radius: Quantity[Variable]
    coil_separation: Quantity[Variable]
    deflection_plate_separation: Quantity[Variable]
    electric_field_trials: list[EFieldOnly] = field(default_factory=list)
    magnetic_field_trials: list[BFieldOnly] = field(default_factory=list)
    cancellation_trials: list[EBCancellation] = field(default_factory=list)

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
