# error propagation libraries
# import chaospy # via polynomial chaos expansion
# import soerp3 # via second order error propagation theory
# import mcerp3 # via Monte-Carlo something

import numpy
import scipy.constants as const
import scipy.optimize
import sympy
import math
import pint

from dataclasses import dataclass, field
from uncertainties import ufloat, unumpy  # linear error propagation theory
from sympy.solvers import solve
from sympy.core.relational import Equality
from lib import quadratic_regression

# CODATA electron charge to mass ratio
cmr = abs(const.physical_constants["electron charge to mass quotient"][0])
ureg = pint.UnitRegistry()


@dataclass
class ElectronBeamData:
    """Measurements for a deflected electron beam.

    Attributes:
        horizontal_beam_points:
            measured horizontal coordinates of points on the beam.
        vertical_beam_points:
            measured vertical coordinates of points on the beam.
    """

    horizontal_beam_points: (
        list[float | ufloat | pint.Quantity[float | ufloat]]
        | numpy.ndarray[pint.Quantity[ufloat]]
    )
    vertical_beam_points: (
        list[float | ufloat | pint.Quantity[float | ufloat]]
        | numpy.ndarray[pint.Quantity[ufloat]]
    )


@dataclass
class EFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by an electric field.

    Attributes:
        deflection_voltage:
            electric potential difference between two parallel metal plates
            above and below the flourescent screen.
    """

    deflection_voltage: float | ufloat | pint.Quantity[float | ufloat]
    id: int = 0


@dataclass
class BFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by a magnetic field.

    Attributes:
        current:
            current as read from the power supply sent to the Helmholtz coils.
    """

    current: float | ufloat | pint.Quantity[float | ufloat]


# @dataclass(frozen=True)
# class EBCancellation:
#    accel_voltage: float|ufloat


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
            TODO
        electric_field_trials:
        magnetic_field_trials:
    """

    accel_voltage: float | ufloat | pint.Quantity[float | ufloat]
    coil_turns: int
    coil_radius: float | ufloat | pint.Quantity[float | ufloat]
    coil_separation: float | ufloat | pint.Quantity[float | ufloat]
    deflection_plate_separation: float | ufloat | pint.Quantity[float | ufloat]
    electric_field_trials: list[EFieldOnly] = field(default_factory=list)
    magnetic_field_trials: list[BFieldOnly] = field(default_factory=list)
    # cancellation_trials: list[EBCancellation]


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

# measurement protocol for grid: zoom on digital image, estimate vertical midpoint in a grid square, then estimate 0.1 cms from the vertical midpoint. gives uncertainty of +- quarter a grid square length (1/4 of 2cm is 0.05 cm)
# the horizontal tickmarks we chose to record the beam height at: 2.5, 3.5, ..., 9.5 cm
horizontal_sample_coords = numpy.arange(2.5, 10.5, 1)
# arc is one quarter circumference of a coil
arc = ufloat(4 + 3 / 4, 1 / 16) * ureg.inch
all_data = ChargeMassRatioMeasurements(
    accel_voltage=ufloat(1, 0.05) * ureg.kilovolt,
    deflection_plate_separation=54 * ureg.millimeter,
    coil_turns=320,
    coil_radius=2 * arc / math.pi,
    coil_separation=ufloat(9.2, 0.1) * ureg.centimeter,
)

# === Electric field only ===

# deflection voltage readings (without uncertainties, that gets added later)
E1v = unumpy.uarray([0.20, 0.41, 0.60, 0.82, 1.00], 0.05)
E2v = unumpy.uarray([0.20, 0.50, 0.60, 0.80, 1.00], 0.05)
E3v = unumpy.uarray([0.20, 0.42, 0.61, 0.81, 1.00], 0.05)

# each element is a list of measured y values (vertical) for a fixed beam shape
E1y = unumpy.uarray(
    (
        [0.00, 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        [0.00, 0.04, 0.12, 0.21, 0.35, 0.50, 0.63, 0.90],
        [0.01, 0.05, 0.18, 0.34, 0.49, 0.82, 0.99, 1.23],
        [0.05, 0.15, 0.30, 0.51, 0.78, 1.12, 1.48, 1.90],
        [0.05, 0.17, 0.36, 0.59, 0.95, 1.31, 1.78, 2.23],
    ),
    0.05,
)
E2y = unumpy.uarray(
    (
        [0.00, 0.00, 0.04, 0.09, 0.17, 0.22, 0.33, 0.40],
        [0.00, 0.05, 0.11, 0.20, 0.33, 0.46, 0.62, 0.84],
        [0.03, 0.10, 0.19, 0.35, 0.56, 0.78, 1.07, 1.37],
        [0.03, 0.14, 0.26, 0.49, 0.77, 1.04, 1.40, 1.79],
        [0.04, 0.18, 0.37, 0.61, 0.95, 1.35, 1.80, 2.28],
    ),
    0.05,
)
E3y = unumpy.uarray(
    (
        [0.00, 0.00, 0.02, 0.05, 0.18, 0.20, 0.26, 0.39],
        [0.00, 0.02, 0.13, 0.21, 0.35, 0.50, 0.70, 0.90],
        [0.01, 0.07, 0.20, 0.38, 0.57, 0.79, 1.08, 1.39],
        [0.02, 0.15, 0.30, 0.50, 0.79, 1.10, 1.48, 1.90],
        [0.03, 0.18, 0.38, 0.60, 0.97, 1.31, 1.79, 2.23],
    ),
    0.05,
)

# populate the dataclass with the electric field data
for run in ((E1v, E1y), (E2v, E2y), (E3v, E3y)):
    for v_ac, y_values in zip(run[0], run[1]):
        all_data.electric_field_trials.append(
            EFieldOnly(
                deflection_voltage=v_ac,
                horizontal_beam_points=horizontal_sample_coords,
                vertical_beam_points=y_values,
            )
        )

# === Magnetic field only ===

cmr_experimental_expression = (2 * V) / (R * B) ** 2

# value_and_covariance(cmr_experimental_expression)

# with electric field ONLY
# fit is Ax^2 + Bx + C, where ABC are the parameters so it is linear model in
# the parameters

# with balanced electric and magnetic forces

# === Analysis ===

regressions = []
cmr, C, V_ac, mu_0, N, I, R = sympy.symbols("cmr C V_ac mu_0 N I R")
# equation (12) from lab guide
cmr_B_expression = 125 / 8 * V_ac * (R * C / mu_0 / N / I) ** 2
# equation (14) from lab guide
cmr_EB_expression = 125 / 128 /V_ac * (R * 4 * C * V_ac / mu_0 / N / I)**2

for i in range(len(all_data.electric_field_trials)):
    fit = quadratic_regression(
        [
            y.nominal_value
            for y in all_data.electric_field_trials[i].vertical_beam_points
        ],
        all_data.electric_field_trials[i].horizontal_beam_points,
    ).fit()
    regressions.append(fit)

for i in range(len(all_data.magnetic_field_trials)):
    fit = quadratic_regression(
        [
            y.nominal_value
            for y in all_data.magnetic_field_trials[i].vertical_beam_points
        ],
        all_data.magnetic_field_trials[i].horizontal_beam_points,
    ).fit()
    calculated_charge_mass_ratio = cmr_B_expression.subs(
        {
            "mu_0": scipy.constants.mu_0,
            "N": all_data.coil_turns,
            "I": all_data.magnetic_field_trials[i].current,
            "C": fit.params[0],
            "R": all_data.coil_radius,
            "V_ac": all_data.accel_voltage,
        }
    )
    EB_results.append(fit)


calculated_charge_mass_ratios = {
    "E": None,
    "B": None,
    "EB": None,
}

if __name__ == "__main__":
    # print(PRELAB_RESULTS)
    # print(results["E_1"].fit().params)

    print(EB_results)
    # breakpoint()

 #   print(all_data.coil_radius)
 #   for reg in regressions:
 #       print(reg.fit().rsquared)

    # check if the statsmodels regression agrees with scipy
    # popt, pcov = scipy.optimize.curve_fit(
    #    lambda x, a, b, c: a * x**2 + b * x + c,
    #    measurements["E_1z"],
    #    measurements["E_1y"],
    #    p0=(0, 0, 0),
    # )
    # print(popt)
