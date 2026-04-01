# error propagation libraries
# import chaospy # via polynomial chaos expansion
from uncertainties import ufloat  # via linear error propagation theory
# import soerp3 # via second order error propagation theory
# import mcerp3 # via Monte-Carlo something

import numpy
import scipy.constants as const
import scipy.optimize
import sympy
from dataclasses import dataclass, field
from lib import quadratic_regression

# CODATA electron charge to mass ratio
cmr = abs(const.physical_constants["electron charge to mass quotient"][0])


@dataclass
class ElectronBeamData:
    """Measurements for a deflected electron beam.

    Attributes:
        accel_voltage:
            electric potential difference through which electrons accelerate.
        horizontal_beam_points:
            measured horizontal coordinates of points on the beam.
        vertical_beam_points:
            measured vertical coordinates of points on the beam.
    """

    accel_voltage: float | ufloat
    horizontal_beam_points: list[float | ufloat]
    vertical_beam_points: list[float | ufloat]


@dataclass
class EFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by an electric field.

    Attributes:
        id:
            if several EFieldOnly's have the same id, their results are averaged
            together for a single figure to present.
        deflection_voltage:
            electric potential difference between two parallel metal plates
            above and below the flourescent screen.
    """

    id: int
    deflection_voltage: float | ufloat


@dataclass
class BFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by a magnetic field.

    Attributes:
        id:
            if several BFieldOnly's have the same id, their results are averaged
            together for a single figure to present.
        coil_turns:
            number of turns in the Helmholtz coils.
        coil_radius:
            radius of Helmholtz coils.
        coil_separation:
            TODO
        current:
            current as read from the power supply sent to the Helmholtz coils.
    """

    id: int
    coil_turns: int
    coil_radius: float | ufloat
    coil_separation: float | ufloat
    current: float | ufloat


# @dataclass(frozen=True)
# class EBCancellation:
#    accel_voltage: float|ufloat


@dataclass
class ChargeMassRatioMeasurements:
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

# deflection voltage readings (without uncertainties, that gets added later)
E1v = numpy.array([0.20, 0.41, 0.60, 0.82, 1.00])
E2v = numpy.array([0.20, 0.50, 0.60, 0.80, 1.00])

# each element is a list of measured y values (vertical) for a fixed beam shape
E1y = numpy.array(
    (
        [0.00, 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        [0.00, 0.04, 0.12, 0.21, 0.35, 0.50, 0.63, 0.90],
        [0.01, 0.05, 0.18, 0.34, 0.49, 0.82, 0.99, 1.23],
        [0.05, 0.15, 0.30, 0.51, 0.78, 1.12, 1.48, 1.90],
        [0.05, 0.17, 0.36, 0.59, 0.95, 1.31, 1.78, 2.23],
    )
)
E2y = numpy.array(
    (
        [0.00, 0.00, 0.04, 0.09, 0.17, 0.22, 0.33, 0.40],
        [0.00, 0.05, 0.11, 0.20, 0.33, 0.46, 0.62, 0.84],
        [0.03, 0.10, 0.19, 0.35, 0.56, 0.78, 1.07, 1.37],
        [0.03, 0.14, 0.26, 0.49, 0.77, 1.04, 1.40, 1.79],
        [0.04, 0.18, 0.37, 0.61, 0.95, 1.35, 1.80, 2.28],
    )
)

horizontal_sample_coords = numpy.arange(2.5, 10.5, 1)  # centimeters
grid_vertical_measurement_uncertainty = 0.05  # centimeters
all_data = ChargeMassRatioMeasurements()
all_data.electric_field_trials.append(
    EFieldOnly(
        id=1,
        accel_voltage=ufloat(1, 0.05),
        deflection_voltage=ufloat(0.2, 0.05),
        horizontal_beam_points=horizontal_sample_coords,
        vertical_beam_points=numpy.array(
            [
                ufloat(z, grid_vertical_measurement_uncertainty)
                for z in (0, 0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.3)
            ]
        ),
    )
)
all_data.electric_field_trials.append(
    EFieldOnly(
        id=2,
        accel_voltage=ufloat(1, 0.05),
        deflection_voltage=ufloat(0.41, 0.05),
        horizontal_beam_points=horizontal_sample_coords,
        vertical_beam_points=numpy.array(
            [
                ufloat(z, grid_vertical_measurement_uncertainty)
                for z in (0, 0.04, 0.12, 0.21, 0.35, 0.5, 0.63, 0.90)
            ]
        ),
    )
)
all_data.electric_field_trials.append(
    EFieldOnly(
        id=3,
        accel_voltage=ufloat(1, 0.05),
        deflection_voltage=ufloat(0.60, 0.05),
        horizontal_beam_points=horizontal_sample_coords,
        vertical_beam_points=numpy.array(
            [
                ufloat(z, grid_vertical_measurement_uncertainty)
                for z in (0.01, 0.05, 0.18, 0.34, 0.49, 0.82, 0.99, 1.23)
            ]
        ),
    )
)
all_data.electric_field_trials.append(
    EFieldOnly(
        id=4,
        accel_voltage=ufloat(1, 0.05),
        deflection_voltage=ufloat(0.82, 0.05),
        horizontal_beam_points=horizontal_sample_coords,
        vertical_beam_points=numpy.array(
            [
                ufloat(z, grid_vertical_measurement_uncertainty)
                for z in (0.05, 0.15, 0.30, 0.51, 0.78, 1.12, 1.48, 1.90)
            ]
        ),
    )
)
all_data.electric_field_trials.append(
    EFieldOnly(
        id=5,
        accel_voltage=ufloat(1, 0.05),
        deflection_voltage=ufloat(1, 0.05),
        horizontal_beam_points=horizontal_sample_coords,
        vertical_beam_points=numpy.array(
            [
                ufloat(z, grid_vertical_measurement_uncertainty)
                for z in (0.05, 0.17, 0.36, 0.59, 0.95, 1.31, 1.78, 2.23)
            ]
        ),
    )
)

# measurement protocol for grid: zoom on digital image, estimate vertical midpoint in a grid square, then estimate 0.1 cms from the vertical midpoint. gives uncertainty of +- quarter a grid square length
# 2.5, 3.5, ..., 9.5
measurements = {
    # "arc": ufloat(
    #    4 + 3 / 4,
    # ),
    # should be sequence of yz points with uncertanties
    "E_1z": horizontal_sample_coords,
    "E_1y": numpy.array([]),
    "E_2z": horizontal_sample_coords,
    "E_2": numpy.array(()),
    "E_3z": horizontal_sample_coords,
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
    # "E_1": quadratic_regression(measurements["E_1y"], measurements["E_1z"]),
    "E_1": None,
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
    # print(results["E_1"].fit().params)

    breakpoint()

    # check if the statsmodels regression agrees with scipy
    # popt, pcov = scipy.optimize.curve_fit(
    #    lambda x, a, b, c: a * x**2 + b * x + c,
    #    measurements["E_1z"],
    #    measurements["E_1y"],
    #    p0=(0, 0, 0),
    # )
    # print(popt)
