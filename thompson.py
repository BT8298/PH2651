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
import statistics
import warnings
import itertools

from dataclasses import dataclass, field
from uncertainties import ufloat, unumpy  # linear error propagation theory
from lib import quadratic_regression

ureg = pint.UnitRegistry()
# CODATA electron charge to mass ratio
_cmr = const.physical_constants["electron charge to mass quotient"]
cmr_accepted_value = ufloat(_cmr[0], _cmr[2]) * pint.Unit(_cmr[1])


@dataclass
class ElectronBeamData:
    """Measurements for a deflected electron beam.

    Attributes:
        horizontal_beam_points:
            measured horizontal coordinates of points on the beam.
        vertical_beam_points:
            measured vertical coordinates of points on the beam.
    """

    horizontal_beam_points: numpy.ndarray[pint.Quantity[ufloat]]
    vertical_beam_points: numpy.ndarray[pint.Quantity[ufloat]]


@dataclass
class EFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by an electric field.

    Attributes:
        deflection_voltage:
            electric potential difference between two parallel metal plates
            above and below the flourescent screen.
    """

    deflection_voltage: pint.Quantity[ufloat]


@dataclass
class BFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by a magnetic field.

    Attributes:
        current:
            current as read from the power supply sent to the Helmholtz coils.
    """

    current: pint.Quantity[ufloat]


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
            TODO
        electric_field_trials:
        magnetic_field_trials:
    """

    accel_voltage: pint.Quantity[ufloat]
    coil_turns: int
    coil_radius: pint.Quantity[ufloat]
    coil_separation: pint.Quantity[ufloat]
    deflection_plate_separation: pint.Quantity[ufloat]
    electric_field_trials: list[EFieldOnly] = field(default_factory=list)
    magnetic_field_trials: list[BFieldOnly] = field(default_factory=list)
    cancellation_trials: list[EBCancellation] = field(default_factory=list)



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

# measurement protocol for grid: zoom on digital image, record at horizontal
# coordinates 2.5, 3.5, ..., 9.5. Estimate center of beam by eyeballing.
# Estimate vertical coordinate by eyeballing beam center's vertical position
# relative to vertical midline or edge of a grid square, whichever is closer.
# Then record to 0.01 cm precision. Uncertainty is +- quarter of a grid square
# length (I could be wrong).

horizontal_sample_coords = numpy.arange(2.5, 10.5, 1) * ureg.centimeter
# arc is one quarter circumference of a coil
arc = ufloat(4 + 3 / 4, 1 / 16) * ureg.inch
all_data = ChargeMassRatioMeasurements(
    accel_voltage=ufloat(1, 0.05) * ureg.kilovolt,
    # TODO input a measurement for this one?
    deflection_plate_separation=54 * ureg.millimeter,
    coil_turns=320,
    coil_radius=2 * arc / math.pi,
    coil_separation=ufloat(9.2, 0.1) * ureg.centimeter,
)
warnings.warn('remember to measure the deflection plates, see how well it agrees with manufacturer ~54 mm measurement')

# === Electric field only ===

# deflection voltage readings, uncertainty is the second argument (0.05)
E1v = unumpy.uarray([0.20, 0.41, 0.60, 0.82, 1.00], 0.05) * ureg.kilovolt
E2v = unumpy.uarray([0.20, 0.50, 0.60, 0.80, 1.00], 0.05) * ureg.kilovolt
E3v = unumpy.uarray([0.20, 0.42, 0.61, 0.81, 1.00], 0.05) * ureg.kilovolt

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
) * ureg.cm
E2y = unumpy.uarray(
    (
        [0.00, 0.00, 0.04, 0.09, 0.17, 0.22, 0.33, 0.40],
        [0.00, 0.05, 0.11, 0.20, 0.33, 0.46, 0.62, 0.84],
        [0.03, 0.10, 0.19, 0.35, 0.56, 0.78, 1.07, 1.37],
        [0.03, 0.14, 0.26, 0.49, 0.77, 1.04, 1.40, 1.79],
        [0.04, 0.18, 0.37, 0.61, 0.95, 1.35, 1.80, 2.28],
    ),
    0.05,
) * ureg.cm
E3y = unumpy.uarray(
    (
        [0.00, 0.00, 0.02, 0.05, 0.18, 0.20, 0.26, 0.39],
        [0.00, 0.02, 0.13, 0.21, 0.35, 0.50, 0.70, 0.90],
        [0.01, 0.07, 0.20, 0.38, 0.57, 0.79, 1.08, 1.39],
        [0.02, 0.15, 0.30, 0.50, 0.79, 1.10, 1.48, 1.90],
        [0.03, 0.18, 0.38, 0.60, 0.97, 1.31, 1.79, 2.23],
    ),
    0.05,
) * ureg.cm

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

# B trials current datums: 0.04, 0.08, ..., 0.20 with uncertainties
BI = unumpy.uarray(numpy.arange(0.04, 0.24, 0.04), 0.01)
warnings.warn('remember to put in the actual uncertainty of the current source (probably in the datasheet somewhere)')
warnings.warn('remember to calculate the current coming through a single coil; do the circuit laws')

B1 = unumpy.uarray(
    (
        [0.04, 0.09, 0.11, 0.16, 0.21, 0.26, 0.30, 0.38],
        [0.08, 0.16, 0.20, 0.28, 0.38, 0.49, 0.60, 0.71],
        [0.11, 0.20, 0.29, 0.41, 0.52, 0.69, 0.88, 1.08],
        [0.14, 0.26, 0.39, 0.53, 0.71, 0.91, 1.14, 1.42],
        [0.18, 0.30, 0.46, 0.66, 0.89, 1.15, 1.45, 1.79],
    ),
     0.05,
) * ureg.cm
B2 = unumpy.uarray((
        # runs 4 and 5
        [0.18, 0.23, 0.39, 0.55, 0.74, 0.95, 1.19, 1.22],
        [0.18, 0.30, 0.45, 0.64, 0.89, 1.16, 1.43, 1.80]
    ), 0.05) * ureg.centimeter
B3 = unumpy.uarray(
        (
            [0.02, 0.06, 0.11, 0.17, 0.20, 0.35, 0.33, 0.38],
            [0.04, 0.15, 0.20, 0.26, 0.38, 0.48, 0.59, 0.72],
            [0.12, 0.19, 0.28, 0.40, 0.56, 0.70, 0.83, 1.03],
            [0.18, 0.23, 0.39, 0.55, 0.71, 0.92, 1.17, 1.43],
            [0.19, 0.32, 0.46, 0.65, 0.90, 1.17, 1.43, 1.81],
            ), 0.05
        ) * ureg.centimeter

# populate the dataclass with the magnetic field data
for run in itertools.product((BI,), (B1, B2, B3)):
    for I, y_values in zip(run[0], run[1]):
        all_data.magnetic_field_trials.append(
                BFieldOnly(
                    current=I,
                    horizontal_beam_points=horizontal_sample_coords,
                    vertical_beam_points=y_values,
                    )
                )

# === Null deflection ===
# Cancelled due to power supply safety concerns.

# y-values of electron beam deflected only by E field, before B field applied
#EB1 = unumpy.uarray()
#EB2 = unumpy.uarray()
#EB3 = unumpy.uarray()

# currents used to reach null deflection
#EB1I = unumpy.uarray()
#EB2I = unumpy.uarray()
#EB3I = unumpy.uarray()

# === Analysis ===

regressions = []
B_cmr_results = []
EB_cmr_results = []
# equations (12) and (14) from lab guide
C, N, I, R, V_ac, mu_0 = sympy.symbols("C N I R V_ac mu_0")
B_cmr_expression = 125 / 8 * V_ac * (R * C / mu_0 / N / I) ** 2
EB_cmr_expression = 125 / 128 / V_ac * (R * 4 * C * V_ac / mu_0 / N / I) ** 2

for i, _ in enumerate(all_data.electric_field_trials):
    fit = quadratic_regression(
            # need to take the magnitude as numpy and statmodels do not play well with pint units
        [
            y.magnitude
            for y in all_data.electric_field_trials[i].vertical_beam_points
        ],
        [ x.magnitude for x in all_data.electric_field_trials[i].horizontal_beam_points ],
    ).fit()
    regressions.append(fit)

for i, _ in enumerate(all_data.magnetic_field_trials):
    fit = quadratic_regression(
        [
            y.nominal_value
            for y in all_data.magnetic_field_trials[i].vertical_beam_points
        ],
        all_data.magnetic_field_trials[i].horizontal_beam_points,
    ).fit()
    calculated_cmr = B_cmr_expression.subs(
        {
            "mu_0": scipy.constants.mu_0,
            "N": all_data.coil_turns,
            "I": all_data.magnetic_field_trials[i].current,
            "C": fit.params[0],
            "R": all_data.coil_radius,
            "V_ac": all_data.accel_voltage,
        }
    )
    B_cmr_results.append(calculated_cmr)

#for i, _ in enumerate(all_data.cancellation_trials):
#    fit = quadratic_regression(
#        [
#            y.nominal_value
#            for y in all_data.magnetic_field_trials[i].vertical_beam_points
#        ],
#        all_data.magnetic_field_trials[i].horizontal_beam_points,
#    ).fit()
#    calculated_cmr = EB_cmr_expression.subs(
#        {
#            "mu_0": scipy.constants.mu_0,
#            "N": all_data.coil_turns,
#            "I": all_data.cancellation_trials[i].current,
#            "C": fit.params[0],
#            "R": all_data.coil_radius,
#            "V_ac": all_data.accel_voltage,
#        }
#    )
#    EB_cmr_results.append(calculated_cmr)

if __name__ == "__main__":
    # prelab
    mu_0, N, I, R, V, B = sympy.symbols("mu_0 N I R V B")
    sp1 = sympy.sqrt(
        2 * 2000 * const.physical_constants["electron charge to mass quotient"][0]
    )
    sp2 = sympy.sqrt(
        2 * 2500 * const.physical_constants["electron charge to mass quotient"][0]
    )
    spt = sympy.sqrt(3 * const.k * 300 / const.m_e)
    sph = sympy.sqrt(2 * (13.6 * 1.602176634e-19) / const.m_e)
    B1 = (const.m_e * sp1) / (const.e * 20e-2)
    I1 = ((B1 * R) / (const.mu_0 * N) * (5 / 4) ** (3 / 2)).subs({R: 6e-2, N: 320})

    PRELAB_RESULTS = f"""
    (1) electron charge to mass ratio is {cmr_accepted_value}.

    (2) 2000 Volts: {sp1} m/s
        2500 Volts: {sp2} m/s

    (3) {B1} T

    (4) Thermal electron: {spt} m/s
        Hydrogenic electron: {sph} m/s

    (5) {I1} A
    """

    #for reg in regressions:
    #    print(reg.rsquared_adj)
    #print(PRELAB_RESULTS)
    #print(
    #    f"""
    #    acceleration voltage: {all_data.accel_voltage}
    #    coil turns: {all_data.coil_turns}
    #    coil radius: {all_data.coil_radius}
    #    coil outer separation: {all_data.coil_separation}
    #    deflection plate separation: {all_data.deflection_plate_separation}
    #    """
    #)
    #print(
    #    f"""
    #    accepted e/m value: {cmr_accepted_value}
    #    e/m from B field only: {statistics.mean(B_cmr_results)} (from {B_cmr_results})
    #    e/m from null force: {statistics.mean(EB_cmr_results)} (from {EB_cmr_results})
    #    """
    #)

# check if the statsmodels regression agrees with scipy
# popt, pcov = scipy.optimize.curve_fit(
#    lambda x, a, b, c: a * x**2 + b * x + c,
#    measurements["E_1z"],
#    measurements["E_1y"],
#    p0=(0, 0, 0),
# )
# print(popt)
