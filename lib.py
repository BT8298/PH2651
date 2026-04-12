import csv
import datetime
import math

import numpy
import uncertainties.core

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import NoneType
from typing import TextIO
from numpy.typing import ArrayLike
from numpy import ndarray
from statsmodels.regression.linear_model import OLS
from sympy import parse_expr, symbols, sqrt, Symbol, Expr
from uncertainties import ufloat


@dataclass
class ElectronBeamData:
    """Measurements for a deflected electron beam.

    Attributes:
        horizontal_beam_points:
            measured horizontal coordinates of points on the beam.
        vertical_beam_points:
            measured vertical coordinates of points on the beam.
    """

    horizontal_beam_points: ndarray
    vertical_beam_points: ndarray


@dataclass
class EFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by an electric field.

    Attributes:
        deflection_voltage:
            electric potential difference between two parallel metal plates
            above and below the flourescent screen.
    """

    deflection_voltage: uncertainties.core.Variable


@dataclass
class BFieldOnly(ElectronBeamData):
    """Measurements for an electron beam deflected only by a magnetic field.

    Attributes:
        current:
            current as read from the power supply sent to the Helmholtz coils.
    """

    current: uncertainties.core.Variable


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

    accel_voltage: uncertainties.core.Variable
    coil_turns: int
    coil_radius: uncertainties.core.Variable
    coil_separation: uncertainties.core.Variable
    deflection_plate_separation: uncertainties.core.Variable
    electric_field_trials: list[EFieldOnly] = field(default_factory=list)
    magnetic_field_trials: list[BFieldOnly] = field(default_factory=list)
    cancellation_trials: list[EBCancellation] = field(default_factory=list)


@dataclass(frozen=True)
class CharacteristicGamma:
    """A characteristic gamma ray emission of an isotope.

    Attributes:
        energy:
            Energy of the gamma, in electron volts.
        intensity:
            Intensity (yield) of the gamma as a decimal (95% = 0.95).
    """

    # some figures are not reported with uncertainty
    energy: float | uncertainties.core.Variable
    intensity: float | uncertainties.core.Variable


@dataclass(frozen=True)
class CheckSource:
    """A check source we used in the calibration procedure and to sample detector efficiency at points.

    Attributes:
        element:
            The element from the periodic table. Specified as the official
            symbol as seen on the periodic table.
        mass_number:
            Number of protons and neutrons in the nucleus.
        half_life:
            Half-life of the isotope, in years.
        assay_date:
            Date and time at which the source was procured.
        record_date:
            Date and time at which we analyzed photon this source's photon
            counts.
        initial_activity:
            The initial activity level printed on the check source.
        characteristic_gammas:
            *Selected* gamma ray emission energies (in keV) used in calibration and
            detector efficiency calculation.
    """

    element: str  # should be periodic table symbol e.g. "Cs"
    mass_number: int  # protons+neutrons
    half_life: uncertainties.core.Variable
    # uncertainties package does not support uncertain datetime objects
    assay_date: datetime.datetime
    record_date: datetime.datetime
    initial_activity: float
    characteristic_gammas: frozenset[CharacteristicGamma]


@dataclass
class PulseHeightAnalysis:
    """Imported from CSV export of ProSpect PHA.

    Attributes:
        start_time:
            Time at which the data collection was started.
        live_time:
            Total time elapsed while the detector was online (i.e. not cooling
            down after a detection).
        real_time:
            Time elapsed on a wall clock.
        energy_calibration:
            Parameters for energy calibration curve; offset, slope, quadratic.
        channels:
            Histogram bins.
        energies:
            Sampled photon energies.
        counts:
            Number of photon detections at a specific energy.
        check_source:
            The check source used for the analysis. If unknown, it is None.
    """

    start_time: datetime.datetime
    live_time: uncertainties.core.Variable
    real_time: float
    energy_calibration: tuple[
        float, float, float
    ]  # TODO offset, slope, quadratic
    channels: ArrayLike[int]
    # TODO uncertainty in energy?
    energies: ArrayLike[float]
    counts: ArrayLike[int]
    check_source: CheckSource | None = None


@dataclass
class MysteryIsotopeMeasurements:
    """Container for all measurements in the mystery isotope experiment.

    Attributes:
        coarse_gain:
            coarse gain detector setting in ProSpect.
        fine_gain:
            fine gain detector setting in ProSpect.
        detector_voltage:
            potential difference configured in ProSpect.
        check_sources:
            the check sources we used to calibrate the energy scale and sample
            points of a detector efficiency curve.
    """

    coarse_gain: float
    fine_gain: float
    detector_voltage: uncertainties.core.Variable
    # check_sources: list[CheckSource] = field(default_factory=list)
    pulse_height_analyses: list[PulseHeightAnalysis] = field(
        default_factory=list
    )


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
    return sqrt(
        sum([f.diff(sym) ** 2 * s_sym**2 for sym, s_sym in zip(syms, s_syms)])
    )


def quadratic_regression(endog, exog):
    """Linear regression for the model Ax^2+Bx+C.

    The model is linear in the parameters ABC.
    """
    return OLS(
        endog,
        numpy.column_stack([numpy.square(exog), exog, numpy.ones(len(exog))]),
    )


def calculate_radioactivity(
    a_0: float | uncertainties.core.Variable,
    t_half: float | uncertainties.core.Variable,
    t: float | uncertainties.core.Variable,
):
    """Calculate the radioactivity of an isotope using the decay law.

    Args:
        a_0:
            Initial radioactivity.
        t_half:
            Half-life.
        t:
            Time elapsed in years.
    """
    return a_0*math.exp(-(math.ln(2)/t_half)*t)


def parse_prospect_csv(
    file: TextIO, check_source: CheckSource | NoneType = None
) -> PulseHeightAnalysis:
    """

    Args:
        file:
            The pulse height analysis CSV file generated by ProSpect.
        check_source:
            Optional. The check source used in this pulse height analysis, if
            known.

    Returns:
        The data transcoded to a PulseHeightAnalysis instance.
    """
    # consume the header lines before invoking the csv reader
    header = []
    for line in file:
        if line.startswith("Spectrum"):
            break
        header.append(line.strip("\r\n"))

    # Fri Apr 10 12:04:11 GMT-0400 2026
    start_time = datetime.datetime.strptime(
        header[0].split(", ")[-1], "%a %b %d %H:%M:%S %Z%z %Y"
    )
    # two spaces after comma
    energy_calibration = []
    for i, coef in enumerate(header[1].split(',  ')[-1].split(', ')):
        energy_calibration.append(coef.split(', ')[-1])
    energy_calibration = tuple(energy_calibration)
    live_time = float(header[2].split(', ')[-1])
    real_time = float(header[3].split(', ')[-1])

    # 5% accuracy on live time correction per the datasheet
    live_time = ufloat(live_time, 0.05*live_time)

    # at this point, spectrum file object is seeked to the header line
    # "Channel, Energy (keV), Counts"
    reader = csv.DictReader(file, skipinitialspace=True)
    spectrum_data = {"channels": [], "energies": [], "counts": []}
    for rowdict in reader:
        spectrum_data["channels"].append(int(rowdict["Channel"]))
        spectrum_data["energies"].append(float(rowdict["Energy (keV)"]))
        spectrum_data["counts"].append(int(rowdict["Counts"]))
    # convert to arrays after, to take advantage of python's list type
    # being linked for quick appending. using numpy.append would do a
    # copy operation for each line in the csv file.
    spectrum_data["channels"] = numpy.array(spectrum_data["channels"])  # ty: ignore[invalid-assignment]
    spectrum_data["energies"] = numpy.array(spectrum_data["energies"])  # ty: ignore[invalid-assignment]
    spectrum_data["counts"] = numpy.array(spectrum_data["counts"])  # ty: ignore[invalid-assignment]

    return PulseHeightAnalysis(
        start_time=start_time,
        live_time=live_time,
        real_time=real_time,
        energy_calibration=energy_calibration,
        check_source=check_source,
        **spectrum_data,
    )
