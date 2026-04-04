# error propagation libraries
# import chaospy # via polynomial chaos expansion
# import soerp3 # via second order error propagation theory
# import mcerp3 # via Monte-Carlo something

import math
import itertools

import matplotlib.pyplot as plt
import numpy
import scipy.constants as const
import scipy.stats
import statsmodels.stats.diagnostic
import statsmodels.stats.stattools

import lib

from uncertainties import ufloat, unumpy  # linear error propagation theory

# CODATA electron charge to mass ratio
cmr_accepted_value = const.physical_constants["electron charge to mass quotient"]

"""
width of front coil: 2.14 cm +- 0.02mm
outer coil separation: 9.08 cm +- 0.02mm
quarter circumference of front coil by tape measure: 4 3/4 in., +- 1/16 in.
deflection plate separation: d = 54 mm (from manual), 5.28 cm +- 0.05 cm (from measurement by grid)
digital multimeter: HP 3466A, uncertainty +- 0.055 kV at >700 V
"""

# measurement protocol for grid: zoom on digital image, record at horizontal
# coordinates 2.5, 3.5, ..., 9.5. Estimate center of beam by eyeballing.
# Estimate vertical coordinate by eyeballing beam center's vertical position
# relative to vertical midline or edge of a grid square, whichever is closer.
# Then record to 0.01 cm precision. Uncertainty is +- quarter of a grid square
# length (I could be wrong).

horizontal_sample_coords = numpy.arange(2.5, 10.5, 1) * 1e-2  # cm to m
# arc is one quarter circumference of a coil
# we measured it to be 4 and 3/4 inches, with 1/8 inch markings
arc = ufloat(4 + 3 / 4, 1 / 16) * 0.0254  # in. to m
all_data = lib.ChargeMassRatioMeasurements(
    accel_voltage=ufloat(1, 0.05) * 1e3,  # kV to V
    deflection_plate_separation=ufloat(5.28, 0.05) * 1e-2,  # cm to m
    coil_turns=320,
    coil_radius=2 * arc / math.pi,
    coil_separation=ufloat(9.2, 0.002) * 1e-2,  # cm to m
)

# === Electric field only ===

# deflection voltage readings, uncertainty is the second argument (0.05)
E1v = unumpy.uarray([0.20, 0.41, 0.60, 0.82, 1.00], 0.05) * 1e3  # kV to V
E2v = unumpy.uarray([0.20, 0.50, 0.60, 0.80, 1.00], 0.05) * 1e3
E3v = unumpy.uarray([0.20, 0.42, 0.61, 0.81, 1.00], 0.05) * 1e3

# each element is a list of measured y values (vertical) for a fixed beam shape
E1y = (
    unumpy.uarray(
        (
            [0.00, 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
            [0.00, 0.04, 0.12, 0.21, 0.35, 0.50, 0.63, 0.90],
            [0.01, 0.05, 0.18, 0.34, 0.49, 0.82, 0.99, 1.23],
            [0.05, 0.15, 0.30, 0.51, 0.78, 1.12, 1.48, 1.90],
            [0.05, 0.17, 0.36, 0.59, 0.95, 1.31, 1.78, 2.23],
        ),
        0.05,
    )
    # convert cm to m
    * 1e-2
)

E2y = (
    unumpy.uarray(
        (
            [0.00, 0.00, 0.04, 0.09, 0.17, 0.22, 0.33, 0.40],
            [0.00, 0.05, 0.11, 0.20, 0.33, 0.46, 0.62, 0.84],
            [0.03, 0.10, 0.19, 0.35, 0.56, 0.78, 1.07, 1.37],
            [0.03, 0.14, 0.26, 0.49, 0.77, 1.04, 1.40, 1.79],
            [0.04, 0.18, 0.37, 0.61, 0.95, 1.35, 1.80, 2.28],
        ),
        0.05,
    )
    # convert cm to m
    * 1e-2
)

E3y = (
    unumpy.uarray(
        (
            [0.00, 0.00, 0.02, 0.05, 0.18, 0.20, 0.26, 0.39],
            [0.00, 0.02, 0.13, 0.21, 0.35, 0.50, 0.70, 0.90],
            [0.01, 0.07, 0.20, 0.38, 0.57, 0.79, 1.08, 1.39],
            [0.02, 0.15, 0.30, 0.50, 0.79, 1.10, 1.48, 1.90],
            [0.03, 0.18, 0.38, 0.60, 0.97, 1.31, 1.79, 2.23],
        ),
        0.05,
    )
    # convert cm to m
    * 1e-2
)

# populate the dataclass with the electric field data
for run in ((E1v, E1y), (E2v, E2y), (E3v, E3y)):
    for v_ac, y_values in zip(run[0], run[1]):
        all_data.electric_field_trials.append(
            lib.EFieldOnly(
                deflection_voltage=v_ac,
                horizontal_beam_points=horizontal_sample_coords,
                vertical_beam_points=y_values,
            )
        )

# === Magnetic field only ===

# B trials current sampled at: 0.04, 0.08, ..., 0.20 amps. Instek GPS-1850D
# current/voltage supply (attached to Helmholtz coils) accuracy is "0.5% of rdg
# + 2 digits" which we interpret as 0.5% of the reading.
_sample_points = numpy.arange(0.04, 0.24, 0.04)  # amps
BI = unumpy.uarray(_sample_points, 0.005 * _sample_points)

# vertical deflection measurements for B field only, in cm
B1 = (
    unumpy.uarray(
        (
            [0.04, 0.09, 0.11, 0.16, 0.21, 0.26, 0.30, 0.38],
            [0.08, 0.16, 0.20, 0.28, 0.38, 0.49, 0.60, 0.71],
            [0.11, 0.20, 0.29, 0.41, 0.52, 0.69, 0.88, 1.08],
            [0.14, 0.26, 0.39, 0.53, 0.71, 0.91, 1.14, 1.42],
            [0.18, 0.30, 0.46, 0.66, 0.89, 1.15, 1.45, 1.79],
        ),
        0.05,
    )
    * 1e-2
)

B2 = (
    unumpy.uarray(
        (
            [0.04, 0.09, 0.10, 0.15, 0.20, 0.26, 0.31, 0.38],
            [0.08, 0.16, 0.21, 0.28, 0.37, 0.48, 0.59, 0.71],
            [0.10, 0.20, 0.29, 0.41, 0.56, 0.69, 0.86, 1.07],
            [0.17, 0.23, 0.39, 0.55, 0.74, 0.95, 1.19, 1.42],
            [0.19, 0.30, 0.45, 0.64, 0.89, 1.17, 1.43, 1.80],
        ),
        0.05,
    )
    * 1e-2
)

B3 = (
    unumpy.uarray(
        (
            [0.02, 0.06, 0.12, 0.17, 0.20, 0.25, 0.33, 0.38],
            [0.04, 0.15, 0.20, 0.26, 0.38, 0.45, 0.59, 0.73],
            [0.12, 0.19, 0.28, 0.40, 0.56, 0.70, 0.83, 1.03],
            [0.18, 0.23, 0.39, 0.55, 0.71, 0.92, 1.17, 1.43],
            [0.19, 0.32, 0.46, 0.65, 0.90, 1.17, 1.43, 1.81],
        ),
        0.05,
    )
    * 1e-2
)

# populate the dataclass with the magnetic field data
for run in itertools.product((BI,), (B1, B2, B3)):
    for I, y_values in zip(run[0], run[1]):
        all_data.magnetic_field_trials.append(
            lib.BFieldOnly(
                # we assume the coils are electrically identical. they are
                # connected in parallel so the current in one coil is half
                # the reading on the power supply.
                current=I / 2,
                horizontal_beam_points=horizontal_sample_coords,
                vertical_beam_points=y_values,
            )
        )

# === Null deflection ===
# Cancelled due to power supply safety concerns.

# === Analysis ===

E_regressions = []
B_regressions = []
B_cmr_results = []

for i, _ in enumerate(all_data.electric_field_trials):
    z_values = all_data.electric_field_trials[i].horizontal_beam_points
    y_values = [
        y.nominal_value for y in all_data.electric_field_trials[i].vertical_beam_points
    ]
    fit = lib.quadratic_regression(y_values, z_values).fit()
    E_regressions.append(fit)

for i, _ in enumerate(all_data.magnetic_field_trials):
    z_values = all_data.magnetic_field_trials[i].horizontal_beam_points
    y_values = [
        y.nominal_value for y in all_data.magnetic_field_trials[i].vertical_beam_points
    ]
    fit = lib.quadratic_regression(y_values, z_values).fit()
    B_regressions.append(fit)

    # equation (12) from lab guide
    # 125 / 8 * V_ac * (R * C / mu_0 / N / I) ** 2
    # uncertainties package auto-propagates uncertainties using linear approximation
    calculated_cmr = (
        125
        / 8
        * all_data.accel_voltage
        * (
            all_data.coil_radius
            * fit.params[0]
            / scipy.constants.mu_0
            / all_data.coil_turns
            / all_data.magnetic_field_trials[i].current
        )
        ** 2
    )
    B_cmr_results.append(calculated_cmr)

if __name__ == "__main__":
    # uncertainties package doesn't play nice with some third party packages
    B_cmr_results_no_uncertainty = [cmr.nominal_value for cmr in B_cmr_results]
    print("Charge to mass ratio results from B field only:")
    for i, cmr in enumerate(B_cmr_results):
        print(
            "trial:",
            i + 1,
            "e/m:",
            cmr,
            "ratio to accepted value:",
            cmr / cmr_accepted_value[0],
        )

    print(
        f"""
    Fixed measurements:
        Coil radius: {all_data.coil_radius}
        Coil turns: {all_data.coil_turns}
        Coil separation: {all_data.coil_separation}
        Deflection plate separation: {all_data.deflection_plate_separation}

    Descriptive statistics for calculated e/m ratios:
        number of observations: {len(B_cmr_results)}
        median: {numpy.median(B_cmr_results_no_uncertainty):e}
        mean: {numpy.mean(B_cmr_results_no_uncertainty):e}
        standard deviation: {numpy.std(B_cmr_results_no_uncertainty, ddof=1):e}
    
    Statistical tests for calculated e/m ratio set:
        Lilliefors normality test (p value): {statsmodels.stats.diagnostic.lilliefors(B_cmr_results_no_uncertainty)[1]}
        Jarque-Bera normality test (p value): {statsmodels.stats.stattools.jarque_bera(B_cmr_results_no_uncertainty)[1]}
        Shapiro-Wilk normality test (p value): {scipy.stats.shapiro(B_cmr_results_no_uncertainty)[1]}

    Statistics for regression models:
    """
        # pprint.pprint(scipy.stats.describe([cmr.nominal_value for cmr in B_cmr_results]))
    )

    # plot of all regressions for beam shapes under B field only
    axs = plt.axes()
    _fit_plot_sample_pts = numpy.linspace(0.025, 0.095, 20)
    for run, fit in zip(all_data.magnetic_field_trials, B_regressions):
        z_values = run.horizontal_beam_points
        y_values = run.vertical_beam_points
        axs.set_title(
            f"Electron Trajectory in a Paired Helmholtz Coil Magnetic Fields with Quadratic Fit ({len(all_data.magnetic_field_trials)} trials)",
            loc="center",
            wrap=True,
        )
        axs.set_xlabel("z (cm)")
        axs.set_ylabel("y (cm)")
        axs.grid(visible=True)
        axs.set_xticks(z_values)
        plt.grid(visible=True)
        axs.errorbar(
            z_values,
            [y.nominal_value for y in y_values],
            yerr=[y.std_dev for y in y_values],
            elinewidth=1,
            capsize=2,
            linestyle="none",
            marker="o",
            markersize=3,
        )
        axs.plot(
            _fit_plot_sample_pts,
            fit.params[0] * _fit_plot_sample_pts**2
            + fit.params[1] * _fit_plot_sample_pts
            + fit.params[2],
            "k--",
            linewidth=1,
        )
        # plt.savefig(f"B_run{i}.svg")
        # plt.show(block=False)
        i = i + 1
    plt.savefig("B_all.svg")

    # i = 1
    # axs = plt.axes()
    # _fit_plot_sample_pts = numpy.linspace(0.025,0.095,20)
    # for y_values, fit in zip(all_data.magnetic_field_trials, B_regressions):
    #    #if i in {6,7,8}:
    #    #axs = plt.axes()
    #    #plt.title(f"B field run {i}")
    #    plt.title("B field all runs")
    #    plt.grid(visible=True)
    #    axs.scatter(y_values.horizontal_beam_points.magnitude, [y.magnitude.nominal_value for y in y_values.vertical_beam_points])
    #    #_pts = numpy.linspace(0.025,0.095,20)
    #    axs.plot(_fit_plot_sample_pts, fit.params[0]*_fit_plot_sample_pts**2 + fit.params[1]*_fit_plot_sample_pts + fit.params[2], 'k--')
    #    #plt.savefig(f"B_run{i}.svg")
    #    #plt.show(block=False)
    #    i=i+1
    # plt.show()

    # prelab
    # mu_0, N, I, R, V, B = sympy.symbols("mu_0 N I R V B")
    # sp1 = sympy.sqrt(
    #    2 * 2000 * const.physical_constants["electron charge to mass quotient"][0]
    # )
    # sp2 = sympy.sqrt(
    #    2 * 2500 * const.physical_constants["electron charge to mass quotient"][0]
    # )
    # spt = sympy.sqrt(3 * const.k * 300 / const.m_e)
    # sph = sympy.sqrt(2 * (13.6 * 1.602176634e-19) / const.m_e)
    # B1 = (const.m_e * sp1) / (const.e * 20e-2)
    # I1 = ((B1 * R) / (const.mu_0 * N) * (5 / 4) ** (3 / 2)).subs({R: 6e-2, N: 320})
