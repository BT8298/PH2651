import argparse
import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from uncertainties import ufloat

import lib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spectrum-files", nargs="+")
    parser.add_argument("-p", "--generate-plots", action="store_true")
    argns = parser.parse_args()

    # TODO detector uncertainty
    data = lib.MysteryIsotopeMeasurements(
        coarse_gain=1.0,
        fine_gain=2.0,
        # 3% voltage accuracy per the manual
        detector_voltage=ufloat(900, 0.03 * 900),
    )
    # TODO fill in dates and initial activities
    our_cesium_source = lib.CheckSource(
        element="Cs",
        mass_number=137,
        half_life=ufloat(30.007, 0.00046),
        assay_date=datetime.datetime(1995, 10, 13),
        record_date=datetime.datetime.today(),
        initial_activity=ufloat(1, 0.20 * 1),
        characteristic_gammas=(
            lib.CharacteristicGamma(
                # convert to eV
                energy=ufloat(661.657, 0.0003) * 1000,
                intensity=ufloat(0.8510, 0.0020),
            ),
        ),
    )
    our_barium_source = lib.CheckSource(
        element="Ba",
        mass_number=133,
        half_life=ufloat(10.536, 0.0008),
        assay_date=datetime.datetime(2013, 1, 3),
        record_date=datetime.datetime.today(),
        initial_activity=ufloat(0.988, 0.05 * 0.988),
        characteristic_gammas=(
            lib.CharacteristicGamma(
                energy=ufloat(275.925, 0.0007) * 1000,
                intensity=0.1769,
            ),
        ),
    )
    our_cobalt_source = lib.CheckSource(
        element="Co",
        mass_number=60,
        half_life=ufloat(5.271, 0.0002),
        assay_date=datetime.datetime(2013, 1, 1),
        record_date=datetime.datetime.today(),
        initial_activity=ufloat(0.49, 0.05 * 0.949),
        characteristic_gammas=(
            lib.CharacteristicGamma(
                energy=ufloat(1173.228, 0.0003) * 1000,
                intensity=ufloat(0.9985, 0.003),
            ),
            lib.CharacteristicGamma(
                energy=ufloat(1332.492, 0.0004) * 1000,
                intensity=ufloat(0.999826, 0.00006),
            ),
        ),
    )
    our_sodium_source = lib.CheckSource(
        element="Na",
        mass_number=22,
        half_life=ufloat(2.6019, 0.00005),
        assay_date=datetime.datetime(2013, 1, 1),
        record_date=datetime.datetime.today(),
        initial_activity=ufloat(0.984, 0.05 * 0.984),  # microcurie
        characteristic_gammas=(
            lib.CharacteristicGamma(
                energy=511.0 * 1000,
                intensity=ufloat(1.7991, 0.0018),
            ),
        ),
    )

    # TODO deprecated
    # data.check_sources.extend(
    #    [
    #        our_cesium_source,
    #        our_cobalt_source,
    #        our_cobalt_source,
    #        our_sodium_source,
    #    ]
    # )

    # import csv data into internal object
    for path in argns.spectrum_files:
        with open(path, "rt", newline="") as spectrum:
            data.pulse_height_analyses.append(lib.parse_prospect_csv(spectrum))

    # TODO efficiency calculation

    # TODO uncertainties?
    # microcuries
    calculated_activities = {
        "cesium": 0.4965690416 * 1e-6 * 3.7e10,
        "cobalt": 0.1655648803 * 1e-6 * 3.7e10,
        "barium": 0.4119891318 * 1e-6 * 3.7e10,
        "sodium": 0.0287673547 * 1e-6 * 3.7e10,
    }

    efficiencies = []

    # cesium
    efficiencies.append(
        (
            # f/YA, f=counts/second (live time)
            661.657,
            (443 / 120)
            / our_cesium_source.characteristic_gammas[0].intensity
            / calculated_activities["cesium"],
        )
    )

    # cobalt
    efficiencies.append(
        (
            1173.228,
            (497 / 120)
            / our_cobalt_source.characteristic_gammas[0].intensity
            / calculated_activities["cobalt"],
        ),
    )
    efficiencies.append(
        (
            1332.492,
            (420 / 120)
            / our_cobalt_source.characteristic_gammas[1].intensity
            / calculated_activities["cobalt"],
        ),
    )

    # barium
    efficiencies.append(
        (
            275.925,
            (1653 / 120)
            / our_barium_source.characteristic_gammas[0].intensity
            / calculated_activities["barium"],
        )
    )

    # sodium
    efficiencies.append(
        (
            511.0,
            (422 / 120)
            / our_sodium_source.characteristic_gammas[0].intensity
            / calculated_activities["sodium"],
        )
    )

    breakpoint()

    # energies = ()
    # counts = ()
    # efficiencies = ()
    # lib.calculate_radioactivity()
    # zip(energies, efficiencies)

    # for pha in data.pulse_height_analyses:
    #    if pha.check_source is None:
    #        continue

    #    # these variables are for the exponential decay equation for
    #    # radioactivity
    #    t_half = pha.check_source.half_life
    #    a_0 = pha.check_source.initial_activity
    #    delta_t = pha.check_source.record_date - pha.check_source.assay_date
    #    # convert datetime.timedelta to years
    #    delta_t = delta_t.days / 365 + delta_t.seconds / 60 / 60 / 24 / 365
    #    # radioactivity equation
    #    a = pha.check_source.initial_activity * math.exp(
    #        -(math.ln(2) / t_half) * delta_t
    #    )
    #    for gamma in pha.check_source.characteristic_gammas:
    #        f = gamma.energy
    #        y = gamma.intensity
    #        # efficiency = f / YA
    #        efficiency = / y / a

    # TODO efficiency curve fitting
    # we use the model in equation 11 of https://doi.org/10.1016/j.jclepro.2024.143910.
    # we do not transform it into a linear model by the natural logarithm
    def exp_log_poly(A, B, C, D, x):
        return math.exp(
            A
            + B * math.log(x)
            + C * (math.log(x)) ** 2
            + D * (math.log(x)) ** 3
        )

    # x: energies in keV
    # y: efficiency counts/photon
    efficiency_fit_params, efficiency_fit_cov = scipy.optimize.curve_fit(
        exp_log_poly,
        [point[0].nominal_value for point in efficiencies],
        [point[1].nominal_value for point in efficiencies],
    )

    if argns.generate_plots:
        # TODO extend to all PHAs, not just the first
        # clip negative energies
        data.pulse_height_analyses[0].energies.sort()
        for index, energy in enumerate(data.pulse_height_analyses[0].energies):
            if energy > 0:
                positive_energies = data.pulse_height_analyses[0].energies[
                    index:
                ]
                positive_counts = data.pulse_height_analyses[0].counts[index:]
                break

        fig, ax = plt.subplots()
        # data.pulse_height_analyses[0].energies,
        # data.pulse_height_analyses[0].counts,
        ax.bar(
            positive_energies,
            positive_counts,
            width=1,
        )
        ax.set_xlabel("gamma ray energy (keV)")
        ax.set_ylabel("gamma ray count")
        ax.set_title("Pulse Height Analysis of Unknown Radioactive Isotope")
        plt.show()
        # fig.savefig('pha.svg')
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(
            [point[0].nominal_value for point in efficiencies],
            [point[1].nominal_value for point in efficiencies],
        )
        sample_pts = np.linspace(0, 1500, 100)
        fit_y_values = exp_log_poly(
            efficiency_fit_params[0],
            efficiency_fit_params[1],
            efficiency_fit_params[2],
            efficiency_fit_params[3],
            sample_pts,
        )
        plt.plot(
            [point[0].nominal_value for point in efficiencies], fit_y_values
        )
        plt.show()
