import argparse
import csv
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import lib

from uncertainties import ufloat, unumpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spectrum-files", nargs="+")
    parser.add_argument("-p", "--generate-plots", action="store_true")
    argns = parser.parse_args()

    data = lib.MysteryIsotopeMeasurements(
        coarse_gain=1.0, fine_gain=2.0, detector_voltage=900
    )
    # TODO fill in assay dates and initial activities
    data.check_sources.extend(
        [
            lib.CheckSource(
                element="Cs",
                mass_number=137,
                assay_date=datetime.datetime.today(),
                initial_activity=1,
                characteristic_gammas=[661.657],
            ),
            lib.CheckSource(
                element="Ba",
                mass_number=133,
                assay_date=datetime.datetime.today(),
                initial_activity=1,
                characteristic_gammas=[275.925],
            ),
            lib.CheckSource(
                element="Co",
                mass_number=60,
                assay_date=datetime.datetime.today(),
                initial_activity=1,
                characteristic_gammas=[1173.228, 1332.492],
            ),
            lib.CheckSource(
                element="Na",
                mass_number=22,
                assay_date=datetime.datetime.today(),
                initial_activity=1,
                characteristic_gammas=[511.0],
            ),
        ]
    )

    # import csv data into internal object
    for path in argns.spectrum_files:
        with open(path, "rt", newline="") as spectrum:
            # consume the header lines before invoking the csv reader
            header = []
            for line in spectrum:
                if line.startswith("Spectrum"):
                    break
                header.append(line.strip("\r\n"))

            # Fri Apr 10 12:04:11 GMT-0400 2026
            # TODO strip until after ", "
            start_time = datetime.datetime.strptime(
                header[0].split(", ")[-1], "%a %b %d %H:%M:%S %Z%z %Y"
            )

            # at this point, spectrum file object is seeked to the header line "Channel, Energy (keV), Counts"
            reader = csv.DictReader(spectrum, skipinitialspace=True)
            spectrum_data = {"channels": [], "energies": [], "counts": []}
            for rowdict in reader:
                spectrum_data["channels"].append(int(rowdict["Channel"]))
                spectrum_data["energies"].append(float(rowdict["Energy (keV)"]))
                spectrum_data["counts"].append(int(rowdict["Counts"]))
            # convert to arrays after, to take advantage of python's list type being linked for quick appending.
            # using numpy.append would do a copy operation for each line in the csv file.
            spectrum_data["channels"] = np.array(spectrum_data["channels"])  # ty: ignore[invalid-assignment]
            spectrum_data["energies"] = np.array(spectrum_data["energies"])  # ty: ignore[invalid-assignment]
            spectrum_data["counts"] = np.array(spectrum_data["counts"])  # ty: ignore[invalid-assignment]

            # collate information in a PulseHeightAnalysis instance
            # TODO parser for CSV header start time, live time, real time, energy calibration
            data.pulse_height_analyses.append(
                lib.PulseHeightAnalysis(
                    start_time=datetime.datetime.today(),
                    live_time=120,
                    real_time=130,
                    energy_calibration=(1.1, 10.5, 4.56),
                    **spectrum_data,
                )
            )

    if argns.generate_plots:
        # clip negative energies
        data.pulse_height_analyses[0].energies.sort()
        for index, energy in enumerate(data.pulse_height_analyses[0].energies):
            if energy > 0:
                positive_energies = data.pulse_height_analyses[0].energies[index:]
                positive_counts = data.pulse_height_analyses[0].counts[index:]
                break

        # TODO matplotlib stuff
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
