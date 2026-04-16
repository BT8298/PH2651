import scipy.optimize
import collections.abc
import json
import numpy as np
import uncertainties
from uncertainties import ufloat
import uncertainties.umath as umath
from datetime import datetime

# 3% voltage accuracy per the manual
detector_voltage=ufloat(900, 0.03 * 900),
# 5% live time correction accuracy per the manual
# TODO not sure if this is the correct formula for 5% accuracy
live_time = ufloat(120, 0.05 * 120)
fine_gain = 1.0
coarse_gain = 2.0

check_source_properties = {
    "137Cs": {
        # in years
        "half_life": ufloat(30.007, 0.00046),
        # we assume time uncertainty is neglibile, since these sources have been decaying for years
        "assay_date": datetime(1995, 10, 13),
        # TODO
        "calibration_date": datetime.today(),
        # TODO and uncertainty
        "initial_activity": 1 * 1e-6 * 3.7e10,
        # some energies have no listed uncertainty on www.nndc.bnl.gov/nudat3
        # all energies are in keV
        "energies": (
            31.817,
            32.194,
            ufloat(661.657, 0.0003),
        ),
        "intensities": (
            ufloat(0.0199, 0.00005),
            ufloat(0.0364, 0.00001),
            ufloat(0.8510, 0.00020),
        ),
        # TODO
        # to be calculated in ProSpect software
        "peak_counts": (1,1,1),
    },
    "133Ba": {
        "half_life": ufloat(10.536, 0.0008),
        "assay_date": datetime(2013, 1, 3),
        # TODO
        "calibration_date": datetime.today(),
        # TODO
        "initial_activity": 1 * 1e-6 * 3.7e10,
        "energies": (
            4.29,
            30.625,
            30.973,
            ufloat(80.9979, 0.000011),
            ufloat(302.8508, 0.00005),
            ufloat(356.0129, 0.00007),
        ),
        "intensities": (
            ufloat(0.157, 0.0008),
            ufloat(0.339, 0.0001),
            ufloat(0.622, 0.00018),
            ufloat(0.329, 0.0003),
            ufloat(0.1834, 0.000013),
            0.6205,
        ),
        # TODO
        "peak_counts": (1,1,1,1,1,1),
    },
    "60Co": {
        "half_life": ufloat(5.271, 0.0002),
        "assay_date": datetime(2013, 1, 1),
        # TODO
        "calibration_date": datetime.today(),
        # NIST traceable, 5% uncertainty, microcuries Bq
        "initial_activity": ufloat(0.988, 0.05 * 0.988) * 1e-6 * 3.7e10,
        "energies": (
            ufloat(1173.228, 0.0003),
            ufloat(1332.492, 0.0004),
        ),
        "intensities": (
            ufloat(0.9985, 0.00003),
            ufloat(0.999826, 0.0000006),
        ),
        # TODO
        "peak_counts": (1,1),
    },
    "22Na": {
        "half_life": ufloat(2.6019, 0.00005),
        "assay_date": datetime(2013, 1, 1),
        "calibration_date": datetime.today(),
        "initial_activity": ufloat(0.984, 0.05 * 0.984) * 1e-6 * 3.7e10,
        "energies": (
            511.0,
            ufloat(1274.537, 0.0007),
        ),
        "intensities": (
            ufloat(1.7991, 0.000018),
            ufloat(0.99940, 0.0000014),
        ),
        # TODO
        "peak_counts": (1,1),
    },
}

all_energies = []
all_intensities = []
all_efficiencies = []
for isotope in check_source_properties.keys():
    # A = A₀ exp( ln(2)/t_1/2 * t )
    activity = check_source_properties[isotope]["initial_activity"] * umath.exp(
        -umath.log(2)
        / check_source_properties[isotope]["half_life"]
        * (
            check_source_properties[isotope]["calibration_date"]
            - check_source_properties[isotope]["assay_date"]
        ).days
        / 365
    )
    for energy, intensity, count in zip(
        check_source_properties[isotope]["energies"],
        check_source_properties[isotope]["intensities"],
        check_source_properties[isotope]["peak_counts"],
    ):
        efficiency = (count / live_time) / intensity / activity
        all_efficiencies.append(efficiency)
        all_energies.append(energy)

def scrub_uncertainties(x):
    if isinstance(x, collections.abc.Sequence):
        scrubbed = []
        for i in x:
            if isinstance(i, uncertainties.UFloat):
                scrubbed.append(i.nominal_value)
            else:
                scrubbed.append(i)
        return scrubbed
    elif isinstance(x, uncertainties.core.UFloat):
        return x.nominal_value

# efficiency curve fit model given by equation 11 in
# https://doi.org/10.1016/j.jclepro.2024.143910. we do not apply the natural
# logarithm transform to make it a linear model as it may magnify or diminish
# errors in an unexpected way
# (https://en.wikipedia.org/wiki/Nonlinear_regression#Transformation). scipy's
# curve_fit uses a nonlinear least squares algorithm.
def exp_log_poly(x, A, B, C, D):
    return np.exp(
        A + B * np.log(x) + C * (np.log(x)) ** 2 + D * (np.log(x)) ** 3
    )
fit = scipy.optimize.curve_fit(
    #exp_log_poly,
    lambda x, A, B, C, D: np.exp(
        A + B * np.log(x) + C * (np.log(x)) ** 2 + D * (np.log(x)) ** 3
    ),
    xdata=scrub_uncertainties(all_energies),
    ydata=scrub_uncertainties(all_efficiencies),
)

print('exp-log-poly fit parameters:', fit)
with open('efficiency_datapoints.json', 'w') as file:
    json.dump({'energy': all_energies, 'efficiency': all_efficiencies}, file)
