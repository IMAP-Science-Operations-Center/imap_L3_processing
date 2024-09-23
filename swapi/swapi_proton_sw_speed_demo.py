import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from spacepy.pycdf import CDF
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    calculate_clock_angle, calculate_deflection_angle, ClockAngleCalibrationTable
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import sine_fit_function, \
    calculate_proton_solar_wind_speed, calculate_proton_centers_of_mass, extract_coarse_sweep


def read_l2_data_from_dat(file_path: str) -> SwapiL2Data:
    data = np.loadtxt(file_path)
    data = data.reshape((-1, 72, 8))

    twelve_seconds_in_nanoseconds = 12_000_000_000
    epochs = twelve_seconds_in_nanoseconds * np.arange(len(data))

    energy_for_first_sweep = data[0, :, 2]
    spin_angles = data[..., 3]
    coarse_sweep_energies = data[:, 1:63, 2]
    assert np.all(coarse_sweep_energies == coarse_sweep_energies[0])

    coincident_count_rates = data[..., 7]

    fake_coincident_count_rate_uncertainties = np.sqrt(6 * coincident_count_rates)
    return SwapiL2Data(epochs,
                       energy_for_first_sweep,
                       coincident_count_rates,
                       spin_angles,
                       fake_coincident_count_rate_uncertainties)


def read_l2_data(cdf_path: str) -> SwapiL2Data:
    cdf = CDF(cdf_path)
    return SwapiL2Data(cdf.raw_var("epoch")[...],
                       cdf["energy"][...],
                       cdf["swp_coin_rate"][...],
                       cdf["spin_angles"][...],
                       cdf["swp_coin_unc"][...])


fig = plt.figure()


def plot_sweeps(data):
    for i in range(len(data.epoch)):
        axes = fig.add_subplot(len(data.epoch) + 1, 1, i + 1)
        axes.loglog(data.energy, data.coincidence_count_rate[i, :], marker='.', linestyle="None")
        axes.set(xlabel="Energy", ylabel="Count Rate")


def plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass):
    fit_xs = np.arange(0, 360, 3)
    fit_ys = sine_fit_function(fit_xs, a.n, phi.n % 360, b.n)
    plot = fig.add_subplot(len(spin_angles) + 1, 1, len(spin_angles) + 1)

    plot.errorbar(spin_angles, nominal_values(centers_of_mass), yerr=std_devs(centers_of_mass), fmt=".")

    plot.plot(fit_xs, fit_ys)
    plot.set(xlabel="Phase Angle", ylabel="Energy")


def run_example_dat_files():
    data = read_l2_data_from_dat("test_data/swapi_test_data_v4.dat")
    coincident_count_rate = uarray(data.coincidence_count_rate, data.coincidence_count_rate_uncertainty)
    proton_sw_speed, a, phi, b = calculate_proton_solar_wind_speed(coincident_count_rate, data.spin_angles, data.energy,
                                                                   data.epoch)

    clock_angle_lut = ClockAngleCalibrationTable.from_file('test_data/example_LUT_flow_angle_v2.dat')

    clock_angle = calculate_clock_angle(clock_angle_lut, proton_sw_speed, a, phi, b)

    deflection_angle = calculate_deflection_angle(clock_angle_lut, proton_sw_speed, a, phi, b)

    print(f"clock: {clock_angle}")
    print(f"deflection angle: {deflection_angle}")


def main(file_path):
    if file_path[-4:] == ".dat":
        data = read_l2_data_from_dat(file_path)
    elif file_path[-4:] == ".cdf":
        data = read_l2_data(os.path.abspath(file_path))
    else:
        raise Exception("The demo can only load data from .dat or .cdf files!")

    coincident_count_rate = uarray(data.coincidence_count_rate, data.coincidence_count_rate_uncertainty)

    plot_sweeps(data)

    centers_of_mass, spin_angles = calculate_proton_centers_of_mass(
        extract_coarse_sweep(coincident_count_rate),
        extract_coarse_sweep(data.spin_angles),
        extract_coarse_sweep(data.energy),
        data.epoch)
    proton_sw_speed, a, phi, b = calculate_proton_solar_wind_speed(coincident_count_rate, data.spin_angles, data.energy,
                                                                   data.epoch)

    plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass)

    print(f"SW H+ speed: {proton_sw_speed}")
    print(f"SW H+ clock angle: {phi}")
    print(f"A {a}")
    print(f"B {b}")
    print(f"A/B {a / b}")

    if __name__ == "__main__":
        fig.legend([f"SW H+ speed: {proton_sw_speed :.3f}",
                    f"SW H+ clock angle: {phi:.3f}",
                    f"A: {a:.3f}",
                    f"B: {b:.3f}",
                    f"A/B: {(a / b):.5f}"],
                   )

        fig.set_figheight(10)
        plt.show()


if __name__ == "__main__":
    run_example_dat_files()
    # try:
    #     file_path = sys.argv[1]
    #     main(file_path)
    # except:
    #     main(os.path.abspath("test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf"))
