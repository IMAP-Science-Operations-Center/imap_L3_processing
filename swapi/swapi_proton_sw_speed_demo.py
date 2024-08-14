import sys

import numpy as np
from matplotlib import pyplot as plt
from spacepy.pycdf import CDF

from imap_processing.swapi.l3a.models import SwapiL2Data, SwapiL3Data
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import sine_fit_function, \
    calculate_proton_solar_wind_speed, calculate_proton_centers_of_mass, extract_coarse_sweep


def read_l2_data(cdf_path: str) -> SwapiL2Data:
    cdf = CDF(cdf_path)
    return SwapiL2Data(cdf.raw_var("epoch")[...],
                       extract_coarse_sweep(cdf["energy"][...]),
                       extract_coarse_sweep(cdf["swp_coin_rate"][...]),
                       extract_coarse_sweep(cdf["spin_angles"][...]))


fig = plt.figure()

def plot_sweeps(data):
    for i in range(len(data.epoch)):
        axes = fig.add_subplot(len(data.epoch) + 1, 1, i + 1)
        axes.loglog(data.energy, data.coincidence_count_rate[i, :], marker='.', linestyle="None")
        axes.set(xlabel="Energy", ylabel="Count Rate")


def plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass):
    fit_xs = np.arange(0, 360, 3)
    fit_ys = sine_fit_function(fit_xs, a, phi % 360, b)
    plot = fig.add_subplot(len(spin_angles) + 1, 1, len(spin_angles) + 1)

    plot.scatter(spin_angles, centers_of_mass)
    plot.plot(fit_xs, fit_ys)
    plot.set(xlabel="Phase Angle", ylabel="Energy")

def main(file_path):
    data = read_l2_data(file_path)

    plot_sweeps(data)

    centers_of_mass, spin_angles = calculate_proton_centers_of_mass(data.coincidence_count_rate, data.spin_angles, data.energy, data.epoch)
    proton_sw_speed, a, phi, b = calculate_proton_solar_wind_speed(data.coincidence_count_rate, data.spin_angles, data.energy, data.epoch)

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
    try:
        file_path = sys.argv[1]
        main(file_path)
    except:
        main("swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v001.cdf")
