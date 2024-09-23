import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from spacepy.pycdf import CDF
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps, AlphaTemperatureDensityCalibrationTable
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    calculate_clock_angle, calculate_deflection_angle
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


def run_example_dat_files():
    data = read_l2_data_from_dat("swapi/test_data/swapi_test_data_v3.dat")
    coincident_count_rate = uarray(data.coincidence_count_rate, data.coincidence_count_rate_uncertainty)
    energy = data.energy
    alpha_speed = calculate_alpha_solar_wind_speed(coincident_count_rate, energy)

    temperature_density_lut = AlphaTemperatureDensityCalibrationTable.from_file(
        r"swapi/test_data/imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf")
    alpha_temperature, alpha_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
        temperature_density_lut, alpha_speed,
        coincident_count_rate, energy)


def main(file_path):
    if file_path[-4:] == ".dat":
        data = read_l2_data_from_dat(file_path)
    elif file_path[-4:] == ".cdf":
        data = read_l2_data(os.path.abspath(file_path))
    else:
        raise Exception("The demo can only load data from .dat or .cdf files!")

    coincident_count_rate = uarray(data.coincidence_count_rate, data.coincidence_count_rate_uncertainty)
    energy = data.energy
    alpha_speed = calculate_alpha_solar_wind_speed(coincident_count_rate, energy)

    plot_sweeps(data)

    temperature_density_lut = AlphaTemperatureDensityCalibrationTable.from_file(
        r"test_data/imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf")
    alpha_temperature, alpha_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
        temperature_density_lut, alpha_speed,
        coincident_count_rate, energy)

    print(f"SW He++ speed: {alpha_speed}")
    print(f"SW He++ temperature: {alpha_temperature}")
    print(f"SW He++ density: {alpha_density}")

    if __name__ == "__main__":
        fig.legend([f"SW He++ speed: {alpha_speed :.3f}",
                    f"SW He++ temperature: {alpha_temperature :.3f}",
                    f"SW He++ density: {alpha_density :.3f}"],
                   )

        fig.set_figheight(10)
        plt.show()


if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
        main(file_path)
    except:
        main(os.path.abspath("test_data/swapi_test_data_v3.dat"))
