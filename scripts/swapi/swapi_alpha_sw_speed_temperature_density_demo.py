import os
import sys

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import ndarray
from spacepy.pycdf import CDF
from uncertainties import ufloat, correlated_values
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed, \
    calculate_combined_sweeps, get_alpha_peak_indices
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps, AlphaTemperatureDensityCalibrationTable, \
    alpha_count_rate_model


def read_l2_data_from_dat(file_path: str) -> SwapiL2Data:
    data = np.loadtxt(file_path)
    data = data.reshape((-1, 72, 8))
    start_time = int(np.datetime64("2010-01-01", "ns") - np.datetime64("2000-01-01T12:00", "ns"))
    twelve_seconds_in_nanoseconds = 12_000_000_000
    epochs = start_time + twelve_seconds_in_nanoseconds * np.arange(len(data))

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
    data = read_l2_data_from_dat("swapi/test_data/swapi_test_data_v5.dat")
    coincident_count_rate = uarray(data.coincidence_count_rate, data.coincidence_count_rate_uncertainty)
    energy = data.energy
    alpha_speed = calculate_alpha_solar_wind_speed(coincident_count_rate, energy)

    temperature_density_lut = AlphaTemperatureDensityCalibrationTable.from_file(
        r"swapi/test_data/imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf")
    alpha_temperature, alpha_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
        temperature_density_lut, alpha_speed,
        coincident_count_rate, energy)


def plot_and_calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
        table: AlphaTemperatureDensityCalibrationTable,
        alpha_sw_speed: ufloat,
        count_rates: uarray,
        energies: ndarray,
):
    average_count_rates, energies = calculate_combined_sweeps(count_rates, energies)

    alpha_particle_peak_slice = get_alpha_peak_indices(average_count_rates, energies)

    peak_energies = energies[alpha_particle_peak_slice]
    peak_average_alpha_count_rates = average_count_rates[alpha_particle_peak_slice]

    initial_parameter_guess = [0.15, 3.6e5, nominal_values(alpha_sw_speed)]
    values, covariance = scipy.optimize.curve_fit(alpha_count_rate_model,
                                                  peak_energies,
                                                  nominal_values(peak_average_alpha_count_rates),
                                                  sigma=std_devs(peak_average_alpha_count_rates),
                                                  absolute_sigma=True,
                                                  bounds=[[0, 0, 0], [np.inf, np.inf, np.inf]],
                                                  p0=initial_parameter_guess)
    residual = abs(alpha_count_rate_model(peak_energies, *values) - nominal_values(peak_average_alpha_count_rates))
    reduced_chisq = np.sum(np.square(residual / std_devs(peak_average_alpha_count_rates))) / (len(peak_energies) - 3)
    if reduced_chisq > 10:
        raise ValueError("Failed to fit - chi-squared too large", reduced_chisq)
    density, temperature, speed = correlated_values(values, covariance)

    density = table.lookup_density(speed, density, temperature)
    temperature = table.lookup_temperature(speed, density, temperature)

    plt.loglog(energies, nominal_values(average_count_rates), marker="o")
    plt.xlabel("Energy")
    plt.ylabel("Average count rates over 5 sweeps")
    es = np.geomspace(peak_energies[0], peak_energies[-1], 100)
    plt.loglog(es, alpha_count_rate_model(es, *values), linewidth=2.0)
    plt.savefig("alpha_temp_dens_fitting")
    plt.show()

    return density, temperature


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

    temperature_density_lut = AlphaTemperatureDensityCalibrationTable.from_file(
        r"../../tests/test_data/swapi/imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf")
    alpha_temperature, alpha_density = plot_and_calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
        temperature_density_lut, alpha_speed,
        coincident_count_rate, energy)

    print(f"SW He++ speed: {alpha_speed.n} +/- {alpha_speed.s}")
    print(f"SW He++ temperature: {alpha_temperature.n} +/- {alpha_temperature.s}")
    print(f"SW He++ density: {alpha_density.n} +/- {alpha_density.s}")

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
        main(os.path.abspath("../../instrument_team_data/swapi/swapi_test_data_v5.dat"))
