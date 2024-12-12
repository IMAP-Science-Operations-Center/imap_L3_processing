from random import random, randint

import numpy as np


def get_sweep_voltages(sweep_table_id=0):
    energies = np.geomspace(100, 19000, 62)
    return energies / get_k_factor()


def generate_sweep_data(center):
    voltages = get_sweep_voltages()
    energies = voltages * get_k_factor()
    background = 0.1
    proton_peak = generate_peak(energies, 1000, center, 30)
    alpha_peak = generate_peak(energies, 30, 2200, 30)

    return proton_peak + alpha_peak + background


def generate_sweeps(n=5):
    time_base = 315576112184000000  # Jan 1, 2010 00:00:46
    start_times_in_seconds = time_base + np.arange(n)*12
    seed = randint(0, 359)
    angles = [seed - 72*i for i in range(5)]
    center_points = 1050 + 20*np.sin(np.deg2rad(angles))
    sweeps = [generate_sweep_data(c) for c in center_points]
    return list(zip(start_times_in_seconds, sweeps))


def generate_sweep_energies():
    coarse_sweep_energies = np.geomspace(100, 19000, 62)
    return np.concatenate((np.array([np.nan]), coarse_sweep_energies, np.array([np.nan] * 9)))


# K Factor may come from a lookup table in the future
def get_k_factor():
    return 1.8


# This is generating fake data. Real implementation will need to look up rotation values using SPICE for each time point
def get_spin_phase_using_spice(time):
    arbitrary_offset = 0.4
    rotation_time = 15
    rotations = arbitrary_offset + time/rotation_time
    fractional, integral = np.modf(rotations)
    return fractional*360


def generate_peak(energies, height, center, narrowness):
    return np.exp(np.log(height) - narrowness * np.square(np.log(energies) - np.log(center)))

