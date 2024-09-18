import itertools

import numpy as np

u_sw_values = np.arange(350, 875, 500)
deflection_values = np.arange(0, 5.25, 5)
clock_angles = np.arange(0, 365, 360)
densities = np.arange(1, 6.5, 0.5)
temps = np.arange(1e4, 1.6e5, 1e4)

with open("imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240905_v002.cdf", 'w') as f:
    f.write("#\n")
    f.write("# Created on: 2024-08-23T20:04:16.44396543502487Z\n")
    f.write("#\n")
    f.write("# Example LUT for SW Density and Temperature:\n")
    f.write("#\n")
    f.write("#   u_sw     n     density (cm^-3)     T     temperature (K)\n")

    for u_sw, density, temp in itertools.product(u_sw_values, densities, temps):
        f.write(
            f"{u_sw}  {density}  {density * 1.021:.4f}  {temp:.4e}  {temp / 1.025:0.4e}\n")
