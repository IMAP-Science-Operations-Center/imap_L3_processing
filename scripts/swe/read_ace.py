#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 16:12:45 2025

@author: hafijulislam
"""
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from pyhdf.HDF import *
from pyhdf.SD import SD, SDC
# import pyhdf.SD
from pyhdf.VS import *
from spacepy.pycdf import CDF

SD
VS


def decompressed_counts(cem_count: int) -> int:
    """
    Decompressed counts from the CEMs.

    Parameters
    ----------
    cem_count : int
        CEM counts. Eg. 243.

    Returns
    -------
    int
        Decompressed count. Eg. 40959.
    """
    # index is the first four bits of input data
    # multi is the last four bits of input data
    index = cem_count // 16
    multi = cem_count % 16

    # This is look up table for the index to get
    # base and step_size to calculate the decompressed count.
    decompress_table = {
        0: {"base": 0, "step_size": 1},
        1: {"base": 16, "step_size": 1},
        2: {"base": 32, "step_size": 2},
        3: {"base": 64, "step_size": 4},
        4: {"base": 128, "step_size": 8},
        5: {"base": 256, "step_size": 16},
        6: {"base": 512, "step_size": 16},
        7: {"base": 768, "step_size": 16},
        8: {"base": 1024, "step_size": 32},
        9: {"base": 1536, "step_size": 32},
        10: {"base": 2048, "step_size": 64},
        11: {"base": 3072, "step_size": 128},
        12: {"base": 5120, "step_size": 256},
        13: {"base": 9216, "step_size": 512},
        14: {"base": 17408, "step_size": 1024},
        15: {"base": 33792, "step_size": 2048},
    }

    # decompression formula from SWE algorithm document CN102D-D0001 and page 16.
    # N = base[index] + multi * step_size[index] + (step_size[index] - 1) / 2
    # NOTE: for (step_size[index] - 1) / 2, we only keep the whole number part of
    # the quotient

    return (
            decompress_table[index]["base"]
            + (multi * decompress_table[index]["step_size"])
            + ((decompress_table[index]["step_size"] - 1) // 2)
    )


def hdf4_to_xarray(hdf_file_path):
    """
    Reads an HDF4 file and converts it to an xarray.Dataset.

    Args:
    - hdf_file_path (str): Path to the HDF4 file.

    Returns:
    - xarray.Dataset: The dataset converted to an xarray.
    """
    # Open the HDF4 file
    hdf_file = SD(hdf_file_path, SDC.READ)

    xarray_datasets = {}

    # Loop through all datasets in the HDF4 file
    for dataset_name, (dim_names, dim_sizes, _, _) in hdf_file.datasets().items():
        # Read the dataset
        dataset = hdf_file.select(dataset_name)
        try:
            data = dataset.get()  # Explicit read using the get method
        except Exception as e:
            print(f"Error reading dataset {dataset_name}: {e}")
            continue

        # Use dim_names as dimension names, and dim_sizes as dimension sizes
        coords = {dim_names[i]: range(dim_sizes[i]) for i in range(len(dim_names))}

        # Convert the dataset into an xarray.DataArray
        data_xarray = xr.DataArray(data, dims=dim_names, coords=coords)

        # Add the DataArray to the dictionary with the dataset's name
        xarray_datasets[dataset_name] = data_xarray

    # Convert the dictionary of DataArrays to an xarray.Dataset
    xarray_dataset = xr.Dataset(xarray_datasets)

    return xarray_dataset


def get_phase_and_spin(l2_swepam_electron_file_path: str):
    l2_swepam_electron_file_path = HDF(l2_swepam_electron_file_path)
    vs_electron = l2_swepam_electron_file_path.vstart()
    vd_electron = vs_electron.attach("swepam_e")

    phase_index = vd_electron.field("phase")._index
    spin_index = vd_electron.field("spin")._index
    phase = np.array([x[phase_index] for x in vd_electron[:]])
    spin = np.array([x[spin_index] for x in vd_electron[:]])
    return phase, spin


def calculate_phase(phase, spin_period):
    N_ENERGIES = 20
    N_ENERGIES_LEVEL1 = 4
    N_PHI_LEVEL1 = 30
    K_FOR_THIS_MODE = 3.1845564
    SWE_MOUNTING_ANGLE = 153
    phi = np.zeros((20, 30))
    for i in range(N_ENERGIES):
        for k in range(30):
            rrr = i % N_ENERGIES_LEVEL1
            sss = i // N_ENERGIES_LEVEL1
            f1 = N_PHI_LEVEL1 * sss + k + 1
            f2 = 120 * sss + N_ENERGIES_LEVEL1 * k + rrr + 1
            offset = (0.0547645 + f1 * 0.0575027 + (f2 - 0.5) * K_FOR_THIS_MODE) * 12.0 / spin_period
            phi[i][k] = phase + offset
    CONVERT_PHI_TO_IMAP_AZIMUTH = 270
    return (phi + CONVERT_PHI_TO_IMAP_AZIMUTH - SWE_MOUNTING_ANGLE) % 360


def calculate_acquisition_time(phase: float, spin_period: float, epoch: datetime) -> np.ndarray:
    met = get_met_from_epoch(epoch)

    met_rounded = (met // 15) * 15
    phase_angle = calculate_phase(phase, spin_period)
    return (phase_angle * 15 / 360) + met_rounded


def deadtime_correction(counts, sample_time):
    correct = 1.0 - ((1.5e-6) * counts / sample_time)
    correct = np.maximum(0.1, correct)
    return counts / correct


def plot_dnswe_data_from_hdf(file_path):
    # Open the HDF4 file for reading
    hdf = SD(file_path, SDC.READ)
    # Select the DNSWE_COUNT dataset
    data = hdf.select('DNSWE_COUNT')[:]

    # Print the shape and data for reference
    print(f"Shape of DNSWE_COUNT dataset: {data.shape}")
    print(f"Data:\n{data}")

    # For simplicity, let's plot the first 2D slice (you can change this depending on your needs)
    data_slice = data[:, 0, 0, 0, :]  # Select a slice (e.g., first 2D slice from the 5D array)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.imshow(data_slice, norm=LogNorm(), aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='DNSWE Values')
    plt.title('Plot of DNSWE_COUNT dataset slice')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def get_met_from_epoch(epoch: datetime) -> float:
    initial_epoch = datetime.fromisoformat("2025-06-30T12:00:00")
    initial_epoch_in_met_time = 488980803.0

    delta = epoch - initial_epoch

    return initial_epoch_in_met_time + delta.total_seconds()


def get_epochs_from_output_file(filepath: str) -> np.array:
    hdf_file = HDF(filepath)
    vs = hdf_file.vstart()
    dataset = vs.attach("swepam_e")

    years_index = dataset.field("year")._index
    month_index = dataset.field("mon")._index
    day_index = dataset.field("mday")._index
    hour_index = dataset.field("hour")._index
    min_index = dataset.field("min")._index
    sec_index = dataset.field("sec")._index

    correction_factor = (datetime(2025, 6, 30) - datetime(1999, 6, 8))

    return np.array([datetime(year=x[years_index], month=x[month_index], day=x[day_index], hour=x[hour_index],
                              minute=x[min_index], second=x[sec_index]) + correction_factor for x in dataset[:]])


initial_epoch = datetime.fromisoformat("2025-06-30T12:00:00")
initial_epoch_in_met_time = 488980803.0
# File path to the HDF4 file
file_path = "instrument_team_data/swe/ACE_LV1_1999-159.swepam.hdf"
xarray_data = hdf4_to_xarray(file_path)
new_ds = xarray_data['DNSWE_COUNT'].transpose('fakeDim32', 'fakeDim33', 'fakeDim35', 'fakeDim34', 'fakeDim36')
new_ds = new_ds.stack(merged_dim=("fakeDim33", "fakeDim35"))
new_ds = new_ds.assign_coords({"merged_dim": range(20)})
new_ds = new_ds.transpose('fakeDim32', 'merged_dim', 'fakeDim34', 'fakeDim36')

epochs = get_epochs_from_output_file("instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf")

ds_expanded = xr.DataArray(
    new_ds.values,
    dims=("dim1", "dim2", "dim3", "dim4"),
    coords={
        "dim1": 'epoch',  # Keep original coordinates
        "dim2": 'energy',  # Update dim2 to be 0-23
        "dim3": 'spin',
        "dim4": 'cem'
    }
)
decompress_table = decompression_table = np.array([decompressed_counts(i) for i in range(256)])
counts = decompress_table[ds_expanded.values.astype(int)]
sample_time = 0.10031
sample_time_microseconds = int(sample_time * 1e6)
deadtime_corrected = deadtime_correction(counts, sample_time)
rates = deadtime_corrected / sample_time

phase, spin = get_phase_and_spin('instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf')

acquisition_time = []

for i in range(len(phase)):
    acquisition_time.append(calculate_acquisition_time(phase[i], spin[i], epochs[i]))

time_between_data_points = 128 * 1e6

if len(sys.argv) > 1:
    truncate_to = int(sys.argv[1])
else:
    truncate_to = len(epochs)
settle_duration_needed_to_fill_time_between_points = time_between_data_points / (20 * 30) - sample_time_microseconds
output_path = "tests/test_data/swe/imap_swe_l1b_sci_20250630_v003.cdf"
shutil.copy("tests/test_data/swe/imap_swe_l1b_sci_20240510_v002.cdf", output_path)

energy = np.array([2.55714286, 3.65142857, 5.16, 7.30571429,
                   10.32857143, 14.34285714, 19.95714286, 27.42857143,
                   38.37142857, 52.82857143, 73.32857143, 102.0,
                   142.14285714, 196.57142857, 272., 372.71428571,
                   519.0, 712.57142857, 987.14285714, 1370.0])

esa_energy = np.full((truncate_to, 20, 30), np.nan)

esa_energy[:] = energy[np.newaxis, :, np.newaxis]

energy = energy[:, np.newaxis, np.newaxis]
with CDF(output_path, readonly=False) as cdf:
    del cdf['science_data']
    del cdf['acquisition_time']
    del cdf['acq_duration']
    del cdf['settle_duration']
    del cdf['esa_step']
    cdf['epoch'] = epochs[:truncate_to]
    cdf['epoch'].attrs['VAR_TYPE'] = 'support_data'
    del cdf['epoch'].attrs['DEPEND_0']
    cdf['science_data'] = rates[:truncate_to, :, :, ::-1]
    cdf['acquisition_time'] = np.array(acquisition_time[:truncate_to])
    cdf['acq_duration'] = np.full((truncate_to, 20, 30), sample_time_microseconds)
    cdf['settle_duration'] = np.full((truncate_to, 4), round(settle_duration_needed_to_fill_time_between_points))
    cdf['esa_table_num'] = np.full((truncate_to, 4), 0)
    cdf.new('esa_energy', esa_energy)
    cdf.new('esa_step', np.arange(20), recVary=False)
    cdf['esa_energy'].attrs['VAR_TYPE'] = 'support_data'
    cdf['esa_step'].attrs['VAR_TYPE'] = 'support_data'
