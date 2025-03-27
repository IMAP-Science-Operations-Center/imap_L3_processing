import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyhdf
import pyhdf.SD
from pyhdf.HDF import *
from pyhdf.VS import *
from spacepy import pycdf
from spacepy.pycdf import CDF


def create_fake_swe_l1b_and_l2_cdf(l1_hdf_path: str, l2_hdf_path: str, output_l1b_swe_cdf_file_path: str,
                                   output_l2_swe_cdf_file_path: str):
    l1_hdf = pyhdf.SD.SD(l1_hdf_path)
    counts = l1_hdf.select("DNSWE_COUNT")[:]
    total_number_of_energies = 20
    spin_spectors = 30
    apertures = 7
    counts_in_swe_shape = np.moveaxis(counts, 3, 2).reshape((-1, total_number_of_energies, spin_spectors, apertures))

    l2_swepam_electron_file = HDF(l2_hdf_path)
    vs_electron = l2_swepam_electron_file.vstart()
    vd_electron = vs_electron.attach("swepam_e")
    epochs = get_epochs_from_output_file(vd_electron)
    fields = ["chisq_c",
              "chisq_h",
              "n_fc",
              "n_fh",
              "n_ic",
              "n_ih",
              "n_i",
              "q_flux_ic",
              "q_flux_ih",
              "q_flux_i",
              "q_flux_phi_ic",
              "q_flux_phi_ih",
              "q_flux_phi_i",
              "q_flux_theta_ic",
              "q_flux_theta_ih",
              "q_flux_theta_i",
              "t_para_fc",
              "t_para_fh",
              "t_para_ic",
              "t_para_ih",
              "t_para_i",
              "t_perp_fc",
              "t_perp_fh",
              "t_perp_ic",
              "t_perp_ih",
              "t_perp_i",
              "t_mat_ic",
              "t_mat_ih",
              "t_mat_i",
              "t_phi_fc",
              "t_phi_fh",
              "t_phi_ic",
              "t_phi_ih",
              "t_phi_i",
              "t_theta_fc",
              "t_theta_fh",
              "t_theta_ic",
              "t_theta_ih",
              "t_theta_i",
              "v_fc",
              "v_fh",
              "v_ic",
              "v_ih",
              "v_i",
              "v_rtn_fc",
              "v_rtn_fh",
              "v_rtn_ic",
              "v_rtn_ih",
              "v_rtn_i"]
    create_expected_cdf(vd_electron, fields, "swe_moments")

    with CDF(output_l2_swe_cdf_file_path, create=True) as cdf:
        cdf.compress(pycdf.const.GZIP_COMPRESSION)
        cdf["epoch"] = epochs

        evenly_spaced_energies = np.geomspace(1, 1350, 20, endpoint=True)
        create_variable(cdf, "energy", evenly_spaced_energies)

        energies_reshaped = evenly_spaced_energies[:, np.newaxis, np.newaxis]
        psd = counts_in_swe_shape / np.square(energies_reshaped)
        create_variable(cdf, "phase_space_density_spin_sector", psd)

        flux = psd * energies_reshaped
        create_variable(cdf, "flux_spin_sector", flux)

        cdf.new("inst_el", [-63, -42, -21, 0, 21, 42, 63], recVary=False)

        shape_without_aperture_axis = psd.shape[:-1]
        placeholder_insta_az_spin_sector = np.random.random(shape_without_aperture_axis) * 360
        create_variable(cdf, "inst_az_spin_sector", placeholder_insta_az_spin_sector)

        minutes = len(counts_in_swe_shape)
        measurement_count = placeholder_insta_az_spin_sector.size
        measurement_times = np.linspace(0, minutes * 60, measurement_count)
        placeholder_acquisition_time = measurement_times.reshape(shape_without_aperture_axis)
        create_variable(cdf, "acquisition_time", placeholder_acquisition_time)

        acquisition_duration = np.full((minutes, 20, 30), 80000)
        cdf["acq_duration"] = acquisition_duration
    with CDF(output_l1b_swe_cdf_file_path, create=True) as cdf:
        cdf["epoch"] = epochs
        cdf["science_data"] = counts_in_swe_shape
        cdf["settle_duration"] = np.full((minutes, 4), 133333.3333333334)


def create_fake_swapi_l3a_cdf(l2_swepam_ion_file_path: str,
                              output_swapi_file_path: str):
    l2_swepam_ion_file = HDF(l2_swepam_ion_file_path)
    vs_ion = l2_swepam_ion_file.vstart()
    vd_ion = vs_ion.attach("swepam_i")

    sw_velocity_index = vd_ion.field(
        "vel_p_rtn")._index
    sw_velocity_data = np.array([x[sw_velocity_index] for x in vd_ion[:]])
    sw_speed = np.linalg.norm(sw_velocity_data, axis=1)
    sw_r, sw_t, sw_n = sw_velocity_data.T
    sw_x, sw_y, sw_z = sw_n, sw_t, -sw_r
    deflection = np.rad2deg(np.arctan2(np.sqrt(sw_x ** 2 + sw_y ** 2), -sw_z))
    clock = np.rad2deg(np.arctan2(sw_x, -sw_y))

    epochs = get_epochs_from_output_file(vd_ion)

    epoch_deltas = np.full(epochs.shape, 32_000_000_000)

    with CDF(output_swapi_file_path, readonly=False, create=True) as cdf:
        cdf.compress(pycdf.const.GZIP_COMPRESSION)
        cdf['epoch'] = epochs
        cdf['epoch_delta'] = epoch_deltas
        cdf['proton_sw_speed'] = sw_speed
        cdf['proton_sw_speed'].attrs["FILLVAL"] = -1e31
        cdf['proton_sw_clock_angle'] = clock
        cdf['proton_sw_clock_angle'].attrs["FILLVAL"] = -1e31
        cdf['proton_sw_deflection_angle'] = deflection
        cdf['proton_sw_deflection_angle'].attrs["FILLVAL"] = -1e31


ATTITUDE_DATA = """Year	DOY	Secofday	Flag	RTN_r		RTN_t		RTN_n		J2GCI_x		J2GCI_y		J2GCI_z		GSE_x		GSE_y		GSE_z
1999	158	   82800	0	-0.99597	-0.04735	-0.07622	 0.17406	 0.92866	 0.32756	 0.99608	 0.05545	-0.06887
1999	159	       0	0	-0.99600	-0.04670	-0.07618	 0.17402	 0.92868	 0.32753	 0.99612	 0.05480	-0.06891
1999	159	    3600	0	-0.99603	-0.04605	-0.07613	 0.17397	 0.92870	 0.32750	 0.99615	 0.05415	-0.06894
1999	159	    7200	0	-0.99607	-0.04541	-0.07608	 0.17393	 0.92871	 0.32747	 0.99618	 0.05351	-0.06897
1999	159	   10800	0	-0.99610	-0.04476	-0.07603	 0.17389	 0.92873	 0.32744	 0.99621	 0.05286	-0.06901
1999	159	   14400	0	-0.99613	-0.04412	-0.07598	 0.17384	 0.92875	 0.32741	 0.99625	 0.05221	-0.06904
1999	159	   18000	0	-0.99616	-0.04347	-0.07593	 0.17380	 0.92877	 0.32738	 0.99628	 0.05156	-0.06908
1999	159	   21600	0	-0.99620	-0.04283	-0.07588	 0.17375	 0.92879	 0.32735	 0.99631	 0.05091	-0.06911
1999	159	   25200	0	-0.99623	-0.04218	-0.07584	 0.17371	 0.92881	 0.32733	 0.99634	 0.05026	-0.06915
1999	159	   28800	0	-0.99626	-0.04153	-0.07579	 0.17366	 0.92883	 0.32730	 0.99637	 0.04961	-0.06918
1999	159	   32400	0	-0.99629	-0.04089	-0.07574	 0.17362	 0.92884	 0.32727	 0.99640	 0.04896	-0.06921
1999	159	   36000	0	-0.99632	-0.04024	-0.07569	 0.17358	 0.92886	 0.32724	 0.99643	 0.04831	-0.06925
1999	159	   39600	0	-0.99635	-0.03960	-0.07564	 0.17353	 0.92888	 0.32721	 0.99646	 0.04767	-0.06928
1999	159	   43200	0	-0.99638	-0.03895	-0.07559	 0.17349	 0.92890	 0.32718	 0.99649	 0.04702	-0.06932
1999	159	   46800	0	-0.99641	-0.03830	-0.07554	 0.17344	 0.92892	 0.32715	 0.99651	 0.04637	-0.06935
1999	159	   50400	0	-0.99643	-0.03766	-0.07549	 0.17340	 0.92894	 0.32712	 0.99654	 0.04572	-0.06938
1999	159	   54000	0	-0.99646	-0.03701	-0.07545	 0.17335	 0.92896	 0.32709	 0.99657	 0.04507	-0.06942
1999	159	   57600	0	-0.99649	-0.03633	-0.07545	 0.17334	 0.92899	 0.32701	 0.99659	 0.04439	-0.06951
1999	159	   61200	1	-0.98889	-0.12246	-0.08431	 0.08652	 0.93877	 0.33350	 0.98909	 0.13096	-0.06744
1999	159	   64800	0	-0.98034	-0.17591	-0.08939	 0.03159	 0.94119	 0.33639	 0.98061	 0.18463	-0.06575
1999	159	   68400	0	-0.98042	-0.17563	-0.08907	 0.03123	 0.94111	 0.33667	 0.98069	 0.18431	-0.06546
1999	159	   72000	0	-0.98050	-0.17535	-0.08874	 0.03086	 0.94102	 0.33694	 0.98076	 0.18399	-0.06518
1999	159	   75600	0	-0.98058	-0.17507	-0.08842	 0.03050	 0.94093	 0.33722	 0.98084	 0.18367	-0.06489
1999	159	   79200	0	-0.98066	-0.17479	-0.08809	 0.03013	 0.94085	 0.33749	 0.98092	 0.18335	-0.06461
1999	159	   82800	0	-0.98074	-0.17451	-0.08777	 0.02977	 0.94076	 0.33776	 0.98100	 0.18303	-0.06432
1999	160	       0	0	-0.98081	-0.17423	-0.08744	 0.02941	 0.94067	 0.33804	 0.98108	 0.18271	-0.06403
1999	160	    3600	0	-0.98089	-0.17395	-0.08711	 0.02904	 0.94059	 0.33831	 0.98116	 0.18238	-0.06375"""


def create_fake_mag_l1d_cdf(l2_swepam_electron_file_path: str,
                            output_mag_file_path: str):
    buffer = io.StringIO(ATTITUDE_DATA)
    data = np.loadtxt(buffer, skiprows=1)

    def time_to_float(ts):
        return [(t - datetime(1999, 1, 1)).total_seconds() for t in ts]

    times = [datetime(1998, 12, 31) + timedelta(days=row[1], seconds=row[2]) for row in data]

    l2_swepam_electron_file_path = HDF(l2_swepam_electron_file_path)
    vs_electron = l2_swepam_electron_file_path.vstart()
    vd_electron = vs_electron.attach("swepam_e")

    mag_epochs = get_epochs_from_output_file(vd_electron)

    correction_factor = (datetime(2025, 6, 30) - datetime(1999, 6, 8))
    mag_epochs_uncorrected = mag_epochs - correction_factor
    r_att = np.interp(time_to_float(mag_epochs_uncorrected), time_to_float(times), data[:, 4])
    t_att = np.interp(time_to_float(mag_epochs_uncorrected), time_to_float(times), data[:, 5])
    n_att = np.interp(time_to_float(mag_epochs_uncorrected), time_to_float(times), data[:, 6])

    rtn_att = np.stack((r_att, t_att, n_att), axis=-1)

    b_rtn_index = vd_electron.field("b_rtn")._index
    b_rtn_data = np.array([x[b_rtn_index] for x in vd_electron[:]])
    mag_despun_vectors = []
    for i in range(len(b_rtn_data)):
        mag_despun_vectors.append(convert_rtn_to_instrument(b_rtn_data[i], rtn_att[i]))

    with CDF(output_mag_file_path, readonly=False, create=True) as cdf:
        cdf.compress(pycdf.const.GZIP_COMPRESSION)
        cdf["epoch"] = get_epochs_from_output_file(vd_electron)
        cdf["vectors"] = mag_despun_vectors
        cdf["vectors"].attrs["FILLVAL"] = -1e31


def convert_rtn_to_instrument(input_vector, att_vector):
    r = att_vector[0]
    t = att_vector[1]
    n = att_vector[2]

    r2 = r * r
    t2 = t * t
    n2 = n * n

    c3 = np.sqrt(r2 + t2 + n2)
    c2 = np.sqrt(t2 + n2)
    c1 = np.sqrt((t2 + n2) * (t2 + n2) + r2 * t2 + r2 * n2)

    mat = np.array([[(-n2 - t2) / c1, 0, r / c3], [r * t / c1, -n / c2, t / c3], [r * n / c1, t / c2, n / c3]])

    return np.linalg.inv(mat) @ input_vector


def create_variable(cdf: CDF, variable_name: str, new_values: np.ndarray):
    cdf[variable_name] = new_values
    cdf[variable_name].attrs["FILLVAL"] = -1e31


def get_epochs_from_output_file(dataset: VD) -> np.array:
    years_index = dataset.field("year")._index
    month_index = dataset.field("mon")._index
    day_index = dataset.field("mday")._index
    hour_index = dataset.field("hour")._index
    min_index = dataset.field("min")._index
    sec_index = dataset.field("sec")._index

    correction_factor = (datetime(2025, 6, 30) - datetime(1999, 6, 8))

    return np.array([datetime(year=x[years_index], month=x[month_index], day=x[day_index], hour=x[hour_index],
                              minute=x[min_index], second=x[sec_index]) + correction_factor for x in dataset[:]])


def create_expected_cdf(dataset: VD, fields: list[str], file_name):
    with pycdf.CDF(str(path.parent.parent.parent / f'temp_cdf_data/expected_{file_name}.cdf'), masterpath="") as file:
        for field in fields:
            index = dataset.field(field)._index
            data = np.array([x[index] for x in dataset[:]])
            file.new(field, data)


if __name__ == "__main__":
    path = Path(__file__)
    l1_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/ACE_LV1_1999-159.swepam.hdf"
    l2_electron_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf"
    l2_ion_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/swepam-swi-1999-159.v2-01.hdf"

    l2_swe_cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swe_l2_sci-with-ace-data_20250630_v002.cdf"
    l1b_swe_cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swe_l1b_sci-with-ace-data_20250630_v002.cdf"
    mag_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_mag_l1d_mago-normal_20250630_v001.cdf"
    swapi_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swapi_l3a_proton-sw_20250630_v001.cdf"

    mag_file_path.unlink(missing_ok=True)
    swapi_file_path.unlink(missing_ok=True)
    l1b_swe_cdf_file_path.unlink(missing_ok=True)
    l2_swe_cdf_file_path.unlink(missing_ok=True)
    create_fake_mag_l1d_cdf(str(l2_electron_hdf_path), str(mag_file_path))
    create_fake_swapi_l3a_cdf(str(l2_ion_hdf_path), str(swapi_file_path))
