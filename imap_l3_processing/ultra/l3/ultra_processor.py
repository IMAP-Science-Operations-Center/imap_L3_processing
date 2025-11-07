from dataclasses import dataclass
from datetime import datetime, timedelta

from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, MapQuantity, SurvivalCorrection, \
    parse_map_descriptor, PixelSize, Sensor
from imap_l3_processing.maps.map_models import HealPixIntensityMapData, IntensityMapData, \
    HealPixCoords, HealPixSpectralIndexMapData, RectangularIntensityDataProduct, \
    RectangularIntensityMapData, RectangularCoords, RectangularSpectralIndexDataProduct, \
    RectangularSpectralIndexMapData, SpectralIndexMapData, combine_healpix_intensity_map_data
from imap_l3_processing.maps.map_processor import MapProcessor
from imap_l3_processing.maps.spectral_fit import calculate_spectral_index_for_multiple_ranges
from imap_l3_processing.ultra.l3.science.ultra_survival_probability import UltraSurvivalProbabilitySkyMap, \
    UltraSurvivalProbability
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies, UltraL3SpectralIndexDependencies, \
    UltraL3CombinedDependencies
from imap_l3_processing.utils import save_data, combine_glows_l3e_with_l1c_pointing


class UltraProcessor(MapProcessor):
    def process(self, spice_frame_name: SpiceFrame = SpiceFrame.ECLIPJ2000):
        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        parent_file_names = self.get_parent_file_names()

        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                ultra_l3_spectral_fit_dependencies = UltraL3SpectralIndexDependencies.fetch_dependencies(
                    self.dependencies)
                healpix_spectral_index_map_data = self._process_spectral_index(ultra_l3_spectral_fit_dependencies)
                data_product = RectangularSpectralIndexDataProduct(input_metadata=self.input_metadata,
                                                                   data=healpix_spectral_index_map_data)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Ultra45 | Sensor.Ultra90,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                deps = UltraL3Dependencies.fetch_dependencies(self.dependencies)
                healpix_intensity_map_data = self._process_survival_probability(deps, spice_frame_name)
                data_product = self._process_healpix_intensity_to_rectangular(healpix_intensity_map_data,
                                                                              parsed_descriptor.grid)
                data_product.add_paths_to_parents(deps.dependency_file_paths)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.UltraCombined,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):

                combined_deps = UltraL3CombinedDependencies.fetch_dependencies(self.dependencies)
                combined_data = self._process_combined_survival_probability(combined_deps, spice_frame_name)

                data_product = self._process_healpix_intensity_to_rectangular(combined_data, parsed_descriptor.grid)
                data_product.add_paths_to_parents(combined_deps.dependency_file_paths)

            case MapDescriptorParts(sensor=Sensor.UltraCombined,
                                    survival_correction=SurvivalCorrection.NotSurvivalCorrected,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                deps = UltraL3CombinedDependencies.fetch_dependencies(self.dependencies)
                healpix_intensity_map_data = combine_healpix_intensity_map_data([deps.u45_l2_map, deps.u90_l2_map])
                data_product = self._process_healpix_intensity_to_rectangular(healpix_intensity_map_data,
                                                                              parsed_descriptor.grid)
                data_product.add_paths_to_parents(deps.dependency_file_paths)
            case _:
                raise NotImplementedError

        data_product.add_filenames_to_parents(parent_file_names)
        return [save_data(data_product)]

    def _process_combined_survival_probability(self, deps: UltraL3CombinedDependencies, spice_frame_name: SpiceFrame):
        u45_dep = UltraL3Dependencies(
            ultra_l2_map=deps.u45_l2_map,
            ultra_l1c_pset=deps.u45_l1c_psets,
            glows_l3e_sp=deps.glows_l3e_psets,
            dependency_file_paths=deps.dependency_file_paths
        )

        u90_dep = UltraL3Dependencies(
            ultra_l2_map=deps.u90_l2_map,
            ultra_l1c_pset=deps.u90_l1c_psets,
            glows_l3e_sp=deps.glows_l3e_psets,
            dependency_file_paths=deps.dependency_file_paths
        )

        u45_survival_corrected = self._process_survival_probability(u45_dep, spice_frame_name)
        u90_survival_corrected = self._process_survival_probability(u90_dep, spice_frame_name)

        return combine_healpix_intensity_map_data([u45_survival_corrected, u90_survival_corrected])

    def _process_survival_probability(self, deps: UltraL3Dependencies, spice_frame_name: SpiceFrame) -> HealPixIntensityMapData:
        combined_psets = combine_glows_l3e_with_l1c_pointing(deps.glows_l3e_sp, deps.ultra_l1c_pset, )
        survival_probability_psets = [UltraSurvivalProbability(_l1c, _l3e) for _l1c, _l3e in
                                      combined_psets]

        intensity_data = deps.ultra_l2_map.intensity_map_data
        coords = deps.ultra_l2_map.coords
        corrected_skymap = UltraSurvivalProbabilitySkyMap(survival_probability_psets, spice_frame_name, coords.nside)
        survival_probability_map = corrected_skymap.to_dataset()["exposure_weighted_survival_probabilities"].values

        corrected_intensity = intensity_data.ena_intensity / survival_probability_map
        corrected_stat_unc = intensity_data.ena_intensity_stat_unc / survival_probability_map
        corrected_sys_unc = intensity_data.ena_intensity_sys_err / survival_probability_map

        healpix_map_data = HealPixIntensityMapData(
            intensity_map_data=IntensityMapData(
                ena_intensity_stat_unc=corrected_stat_unc,
                ena_intensity_sys_err=corrected_sys_unc,
                ena_intensity=corrected_intensity,
                epoch=intensity_data.epoch,
                epoch_delta=intensity_data.epoch_delta,
                energy=intensity_data.energy,
                energy_delta_plus=intensity_data.energy_delta_plus,
                energy_delta_minus=intensity_data.energy_delta_minus,
                energy_label=intensity_data.energy_label,
                latitude=intensity_data.latitude,
                longitude=intensity_data.longitude,
                exposure_factor=intensity_data.exposure_factor,
                obs_date=intensity_data.obs_date,
                obs_date_range=intensity_data.obs_date_range,
                solid_angle=intensity_data.solid_angle,
            ),
            coords=HealPixCoords(
                pixel_index=coords.pixel_index,
                pixel_index_label=coords.pixel_index_label,
            )
        )
        return healpix_map_data

    def _process_spectral_index(self,
                                dependencies: UltraL3SpectralIndexDependencies) -> RectangularSpectralIndexMapData:
        map_data = calculate_spectral_index_for_multiple_ranges(
            dependencies.map_data.intensity_map_data,
            dependencies.get_fit_energy_ranges()
        )
        return RectangularSpectralIndexMapData(
            spectral_index_map_data=map_data,
            coords=dependencies.map_data.coords
        )

    def _process_healpix_intensity_to_rectangular(self, healpix_map_data: HealPixIntensityMapData,
                                                  spacing_deg: int) -> RectangularIntensityDataProduct:
        variables_to_convert_to_rectangular = [
            "exposure_factor",
            "ena_intensity",
            "ena_intensity_stat_unc",
            "ena_intensity_sys_err",
            "obs_date",
            "obs_date_range",
        ]

        healpix_map = healpix_map_data.to_healpix_skymap()

        rectangular_map, _ = healpix_map.to_rectangular_skymap(spacing_deg, variables_to_convert_to_rectangular)
        rectangular_map_xarray_dataset = rectangular_map.to_dataset()

        obs_date = datetime(year=1970, month=1, day=1) + timedelta(seconds=1) * (
                rectangular_map_xarray_dataset["obs_date"].values / 1e9)

        input_map_intensity_data = healpix_map_data.intensity_map_data
        intensity_map_data = IntensityMapData(
            epoch=input_map_intensity_data.epoch,
            epoch_delta=input_map_intensity_data.epoch_delta,
            energy=input_map_intensity_data.energy,
            energy_delta_plus=input_map_intensity_data.energy_delta_plus,
            energy_delta_minus=input_map_intensity_data.energy_delta_minus,
            energy_label=input_map_intensity_data.energy_label,
            latitude=rectangular_map.sky_grid.el_bin_midpoints,
            longitude=rectangular_map.sky_grid.az_bin_midpoints,
            obs_date=obs_date,
            obs_date_range=rectangular_map_xarray_dataset["obs_date_range"].values,
            solid_angle=rectangular_map.solid_angle_grid.T,
            exposure_factor=rectangular_map_xarray_dataset["exposure_factor"].values,
            ena_intensity=rectangular_map_xarray_dataset["ena_intensity"].values,
            ena_intensity_stat_unc=rectangular_map_xarray_dataset["ena_intensity_stat_unc"].values,
            ena_intensity_sys_err=rectangular_map_xarray_dataset["ena_intensity_sys_err"].values,
        )
        rect_intensity_map_data = RectangularIntensityMapData(intensity_map_data, coords=RectangularCoords(
            latitude_delta=(rectangular_map.sky_grid.el_bin_midpoints - rectangular_map.sky_grid.el_bin_edges[:-1]),
            latitude_label=intensity_map_data.latitude.astype(str),
            longitude_delta=(rectangular_map.sky_grid.az_bin_midpoints - rectangular_map.sky_grid.az_bin_edges[
                                                                         :-1]),
            longitude_label=intensity_map_data.longitude.astype(str),
        ))

        return RectangularIntensityDataProduct(data=rect_intensity_map_data, input_metadata=self.input_metadata)

    def _process_healpix_spectral_index_to_rectangular(self, healpix_map_data: HealPixSpectralIndexMapData,
                                                       spacing_deg: int) -> RectangularSpectralIndexDataProduct:
        spectral_index_skymap = healpix_map_data.to_healpix_skymap()

        variables_to_project = [
            'exposure_factor',
            'ena_spectral_index',
            'ena_spectral_index_stat_unc',
            'obs_date',
            'obs_date_range',
        ]

        rectangular_skymap, _ = spectral_index_skymap.to_rectangular_skymap(spacing_deg, variables_to_project)

        rectangular_dataset = rectangular_skymap.to_dataset()

        obs_date = datetime(year=1970, month=1, day=1) + timedelta(seconds=1) * (
                rectangular_dataset["obs_date"].values / 1e9)

        latitude = rectangular_skymap.sky_grid.el_bin_midpoints
        longitude = rectangular_skymap.sky_grid.az_bin_midpoints
        latitude_deltas = latitude - rectangular_skymap.sky_grid.el_bin_edges[:-1]
        longitude_deltas = longitude - rectangular_skymap.sky_grid.az_bin_edges[:-1]

        healpix_spectral_index_map_data = healpix_map_data.spectral_index_map_data
        return RectangularSpectralIndexDataProduct(
            input_metadata=self.input_metadata,
            data=RectangularSpectralIndexMapData(
                spectral_index_map_data=SpectralIndexMapData(
                    ena_spectral_index=rectangular_dataset["ena_spectral_index"].values,
                    ena_spectral_index_stat_unc=rectangular_dataset["ena_spectral_index_stat_unc"].values,
                    epoch=healpix_spectral_index_map_data.epoch,
                    epoch_delta=healpix_spectral_index_map_data.epoch_delta,
                    energy=healpix_spectral_index_map_data.energy,
                    energy_delta_plus=healpix_spectral_index_map_data.energy_delta_plus,
                    energy_delta_minus=healpix_spectral_index_map_data.energy_delta_minus,
                    energy_label=healpix_spectral_index_map_data.energy_label,
                    latitude=latitude,
                    longitude=longitude,
                    exposure_factor=rectangular_dataset["exposure_factor"].values,
                    obs_date=obs_date,
                    obs_date_range=rectangular_dataset["obs_date_range"].values,
                    solid_angle=rectangular_skymap.solid_angle_grid.T,
                ),
                coords=RectangularCoords(
                    latitude_delta=latitude_deltas,
                    latitude_label=latitude.astype(str),
                    longitude_delta=longitude_deltas,
                    longitude_label=longitude.astype(str),
                ),
            )
        )


@dataclass
class UltraMapDescriptorParts:
    grid_size: int
