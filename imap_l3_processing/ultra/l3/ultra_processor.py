from dataclasses import dataclass

from imap_data_access import upload
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, MapQuantity, SurvivalCorrection, \
    parse_map_descriptor, PixelSize
from imap_l3_processing.maps.map_models import HealPixIntensityDataProduct, HealPixIntensityMapData, IntensityMapData, \
    HealPixCoords, HealPixSpectralIndexDataProduct, HealPixSpectralIndexMapData, RectangularIntensityDataProduct, \
    RectangularIntensityMapData
from imap_l3_processing.maps.spectral_fit import calculate_spectral_index_for_multiple_ranges
from imap_l3_processing.processor import Processor
from imap_l3_processing.ultra.l3.science.ultra_survival_probability import UltraSurvivalProbabilitySkyMap, \
    UltraSurvivalProbability
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies, UltraL3SpectralIndexDependencies
from imap_l3_processing.ultra.l3.ultra_l3_to_rectangular_dependencies import UltraL3ToRectangularDependencies
from imap_l3_processing.utils import save_data, combine_glows_l3e_with_l1c_pointing


class UltraProcessor(Processor):
    def process(self):
        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        parent_file_names = self.get_parent_file_names()

        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                ultra_l3_spectral_fit_dependencies = UltraL3SpectralIndexDependencies.fetch_dependencies(
                    self.dependencies)
                spectral_index_data_product = self._process_spectral_index(ultra_l3_spectral_fit_dependencies)
                spectral_index_data_product.parent_file_names = parent_file_names
                data_product_path = save_data(spectral_index_data_product)
                upload(data_product_path)
            case MapDescriptorParts(grid=PixelSize.FourDegrees | PixelSize.SixDegrees):
                deps = UltraL3ToRectangularDependencies.fetch_dependencies(self.dependencies)
                data_product = self._process_healpix_to_rectangular(deps, parsed_descriptor.grid)
                data_product.parent_file_names = parent_file_names
                data_product.add_paths_to_parents(deps.dependency_file_paths)
                data_product_path = save_data(data_product)
                upload(data_product_path)
                pass
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected):
                deps = UltraL3Dependencies.fetch_dependencies(self.dependencies)
                data_product = self._process_survival_probability(deps)
                data_product.parent_file_names = parent_file_names
                data_product.add_paths_to_parents(deps.dependency_file_paths)
                data_product_path = save_data(data_product)
                upload(data_product_path)
            case _:
                raise NotImplementedError

    def _process_survival_probability(self, deps: UltraL3Dependencies) -> HealPixIntensityDataProduct:
        combined_psets = combine_glows_l3e_with_l1c_pointing(deps.glows_l3e_sp, deps.ultra_l1c_pset, )
        survival_probability_psets = [UltraSurvivalProbability(_l1c, _l3e) for _l1c, _l3e in
                                      combined_psets]

        intensity_data = deps.ultra_l2_map.intensity_map_data
        coords = deps.ultra_l2_map.coords
        corrected_skymap = UltraSurvivalProbabilitySkyMap(survival_probability_psets, geometry.SpiceFrame.ECLIPJ2000,
                                                          coords.nside)
        survival_probability_map = corrected_skymap.to_dataset()["exposure_weighted_survival_probabilities"].values

        corrected_skymap.data_1d = corrected_skymap.data_1d.assign({
            "corrected_ena_intensity": ([CoordNames.TIME, CoordNames.ENERGY_ULTRA, CoordNames.GENERIC_PIXEL],
                                        intensity_data.ena_intensity / survival_probability_map),
            "corrected_ena_intensity_stat_unc": ([CoordNames.TIME, CoordNames.ENERGY_ULTRA, CoordNames.GENERIC_PIXEL],
                                                 intensity_data.ena_intensity_stat_unc / survival_probability_map),
            "corrected_ena_intensity_sys_err": ([CoordNames.TIME, CoordNames.ENERGY_ULTRA, CoordNames.GENERIC_PIXEL],
                                                intensity_data.ena_intensity_sys_err / survival_probability_map),
        })

        rectangular_corrected_skymap = corrected_skymap.to_rectangular_skymap(4, [
            "corrected_ena_intensity",
            "corrected_ena_intensity_stat_unc",
            "corrected_ena_intensity_sys_err",
        ])

        corrected_intensity = intensity_data.ena_intensity / survival_probability_map
        corrected_stat_unc = intensity_data.ena_intensity_stat_unc / survival_probability_map
        corrected_sys_unc = intensity_data.ena_intensity_sys_err / survival_probability_map

        return HealPixIntensityDataProduct(
            input_metadata=self.input_metadata.to_upstream_data_dependency(self.input_metadata.descriptor),
            data=HealPixIntensityMapData(
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
                ),
            )
        )

    def _process_spectral_index(self,
                                dependencies: UltraL3SpectralIndexDependencies) -> HealPixSpectralIndexDataProduct:
        map_data = calculate_spectral_index_for_multiple_ranges(
            dependencies.map_data.intensity_map_data,
            dependencies.get_fit_energy_ranges()
        )
        return HealPixSpectralIndexDataProduct(
            data=HealPixSpectralIndexMapData(
                spectral_index_map_data=map_data,
                coords=dependencies.map_data.coords
            ),
            input_metadata=self.input_metadata,
        )

    def _process_healpix_to_rectangular(self, dependencies: UltraL3ToRectangularDependencies,
                                        spacing_deg: int) -> RectangularIntensityDataProduct:
        variables_to_convert_to_rectangular = [
            "exposure_factor",
            "ena_intensity",
            "ena_intensity_stat_unc",
            "ena_intensity_sys_err",
        ]

        healpix_map = dependencies.healpix_map_data.to_healpix_skymap()

        rectangular_map, _ = healpix_map.to_rectangular_skymap(spacing_deg, variables_to_convert_to_rectangular)
        rectangular_map_xarray_dataset = rectangular_map.to_dataset()

        input_map_intensity_data = dependencies.healpix_map_data.intensity_map_data
        intensity_map_data = IntensityMapData(
            epoch=input_map_intensity_data.epoch,
            epoch_delta=input_map_intensity_data.epoch_delta,
            energy=input_map_intensity_data.energy,
            energy_delta_plus=input_map_intensity_data.energy_delta_plus,
            energy_delta_minus=input_map_intensity_data.energy_delta_minus,
            energy_label=input_map_intensity_data.energy_label,
            latitude=None,
            longitude=None,
            obs_date=None,
            obs_date_range=None,
            solid_angle=None,
            exposure_factor=rectangular_map_xarray_dataset["exposure_factor"].values,
            ena_intensity=rectangular_map_xarray_dataset["ena_intensity"].values,
            ena_intensity_stat_unc=rectangular_map_xarray_dataset["ena_intensity_stat_unc"].values,
            ena_intensity_sys_err=rectangular_map_xarray_dataset["ena_intensity_sys_err"].values,
        )
        rect_intensity_map_data = RectangularIntensityMapData(intensity_map_data, None)

        return RectangularIntensityDataProduct(data=rect_intensity_map_data, input_metadata=self.input_metadata)


@dataclass
class UltraMapDescriptorParts:
    grid_size: int
