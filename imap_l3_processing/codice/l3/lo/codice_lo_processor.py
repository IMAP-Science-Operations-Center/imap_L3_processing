import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aDataProduct
from imap_l3_processing.codice.l3.lo.science.esa_calculations import calculate_partial_densities
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class CodiceLoProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        dependencies = CodiceLoL3aDependencies.fetch_dependencies(self.dependencies)
        l3a_data = self.process_l3a(dependencies)
        saved_cdf = save_data(l3a_data)
        upload(saved_cdf)

    def process_l3a(self, dependencies: CodiceLoL3aDependencies):
        species_index = 1
        for species_name, species_intensities in dependencies.codice_l2_lo_data.get_species_intensities().items():
            partial_density = calculate_partial_densities(species_intensities)
            match species_name:
                case "H+":
                    h_partial_density = partial_density
                case "He++":
                    he_partial_density = partial_density
                case "C+4":
                    c4_partial_density = partial_density
                case "C+5":
                    c5_partial_density = partial_density
                case "C+6":
                    c6_partial_density = partial_density
                case "O+5":
                    o5_partial_density = partial_density
                case "O+6":
                    o6_partial_density = partial_density
                case "O+7":
                    o7_partial_density = partial_density
                case "O+8":
                    o8_partial_density = partial_density
                case "Mg":
                    mg_partial_density = partial_density
                case "Si":
                    si_partial_density = partial_density
                case "Fe (low Q)":
                    fe_low_partial_density = partial_density
                case "Fe (high Q)":
                    fe_high_partial_density = partial_density
                case _:
                    raise NotImplementedError
        epoch = np.array([np.nan])
        epoch_delta = np.full(len(epoch), 4.8e+11)
        return CodiceLoL3aDataProduct(epoch=epoch, epoch_delta=epoch_delta, h_partial_density=h_partial_density,
                                      he_partial_density=he_partial_density, c4_partial_density=c4_partial_density,
                                      c5_partial_density=c5_partial_density, c6_partial_density=c6_partial_density,
                                      o5_partial_density=o5_partial_density, o6_partial_density=o6_partial_density,
                                      o7_partial_density=o7_partial_density, o8_partial_density=o8_partial_density,
                                      mg_partial_density=mg_partial_density, si_partial_density=si_partial_density,
                                      fe_low_partial_density=fe_low_partial_density,
                                      fe_high_partial_density=fe_high_partial_density)
