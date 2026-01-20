import dataclasses
import shutil
from datetime import datetime
from pathlib import Path

from imap_data_access import ScienceFilePath, ProcessingInputCollection, ScienceInput

from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, MapQuantity, map_descriptor_parts_to_string
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.ultra_processor import UltraProcessor


def run_spectral_index(input: Path):
    parsed = ScienceFilePath(input.name)

    processors = {"hi": HiProcessor, "lo": LoProcessor, "ultra": UltraProcessor}
    parsed_descriptor = parse_map_descriptor(parsed.descriptor)

    output_descriptor = map_descriptor_parts_to_string(
        dataclasses.replace(parsed_descriptor, quantity=MapQuantity.SpectralIndex)
    )

    dependencies = ProcessingInputCollection(ScienceInput(input.name))

    data_path = parsed.construct_path()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input, data_path)
    processor = processors[parsed.instrument](dependencies, InputMetadata(instrument=parsed.instrument,
                                   data_level="l3",
                                   start_date=datetime.strptime(parsed.start_date, "%Y%m%d"),
                                   end_date=None,
                                   version="v000",
                                   descriptor=output_descriptor,
                                   ))
    return processor.process()

if __name__ == "__main__":
    outputs = []
    for file in Path(r"C:\Users\Harrison\Downloads\hi_spectral_inputs").iterdir():
        outputs.extend(run_spectral_index(file))
    print(outputs)


