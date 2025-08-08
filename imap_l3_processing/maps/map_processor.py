from abc import abstractmethod
from pathlib import Path

from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.processor import Processor


class MapProcessor(Processor):

    @abstractmethod
    def process(self, spice_reference_frame: SpiceFrame = SpiceFrame.ECLIPJ2000) -> list[Path]:
        pass


