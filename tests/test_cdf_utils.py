import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import write_cdf
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata, FrameAttribute
from imap_l3_processing.spice_wrapper import spiceypy


class TestCDFUtils(unittest.TestCase):

    def test_write_cdf_transforms_frame_velocity(self):
        instrument_metadata = InputMetadata(
            instrument="swe",
            data_level="l3",
            version="v001",
            descriptor="sci",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 1),
        )

        position_vectors = [[0, 0, 0], [1, 0, 0], [0, 0, 50000]]
        epochs = np.arange(0, len(position_vectors)) * timedelta(days=0) + datetime(2025, 8, 20)

        ets = spiceypy.datetime2et(epochs)
        matrices = spiceypy.sxform("IMAP_RTN", "ECLIPJ2000", ets)
        states = np.zeros((len(position_vectors), 6))
        states[:, 0:3] = position_vectors
        transformed = np.matmul(matrices, states[:, :, np.newaxis])[:, :, 0]
        print(transformed)
        imap_state, _ = spiceypy.spkezr("IMAP", ets, "ECLIPJ2000", "NONE", "SUN")
        out = transformed + imap_state
        print(out[:, 0:3])

        fake_product = SimpleDataProduct(instrument_metadata, "core_velocity_vector_rtn_integrated", vectors, epochs)
        attribute_manager = ImapAttributeManager()
        attribute_manager.add_instrument_attrs("swe", "l3", "")
        attribute_manager.add_global_attribute("Data_version", "001")
        attribute_manager.add_global_attribute("Logical_file_id", "logicalsrc")
        attribute_manager.add_global_attribute("Logical_source", "desc")

        to_frame = FrameAttribute("ECLIPJ2000", "SUN", "INTERTAL")

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.cdf"
            write_cdf(str(out_path), fake_product, attribute_manager, to_frame)

            with CDF(str(out_path)) as cdf:
                velocity_var = cdf["core_velocity_vector_rtn_integrated"]


@dataclass
class SimpleDataProduct(DataProduct):
    name: str
    data: np.ndarray
    epoch: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(name=self.name,
                                value=self.data),
            DataProductVariable(name="epoch", value=self.epoch)
        ]
