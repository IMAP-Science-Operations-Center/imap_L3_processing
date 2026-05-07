"""Pin SWAPI descriptor strings.

Descriptors are used as `source/descriptor` keys against the SDC processing
input collection — typos there silently cause `get_file_paths` to return
empty lists, which downstream surfaces as `IndexError` in `fetch_dependencies`.
These tests catch accidental rename/typo regressions cheaply.
"""

import unittest

from imap_l3_processing.swapi import descriptors


class TestDescriptors(unittest.TestCase):
    def test_known_descriptor_strings(self):
        # Pin the wire-format strings: changing these breaks SDC dependency lookup.
        self.assertEqual(descriptors.SWAPI_L2_DESCRIPTOR, "sci")
        self.assertEqual(descriptors.MAG_RTN_DESCRIPTOR, "norm-rtn")
        self.assertEqual(descriptors.SWAPI_L3A_ALPHA_SW_DESCRIPTOR, "alpha-sw")
        self.assertEqual(
            descriptors.GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR,
            "energy-gf-pui-lut",
        )
        self.assertEqual(
            descriptors.GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR, "energy-gf-sw-lut"
        )
        self.assertEqual(
            descriptors.EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, "efficiency-lut"
        )
        self.assertEqual(
            descriptors.ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR,
            "alpha-density-temperature-lut",
        )
        self.assertEqual(
            descriptors.INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR,
            "instrument-response-lut",
        )
        self.assertEqual(
            descriptors.DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR,
            "density-of-neutral-helium-lut",
        )
        self.assertEqual(
            descriptors.HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, "hydrogen-inflow-vector"
        )
        self.assertEqual(
            descriptors.HELIUM_INFLOW_VECTOR_DESCRIPTOR, "helium-inflow-vector"
        )
        self.assertEqual(
            descriptors.AZIMUTHAL_TRANSMISSION_DESCRIPTOR, "azimuthal-transmission"
        )
        self.assertEqual(
            descriptors.CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, "central-effective-area"
        )
        self.assertEqual(
            descriptors.PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR,
            "passband-fit-coefficients",
        )


if __name__ == "__main__":
    unittest.main()
