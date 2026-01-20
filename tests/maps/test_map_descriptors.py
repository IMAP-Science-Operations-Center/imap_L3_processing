import unittest

from imap_l3_processing.maps.map_descriptors import ReferenceFrame, SurvivalCorrection, MapDescriptorParts, Sensor, \
    MapQuantity, SpinPhase, PixelSize, parse_map_descriptor, map_descriptor_parts_to_string


class TestMapDescriptors(unittest.TestCase):

    def test_parse_map_descriptor(self):
        cg = ReferenceFrame.Heliospheric
        no_cg = ReferenceFrame.Spacecraft
        sp = SurvivalCorrection.SurvivalCorrected
        no_sp = SurvivalCorrection.NotSurvivalCorrected

        test_cases = [
            ("h45-ena-h-hf-sp-ram-hae-4deg-3mo",
             MapDescriptorParts(sensor=Sensor.Hi45, quantity=MapQuantity.Intensity, quantity_suffix="",
                                survival_correction=sp, reference_frame=cg,
                                spin_phase=SpinPhase.RamOnly, coord="hae",
                                duration="3mo",
                                grid=PixelSize.FourDegrees)),
            ("h45-ena-h-sf-sp-anti-hae-4deg-3mo",
             MapDescriptorParts(Sensor.Hi45, no_cg, sp, SpinPhase.AntiRamOnly, "hae",
                                PixelSize.FourDegrees, "3mo",
                                MapQuantity.Intensity, "")),
            ("h90-ena-h-hf-nsp-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.RamOnly, "hae",
                                                                     PixelSize.SixDegrees, "1yr",
                                                                     MapQuantity.Intensity, "")),
            ("h90-ena-h-hf-nsp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.FullSpin, "hae",
                                                                      PixelSize.SixDegrees, "6mo",
                                                                      MapQuantity.Intensity, "")),
            ("h45-spx-h-hf-sp-full-hae-4deg-6mo", MapDescriptorParts(Sensor.Hi45, cg, sp, SpinPhase.FullSpin, "hae",
                                                                     PixelSize.FourDegrees, "6mo",
                                                                     MapQuantity.SpectralIndex, "")),
            ("hic-ena-h-hf-nsp-full-hae-6deg-6mo",
             MapDescriptorParts(Sensor.HiCombined, cg, no_sp, SpinPhase.FullSpin, "hae",
                                PixelSize.SixDegrees, "6mo",
                                MapQuantity.Intensity, "")),
            (
                "l090-ena-h-hf-nsp-full-hae-6deg-6mo",
                MapDescriptorParts(Sensor.Lo90, cg, no_sp, SpinPhase.FullSpin, "hae",
                                   PixelSize.SixDegrees, "6mo",
                                   MapQuantity.Intensity, "")),
            ("l090-ena-h-hk-sp-ram-hae-4deg-1yr", MapDescriptorParts(Sensor.Lo90, ReferenceFrame.HeliosphericKinematic,
                                                                     sp, SpinPhase.RamOnly, "hae",
                                                                     PixelSize.FourDegrees, "1yr",
                                                                     MapQuantity.Intensity, "")),
            ("u90-spx-h-sf-sp-full-hae-nside8-6mo", MapDescriptorParts(Sensor.Ultra90, ReferenceFrame.Spacecraft,
                                                                       sp, SpinPhase.FullSpin, "hae", PixelSize.Nside8,
                                                                       "6mo", MapQuantity.SpectralIndex, "")),
            ("u45-spx-h-sf-sp-full-hae-2deg-6mo", MapDescriptorParts(Sensor.Ultra45, ReferenceFrame.Spacecraft,
                                                                     sp, SpinPhase.FullSpin, "hae",
                                                                     PixelSize.TwoDegrees,
                                                                     "6mo", MapQuantity.SpectralIndex, "")),
            ("ulc-spx-h-sf-sp-full-hae-nside16-6mo", MapDescriptorParts(Sensor.UltraCombined, ReferenceFrame.Spacecraft,
                                                                        sp, SpinPhase.FullSpin, "hae",
                                                                        PixelSize.Nside16,
                                                                        "6mo", MapQuantity.SpectralIndex, "")),
            ("u45-spx-h-sf-sp-full-hae-nside16-6mo", MapDescriptorParts(Sensor.Ultra45, ReferenceFrame.Spacecraft,
                                                                        sp, SpinPhase.FullSpin, "hae",
                                                                        PixelSize.Nside16,
                                                                        "6mo", MapQuantity.SpectralIndex, "")),
            ("u45-spx-h-sf-sp-full-custom-nside16-6mo", MapDescriptorParts(Sensor.Ultra45, ReferenceFrame.Spacecraft,
                                                                           sp, SpinPhase.FullSpin, "custom",
                                                                           PixelSize.Nside16,
                                                                           "6mo", MapQuantity.SpectralIndex, "")),
            ("ilo-ena-h-sf-sp-ram-imaphae-2deg-3mo", MapDescriptorParts(Sensor.Lo, ReferenceFrame.Spacecraft,
                                                                        sp, SpinPhase.RamOnly, "imaphae",
                                                                        PixelSize.TwoDegrees,
                                                                        "3mo", MapQuantity.Intensity, "")),
            ("not-valid-at-all", None),
            ("invalid_prefix-hic-ena-h-hf-nsp-full-hae-6deg-6mo", None),
            ("hic-ena-h-hf-nsp-full-hae-6deg-6mo-invalid-suffix", None),
            ("hic-ena-h-hf-nsp-full-imap?-6deg-6mo-invalid-suffix", None),
            ("hic-ena-h-hf-nsp-full-imap_v1-6deg-6mo-invalid-suffix", None),
            ("hic-spxwItHcUsToMsUfFiX-h-sf-sp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.HiCombined,
                                                                                     ReferenceFrame.Spacecraft, sp,
                                                                                     SpinPhase.FullSpin, "hae",
                                                                                     PixelSize.SixDegrees,
                                                                                     "6mo",
                                                                                     MapQuantity.SpectralIndex,
                                                                                     "wItHcUsToMsUfFiX")),
            ("hic-enaCUSTOM-h-sf-sp-full-CUSTOM1-6deg-6mo", MapDescriptorParts(Sensor.HiCombined,
                                                                               ReferenceFrame.Spacecraft, sp,
                                                                               SpinPhase.FullSpin, "CUSTOM1",
                                                                               PixelSize.SixDegrees,
                                                                               "6mo",
                                                                               MapQuantity.Intensity, "CUSTOM")),
            ("hic-enaCUSTOM-h-sf-sp-full-imapeclipj2000-6deg-6mo", MapDescriptorParts(Sensor.HiCombined,
                                                                                      ReferenceFrame.Spacecraft, sp,
                                                                                      SpinPhase.FullSpin,
                                                                                      "imapeclipj2000",
                                                                                      PixelSize.SixDegrees,
                                                                                      "6mo",
                                                                                      MapQuantity.Intensity, "CUSTOM")),
            ("hic-enaCUSTOM-h-sf-sp-full-hae-6deg-0mo", MapDescriptorParts(Sensor.HiCombined,
                                                                           ReferenceFrame.Spacecraft, sp,
                                                                           SpinPhase.FullSpin, "hae",
                                                                           PixelSize.SixDegrees,
                                                                           "0mo",
                                                                           MapQuantity.Intensity, "CUSTOM")),
            ("hic-enaCUSTOM-h-sf-sp-full-hae-6deg-50yr", MapDescriptorParts(Sensor.HiCombined,
                                                                            ReferenceFrame.Spacecraft, sp,
                                                                            SpinPhase.FullSpin, "hae",
                                                                            PixelSize.SixDegrees,
                                                                            "50yr",
                                                                            MapQuantity.Intensity, "CUSTOM")),
            ("l090-isn-h-sf-nsp-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Lo90, ReferenceFrame.Spacecraft,
                                                                      no_sp, SpinPhase.RamOnly, "hae",
                                                                      PixelSize.SixDegrees,
                                                                      "1yr",
                                                                      MapQuantity.ISNBackgroundSubtracted, "")),
            ("l090-spxnbs-h-sf-nsp-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Lo90, ReferenceFrame.Spacecraft,
                                                                      no_sp, SpinPhase.RamOnly, "hae",
                                                                      PixelSize.SixDegrees,
                                                                      "1yr",
                                                                      MapQuantity.SpectralIndex, "nbs")),
            ("l090-spxnbs-h-hk-nsp-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Lo90, ReferenceFrame.HeliosphericKinematic,
                                                                         no_sp, SpinPhase.RamOnly, "hae",
                                                                         PixelSize.SixDegrees,
                                                                         "1yr",
                                                                         MapQuantity.SpectralIndex, "nbs")),
        ]

        for descriptor, expected in test_cases:
            with self.subTest(descriptor):
                descriptor_parts = parse_map_descriptor(descriptor)
                self.assertEqual(expected, descriptor_parts)

                if descriptor_parts is not None:
                    self.assertEqual(descriptor, map_descriptor_parts_to_string(descriptor_parts))


if __name__ == '__main__':
    unittest.main()
