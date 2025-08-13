import unittest

from imap_l3_processing.maps.map_descriptors import ReferenceFrame, SurvivalCorrection, MapDescriptorParts, Sensor, \
    MapQuantity, SpinPhase, Duration, PixelSize, parse_map_descriptor


class TestMapDescriptors(unittest.TestCase):

    def test_parse_map_descriptor(self):
        cg = ReferenceFrame.Heliospheric
        no_cg = ReferenceFrame.Spacecraft
        sp = SurvivalCorrection.SurvivalCorrected
        no_sp = SurvivalCorrection.NotSurvivalCorrected

        test_cases = [
            ("h45-ena-h-hf-sp-ram-hae-4deg-3mo", MapDescriptorParts(sensor=Sensor.Hi45, quantity=MapQuantity.Intensity,
                                                                    survival_correction=sp, reference_frame=cg,
                                                                    spin_phase=SpinPhase.RamOnly,
                                                                    duration=Duration.ThreeMonths,
                                                                    grid=PixelSize.FourDegrees)),
            ("h45-ena-h-sf-sp-anti-hae-4deg-3mo", MapDescriptorParts(Sensor.Hi45, no_cg, sp, SpinPhase.AntiRamOnly,
                                                                     PixelSize.FourDegrees, Duration.ThreeMonths,
                                                                     MapQuantity.Intensity)),
            ("h90-ena-h-hf-nsp-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.RamOnly,
                                                                     PixelSize.SixDegrees, Duration.OneYear,
                                                                     MapQuantity.Intensity)),
            ("h90-ena-h-hf-nsp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.FullSpin,
                                                                      PixelSize.SixDegrees, Duration.SixMonths,
                                                                      MapQuantity.Intensity)),
            ("h45-spx-h-hf-sp-full-hae-4deg-6mo", MapDescriptorParts(Sensor.Hi45, cg, sp, SpinPhase.FullSpin,
                                                                     PixelSize.FourDegrees, Duration.SixMonths,
                                                                     MapQuantity.SpectralIndex)),
            ("hic-ena-h-hf-nsp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.HiCombined, cg, no_sp, SpinPhase.FullSpin,
                                                                      PixelSize.SixDegrees, Duration.SixMonths,
                                                                      MapQuantity.Intensity)),
            ("l090-ena-h-hf-nsp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.Lo90, cg, no_sp, SpinPhase.FullSpin,
                                                                       PixelSize.SixDegrees, Duration.SixMonths,
                                                                       MapQuantity.Intensity)),
            ("l090-ena-h-hk-sp-ram-hae-4deg-1yr", MapDescriptorParts(Sensor.Lo90, ReferenceFrame.HeliosphericKinematic,
                                                                     sp, SpinPhase.RamOnly,
                                                                     PixelSize.FourDegrees, Duration.OneYear,
                                                                     MapQuantity.Intensity)),
            ("u90-spx-h-sf-sp-full-hae-nside8-6mo", MapDescriptorParts(Sensor.Ultra90, ReferenceFrame.Spacecraft,
                                                                       sp, SpinPhase.FullSpin, PixelSize.Nside8,
                                                                       Duration.SixMonths, MapQuantity.SpectralIndex)),
            ("u45-spx-h-sf-sp-full-hae-2deg-6mo", MapDescriptorParts(Sensor.Ultra45, ReferenceFrame.Spacecraft,
                                                                     sp, SpinPhase.FullSpin, PixelSize.TwoDegrees,
                                                                     Duration.SixMonths, MapQuantity.SpectralIndex)),
            ("ulc-spx-h-sf-sp-full-hae-nside16-6mo", MapDescriptorParts(Sensor.UltraCombined, ReferenceFrame.Spacecraft,
                                                                        sp, SpinPhase.FullSpin, PixelSize.Nside16,
                                                                        Duration.SixMonths, MapQuantity.SpectralIndex)),
            ("u45-spx-h-sf-sp-full-hae-nside16-6mo", MapDescriptorParts(Sensor.Ultra45, ReferenceFrame.Spacecraft,
                                                                        sp, SpinPhase.FullSpin, PixelSize.Nside16,
                                                                        Duration.SixMonths, MapQuantity.SpectralIndex)),
            ("u45-spx-h-sf-sp-full-custom-nside16-6mo", MapDescriptorParts(Sensor.Ultra45, ReferenceFrame.Spacecraft,
                                                                        sp, SpinPhase.FullSpin, PixelSize.Nside16,
                                                                        Duration.SixMonths, MapQuantity.SpectralIndex)),
            ("not-valid-at-all", None),
            ("invalid_prefix-hic-ena-h-hf-nsp-full-hae-6deg-6mo", None),
            ("hic-ena-h-hf-nsp-full-hae-6deg-6mo-invalid-suffix", None),
            ("hic-ena-h-hf-nsp-full-imap?-6deg-6mo-invalid-suffix", None),
            ("hic-ena-h-hf-nsp-full-imap_v1-6deg-6mo-invalid-suffix", None),
            ("hic-spxwItHcUsToMsUfFiX-h-sf-sp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.HiCombined,
                                                                                     ReferenceFrame.Spacecraft, sp,
                                                                                     SpinPhase.FullSpin,
                                                                                     PixelSize.SixDegrees,
                                                                                     Duration.SixMonths,
                                                                                     MapQuantity.SpectralIndex)),
            ("hic-enaCUSTOM-h-sf-sp-full-CUSTOM1-6deg-6mo", MapDescriptorParts(Sensor.HiCombined,
                                                                              ReferenceFrame.Spacecraft, sp,
                                                                              SpinPhase.FullSpin,
                                                                              PixelSize.SixDegrees,
                                                                              Duration.SixMonths,
                                                                              MapQuantity.Intensity)),
            ("hic-enaCUSTOM-h-sf-sp-full-imapeclipj2000-6deg-6mo", MapDescriptorParts(Sensor.HiCombined,
                                                                              ReferenceFrame.Spacecraft, sp,
                                                                              SpinPhase.FullSpin,
                                                                              PixelSize.SixDegrees,
                                                                              Duration.SixMonths,
                                                                              MapQuantity.Intensity)),
        ]

        for descriptor, expected in test_cases:
            with self.subTest(descriptor):
                descriptor_parts = parse_map_descriptor(descriptor)
                self.assertEqual(expected, descriptor_parts)


if __name__ == '__main__':
    unittest.main()
