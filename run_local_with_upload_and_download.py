import subprocess
import sys

if __name__ == "__main__":
    target = sys.argv[1]
    subtarget = sys.argv[2] if len(sys.argv) > 2 else None
    match target, subtarget:
        case "hi", "combine-l2":
            version = "v004"
            subprocess.run([sys.executable,
                            'imap_l3_data_processor.py', '--instrument', 'hi', '--data-level', 'l3', '--start-date',
                            '20250415', '--version', version, '--descriptor', 'hic-ena-h-hf-nsp-full-hae-4deg-1yr',
                            '--dependency',
                            f"""[{{"type": "science", "files": ["imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-1yr_20250415_{version}.cdf"]}},
            {{"type": "science", "files": ["imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-1yr_20250415_{version}.cdf"]}},
            {{"type": "science", "files": ["imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-1yr_20250415_{version}.cdf"]}},
            {{"type": "science", "files": ["imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-1yr_20250415_{version}.cdf"]}}]"""])
        case "hi", "spx":
            subprocess.run([sys.executable, 'imap_l3_data_processor.py', '--instrument', 'hi', '--data-level', 'l3',
                            '--start-date', '20250415', '--version', 'v002', '--descriptor',
                            'h90-spx-h-hf-sp-full-hae-4deg-6mo', '--dependency',
                            '[{"type": "science", "files": ["imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v002.cdf"]}]'])
        case "hi", "survival":
            subprocess.run([sys.executable, 'imap_l3_data_processor.py', '--instrument', 'hi', '--data-level', 'l3',
                            '--start-date', '20250415', '--version', 'v002', '--descriptor',
                            'h90-ena-h-sf-sp-full-hae-4deg-6mo', '--dependency',
                            '[{"type": "science", "files": ["imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001.cdf"]},\n{"type": "science", "files": ["imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20250415_v001.cdf"]}]'])
        case "glows", None:
            subprocess.run([sys.executable, 'imap_l3_data_processor.py', '--instrument', 'glows', '--data-level', 'l3b',
                            '--start-date', '20100104', '--version', 'v004', '--dependency',
                            '[{"type": "science", "files": ["imap_glows_l3a_hist_20100104_v002.cdf"]},'
                            '{"type": "ancillary", "files":["imap_glows_bad-days-list_20100101_v001.dat"]},'
                            '{"type": "ancillary", "files":["imap_glows_WawHelioIonMP_20100101_v002.json"]},'
                            '{"type": "ancillary", "files":["imap_glows_uv-anisotropy-1CR_20100101_v001.json"]},'
                            '{"type": "ancillary", "files":["imap_glows_pipeline-settings-L3bc_20250707_v002.json"]}]'])
        case "swapi", None:
            subprocess.run([sys.executable, 'imap_l3_data_processor.py', '--instrument', 'swapi', '--data-level', 'l3a',
                            '--descriptor', 'proton-sw', '--start-date', '20250606', '--version', 'v000',
                            '--dependency',
                            '[{"type":"science","files":["imap_swapi_l2_sci_20250606_v007.cdf"]},{"type":"ancillary","files":["imap_swapi_density-temperature-lut_20240905_v000.dat"]},{"type":"ancillary","files":["imap_swapi_alpha-density-temperature-lut_20240920_v000.dat"]},{"type":"ancillary","files":["imap_swapi_clock-angle-and-flow-deflection-lut_20240918_v000.dat"]},{"type":"ancillary","files":["imap_swapi_energy-gf-lut_20240923_v000.dat"]},{"type":"ancillary","files":["imap_swapi_instrument-response-lut_20241023_v000.zip"]},{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"]}]'])
            subprocess.run([sys.executable, 'imap_l3_data_processor.py', '--instrument', 'swapi', '--data-level', 'l3b',
                            '--start-date', '20250606', '--version', 'v003', '--dependency',
                            '[{"type": "science", "files": ["imap_swapi_l2_sci_20250606_v002.cdf"]}]'])
        case "codice", "lo-direct-events":
            subprocess.run(
                [sys.executable, 'imap_l3_data_processor.py', '--instrument', 'codice', '--data-level', 'l3a',
                 '--descriptor', 'lo-direct-events', '--start-date', '20241110', '--version', 'v000',
                 '--dependency',
                 '['
                 '{"type": "science", "files": ["imap_codice_l1a_lo-sw-priority_20241110_v002.cdf"]},'
                 '{"type": "science", "files": ["imap_codice_l1a_lo-nsw-priority_20241110_v002.cdf"]},'
                 '{"type": "science", "files": ["imap_codice_l2_lo-direct-events_20241110_v002.cdf"]},'
                 '{"type": "ancillary", "files": ["imap_codice_mass-coefficient-lookup_20241110_v002.csv"]}]'
                 ])
        case "codice", ("lo-partial-densities" | "partial-densities"):
            subprocess.run(
                [sys.executable, 'imap_l3_data_processor.py', '--instrument', 'codice', '--data-level', 'l3a',
                 '--descriptor', 'lo-partial-densities', '--start-date', '20241110', '--version', 'v000',
                 '--dependency',
                 '['
                 '{"type": "science", "files": ["imap_codice_l2_lo-sw-species_20241110_v002.cdf"]},'
                 '{"type": "ancillary", "files": ["imap_codice_mass-per-charge_20241110_v002.csv"]}]'
                 ])
        case "codice", ("lo-sw-ratios" | "ratios"):
            subprocess.run(
                [sys.executable, 'imap_l3_data_processor.py', '--instrument', 'codice', '--data-level', 'l3a',
                 '--descriptor', 'lo-sw-ratios', '--start-date', '20241110', '--version', 'v000',
                 '--dependency',
                 '['
                 '{"type": "science", "files": ["imap_codice_l3a_lo-partial-densities_20241110_v000.cdf"]}'
                 ']'
                 ])
        case "codice", ("lo-sw-abundances" | "abundances"):
            subprocess.run(
                [sys.executable, 'imap_l3_data_processor.py', '--instrument', 'codice', '--data-level', 'l3a',
                 '--descriptor', 'lo-sw-abundances', '--start-date', '20241110', '--version', 'v000',
                 '--dependency',
                 '['
                 '{"type": "science", "files": ["imap_codice_l3a_lo-partial-densities_20241110_v000.cdf"]}'
                 ']'
                 ])
        case "codice", "hi-direct-events":
            subprocess.run(
                [sys.executable, 'imap_l3_data_processor.py', '--instrument', 'codice', '--data-level', 'l3a',
                 '--descriptor', 'hi-direct-events', '--start-date', '20241110', '--version', 'v000',
                 '--dependency',
                 '['
                 '{"type": "science", "files": ["imap_codice_l2_hi-direct-events_20241110_v002.cdf"]},'
                 '{"type": "ancillary", "files": ["imap_codice_tof-lookup_20241110_v002.csv"]}]'
                 ])
        case "codice", "hi-pitch-angle":
            subprocess.run(
                [sys.executable, 'imap_l3_data_processor.py', '--instrument', 'codice', '--data-level', 'l3b',
                 '--descriptor', 'hi-pitch-angle', '--start-date', '20241110', '--version', 'v000',
                 '--dependency',
                 '['
                 '{"type": "science", "files": ["imap_codice_l2_hi-sectored_20241110_v002.cdf"]},'
                 '{"type": "science", "files": ["imap_mag_l1d_norm-mago_20250630_v001.cdf"]}'
                 ']'
                 ])
        case _:
            raise NotImplementedError("Target instrument or product not implemented")
