import yaml

instrument = "lo"

coordinate_systems = {
    "hae": "ECLIPJ2000 (HAE)",
    "hgi": "Heliographic Inertial (HGI)",
    "hre": "Heliospheric Ram Ecliptic (HRE)",
    "hnu": "Heliospheric Nose Upfield (HNU)",
    "gcs": "Galactic (GCS)",
    "rc": "Ribbon-centered (RC)",
    "ccs": "custom (CCS)"
}

logical_source_description_parts = {
    "l090": "Lo-90 Sensor",
    "lc": "Lo Combined",
    "ena": "ENA Intensity",
    "spx": "Spectral Index",
    'h': "Hydrogen",
    'sf': "Spacecraft Frame",
    'hf': "Heliospheric Frame",
    'hk': "Heliospheric Kinematic Frame",
    'sp': "Survival Probability-Corrected",
    'nsp': "Non-Survival Probability-Corrected",
    'full': "Full Spin",
    'ram': "Ram",
    'anti': "Anti-Ram",
    "hae": "HAE",
    "2deg": "Rectangular 2 degree",
    "4deg": "Rectangular 4 degree",
    "6deg": "Rectangular 6 degree",
    "3mo": "3 Month Map",
    "6mo": "6 Month Map",
    "12mo": "12 Month Map",
}

descriptors = [
    "l090-ena-h-sf-sp-ram-hae-6deg-12mo",
    "l090-ena-h-hf-sp-ram-hae-6deg-12mo",
    "l090-spx-h-sf-nsp-ram-hae-6deg-12mo",
    "l090-spx-h-sf-sp-ram-hae-6deg-12mo",
    "l090-spx-h-hf-nsp-ram-hae-6deg-12mo",
    "l090-spx-h-hf-sp-ram-hae-6deg-12mo",
]

for descriptor in descriptors:
    descriptor_parts = descriptor.split('-')
    sensor = descriptor_parts[0][1:]
    [quantity, species, frame, sp_corrected, spin_range, coordinate_system, pixelation, time_range] = descriptor_parts[
                                                                                                      1:]

    products = {
        f"imap_{instrument}_l3_{descriptor}": {
            "Logical_source_description": "IMAP Lo Instrument Level 3 " + ', '.join(
                [logical_source_description_parts[part] for part in descriptor.split("-")]),
            "Data_level": "3",
            "Data_type": f"L3_{time_range}>Level-3 {time_range}",
            "Map_descriptor": descriptor,
            "Map_duration": time_range,
            "Instrument": f"Lo {sensor}" if sensor in ["45", "90"] else "Lo",
            "Reference_frame": "Heliospheric Kinematic (HK)" if frame == "hk" else (
                "Spacecraft" if frame == "sf" else "Heliospheric (CG)"),
            "Coordinate_system": coordinate_systems[coordinate_system],
            "Tessellation_type": "lon, lat" if "deg" in pixelation else "pix index",
            "Spin_range": "Ram",
            "Survival_corrected": "True" if sp_corrected == "sp" else "False",
            "Species": species.capitalize(),
            "Principal_data_quantity": "ENA Intensity" if quantity == "ena" else "ENA Spectral Index",
        }
    }

    print(yaml.dump(products))
