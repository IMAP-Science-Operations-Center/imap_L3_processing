import yaml

instrument = "hi"

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
    "h90": "Hi-90 Sensor",
    "h45": "Hi-45 Sensor",
    "hic": "Hi Combined",
    "ena": "ENA Intensity",
    "spx": "Spectral Index",
    'h': "Hydrogen",
    'sf': "Spacecraft Frame",
    'hf': "Heliospheric Frame",
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
    "1yr": "12 Month Map",
}

descriptors = [
    "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
    "hic-ena-h-hf-sp-full-hae-6deg-1yr",
    "h90-ena-h-sf-sp-ram-hae-6deg-1yr",
    "h90-ena-h-hf-sp-ram-hae-6deg-1yr",
    "h90-ena-h-sf-sp-anti-hae-6deg-1yr",
    "h90-ena-h-hf-sp-anti-hae-6deg-1yr",
    "h90-ena-h-sf-sp-full-hae-6deg-6mo",
    "h90-ena-h-hf-sp-full-hae-6deg-6mo",
    "h45-ena-h-sf-sp-ram-hae-6deg-1yr",
    "h45-ena-h-hf-sp-ram-hae-6deg-1yr",
    "h45-ena-h-sf-sp-anti-hae-6deg-1yr",
    "h45-ena-h-hf-sp-anti-hae-6deg-1yr",
    "h45-ena-h-sf-sp-full-hae-6deg-6mo",
    "h45-ena-h-hf-sp-full-hae-6deg-6mo",
    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
    "hic-ena-h-hf-sp-full-hae-4deg-1yr",
    "h90-ena-h-sf-sp-ram-hae-4deg-1yr",
    "h90-ena-h-hf-sp-ram-hae-4deg-1yr",
    "h90-ena-h-sf-sp-anti-hae-4deg-1yr",
    "h90-ena-h-hf-sp-anti-hae-4deg-1yr",
    "h90-ena-h-sf-sp-full-hae-4deg-6mo",
    "h90-ena-h-hf-sp-full-hae-4deg-6mo",
    "h45-ena-h-sf-sp-ram-hae-4deg-1yr",
    "h45-ena-h-hf-sp-ram-hae-4deg-1yr",
    "h45-ena-h-sf-sp-anti-hae-4deg-1yr",
    "h45-ena-h-hf-sp-anti-hae-4deg-1yr",
    "h45-ena-h-sf-sp-full-hae-4deg-6mo",
    "h45-ena-h-hf-sp-full-hae-4deg-6mo",
    "hic-spx-h-hf-sp-full-hae-6deg-1yr",
    "h45-spx-h-hf-sp-ram-hae-6deg-1yr",
    "h45-spx-h-hf-sp-anti-hae-6deg-1yr",
    "h45-spx-h-hf-sp-full-hae-6deg-6mo",
    "h90-spx-h-hf-sp-ram-hae-6deg-1yr",
    "h90-spx-h-hf-sp-anti-hae-6deg-1yr",
    "h90-spx-h-hf-sp-full-hae-6deg-6mo",
    "hic-spx-h-hf-sp-full-hae-4deg-1yr",
    "h45-spx-h-hf-sp-ram-hae-4deg-1yr",
    "h45-spx-h-hf-sp-anti-hae-4deg-1yr",
    "h45-spx-h-hf-sp-full-hae-4deg-6mo",
    "h90-spx-h-hf-sp-ram-hae-4deg-1yr",
    "h90-spx-h-hf-sp-anti-hae-4deg-1yr",
    "h90-spx-h-hf-sp-full-hae-4deg-6mo",
]

for descriptor in descriptors:
    descriptor_parts = descriptor.split('-')
    sensor = descriptor_parts[0][1:]
    [quantity, species, frame, sp_corrected, spin_range, coordinate_system, pixelation, time_range] = descriptor_parts[
                                                                                                      1:]

    products = {
        f"imap_{instrument}_l3_{descriptor}": {
            "Logical_source_description": "IMAP Hi Instrument Level 3 " + ', '.join(
                [logical_source_description_parts[part] for part in descriptor.split("-")]),
            "Data_level": "3",
            "Data_type": f"L3_{time_range}>Level-3 {time_range}",
            "Map_descriptor": descriptor,
            "Map_duration": time_range,
            "Instrument": f"Hi {sensor}" if sensor in ["45", "90"] else "Hi",
            "Reference_frame": "Spacecraft" if frame == "sf" else "Heliospheric (CG)",
            "Coordinate_system": coordinate_systems[coordinate_system],
            "Tessellation_type": "lon, lat" if "deg" in pixelation else "pix index",
            "Spin_range": logical_source_description_parts[spin_range],
            "Survival_corrected": "True" if sp_corrected == "sp" else "False",
            "Species": species.capitalize(),
            "Principal_data_quantity": "ENA Intensity" if quantity == "ena" else "ENA Spectral Index",
        }
    }

    print(yaml.dump(products))
