from datetime import datetime

import imap_data_access

LAUNCH_DATE = "20250924"

GLOWS_PRODUCTS= "glows", {
    "l3a": [
        "hist"
    ]
}
SWAPI_PRODUCTS= "swapi",{
    "l3a": [
        "proton-sw",
        "alpha-sw",
        "pui-he",
    ],
    "l3b":[
        "combined"
    ]
}
HIT_PRODUCTS = "hit", {
    "l3": [
        "direct-events",
        "macropixel"
    ]
}

CODICE_PRODUCTS= "codice",{
    "l3a": [
        "lo-partial-densities",
        "lo-direct-events",
        "hi-direct-events",
    ],
    "l3b": [
        "hi-pitch-angle"
    ]
}

ALL_PRODUCTS = [GLOWS_PRODUCTS, SWAPI_PRODUCTS, HIT_PRODUCTS, CODICE_PRODUCTS]

TODAY = datetime.now().date().strftime("%Y%m%d")

if __name__ == "__main__":
    for instrument, descriptors_by_level in ALL_PRODUCTS:
        for data_level, descriptors in descriptors_by_level.items():
            for descriptor in descriptors:
                imap_data_access.reprocess(
                    instrument = instrument,
                    data_level = data_level,
                    descriptor = descriptor,
                    start_date = LAUNCH_DATE,
                    end_date = TODAY
                )