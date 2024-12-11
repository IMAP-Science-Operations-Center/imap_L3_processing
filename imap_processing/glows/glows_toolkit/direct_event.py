"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""
import struct
from dataclasses import dataclass
from .constants import SUBSECOND_LIMIT

@dataclass(frozen=True)
class DirectEvent:
    """
    DirectEvent() class for IMAP/GLOWS
    """
    seconds: int
    subseconds: int
    impulse_length: int
    multi_event: bool = False

    # based on deserialize() in glows_appsw-master/tooling/science/models/direct_event.py by KPLabs
    def build_event_from_uncompressed_data(raw) -> 'DirectEvent':
        """
        Build direct event from raw binary 8-byte array assuming that it contains uncompressed timestamps
        """
        assert len(raw) == 8
        values = struct.unpack('>II', raw)
        return DirectEvent( # see glows_appsw/-/blob/master/libs/science/data/include/science/science_direct_event.h (bits 22,23 are not used)
            seconds=values[0],
            subseconds=values[1] & 0x1FFFFF, # subsecond encoding on the least significant 21 bits
            impulse_length=(values[1] >> 24) & 0xFF, # first byte encodes the impulse length
            multi_event=bool((values[1] >> 23) & 0b1), # KPLabs says it is set by FPGA and currently not used by AppSW at all
        )

    # based on  _build_event() in glows_appsw-master/tooling/science/models/direct_event_file_decompressor.py by KPLabs
    def build_event_from_compressed_data(diff: int, length: int, current_event: 'DirectEvent') -> 'DirectEvent':
        """
        Build direct event assuming that it contains timestamps compressed as timedeltas
        """
        subseconds = current_event.subseconds + diff
        seconds = current_event.seconds

        if subseconds >= SUBSECOND_LIMIT:
            full_seconds = int(subseconds/SUBSECOND_LIMIT)
            seconds += full_seconds
            subseconds -= full_seconds*SUBSECOND_LIMIT

        return DirectEvent(seconds=seconds, subseconds=subseconds, impulse_length=length, multi_event=False)
