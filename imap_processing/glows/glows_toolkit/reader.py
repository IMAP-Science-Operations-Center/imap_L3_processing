"""@package docstring
Author: KPLabs with some modifications by Marek Strumik, maro at cbk.waw.pl
"""
import struct
from typing import Any, cast, Tuple, Union

class Reader:
    """
    Reader() class for reading binary stream with GLOWS data
    """

    def __init__(self, buffer: Union[memoryview, bytes]):
        """
        Constructor for this class, sets init values of its fields
        """
        self._buffer = buffer
        self._offset = 0
        self._overrun = False

    @property
    def overrun(self) -> bool:
        """ overrun property """
        return self._overrun

    @property
    def remaining_size(self) -> int:
        """ remaining_size property"""
        if self._overrun:
            return 0
        return len(self._buffer) - self._offset

    def _read_part(self, length: int) -> bytes:
        """ read bytes from binary stream """
        if self._overrun:
            return bytes([0] * length)

        if self._offset + length > len(self._buffer):
            self._overrun = True
            return bytes([0] * length)

        part = self._buffer[self._offset:self._offset + length]
        self._offset += length
        return part

    def read(self, pack_format: str) -> Tuple[Any, ...]:
        """ read bytes according to pack_format """
        length = struct.calcsize(pack_format)
        part = self._read_part(length)
        return struct.unpack(pack_format, part)

    def read_uint8(self) -> int:
        """ read one byte as uint """
        return cast(int, self.read('<B')[0])

    def read_uint16_be(self) -> int:
        """ read 2 bytes as big-endian uint """
        return cast(int, self.read('>H')[0])

    def read_uint24_be(self) -> int:
        """ read 3 bytes as big-endian uint """
        high, low = cast(Tuple[int, int], self.read('>BH'))
        return high << 16 | low

    def read_uint32_be(self) -> int:
        """ read 4 bytes as big-endian uint """
        return cast(int, self.read('>L')[0])

    def read_buffer(self, size: int) -> bytes:
        """ read bytes from binary stream """
        return self._read_part(size)
