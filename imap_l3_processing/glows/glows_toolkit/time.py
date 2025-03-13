"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Time:
    """
    Time() class for IMAP and GLOWS times (with subseconds)
    """
    seconds: int
    subseconds: int
