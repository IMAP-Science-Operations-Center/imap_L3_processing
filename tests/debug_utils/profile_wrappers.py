import cProfile
from datetime import datetime

import pyinstrument


def cprofile_wrapper(function_to_be_decorated):
    def wrapper_cprofile(*args, **kwargs):
        with cProfile.Profile() as p:
            result = function_to_be_decorated(*args, **kwargs)
            print(f"========\n{function_to_be_decorated}")
            p.dump_stats(f"profile_{datetime.now().strftime("%Y%m%dT%H%M%S")}.prof")
        return result

    return wrapper_cprofile


def pyinstrument_wrapper(function_to_be_decorated):
    def wrapper_pyinstrument(*args, **kwargs):
        with pyinstrument.Profiler() as p:
            result = function_to_be_decorated(*args, **kwargs)
        p.open_in_browser()
        return result

    return wrapper_pyinstrument
