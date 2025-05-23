import spiceypy as spice


class SpiceypyFactory:

    @staticmethod
    def furnish(list_of_kernel_paths):
        for kernel in list_of_kernel_paths:
            spice.furnsh(str(kernel))

    @staticmethod
    def get_spiceypy():
        return spice
