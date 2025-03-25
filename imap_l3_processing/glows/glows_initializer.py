from imap_l3_processing.glows.l3b.glows_l3b_dependencies import GlowsL3BDependencies, F107_INDEX_KEY, \
    LYMAN_ALPHA_COMPOSITE_KEY, OMNI2_DATA_KEY


class GlowsInitializer:
    def should_process(self, glows_l3b_dependencies: GlowsL3BDependencies) -> bool:
        f107_exists = glows_l3b_dependencies.ancillary_files[F107_INDEX_KEY] is not None
        lyman_alpha_exists = glows_l3b_dependencies.ancillary_files[LYMAN_ALPHA_COMPOSITE_KEY] is not None
        omni2_exists = glows_l3b_dependencies.ancillary_files[OMNI2_DATA_KEY] is not None
        return f107_exists and lyman_alpha_exists and omni2_exists
