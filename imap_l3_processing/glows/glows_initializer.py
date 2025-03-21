from imap_l3_processing.models import UpstreamDataDependency


class GlowsInitializer:
    def should_process(self, upstream_dependencies: [UpstreamDataDependency]) -> bool:
        pass
