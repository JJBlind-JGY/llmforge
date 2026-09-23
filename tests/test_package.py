from importlib.metadata import version as distribution_version

import llmforge


def test_package_version_matches_distribution_metadata() -> None:
    assert llmforge.__version__ == distribution_version("llmforge")
