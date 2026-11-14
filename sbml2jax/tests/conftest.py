from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
#: the pinned models which are verified against roadrunner
MODELS_TOML = ROOT / "examples" / "models" / "models.toml"
#: small model of the documentation, the model of the tests without network
ABSORPTION = ROOT.parent / "docs" / "examples" / "absorption.xml"


@pytest.fixture
def absorption() -> Path:
    return ABSORPTION
