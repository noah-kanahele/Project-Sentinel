from pathlib import Path

import pytest

from edge_vision.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def cfg() -> dict:
    """Fresh copy of configs/default.yaml for each test (tests may mutate it)."""
    return load_config(CONFIG_PATH)
