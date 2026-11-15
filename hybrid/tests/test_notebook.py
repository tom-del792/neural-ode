"""The walk-through notebook holds the code of the package as it is."""

import importlib.util
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def _builder():
    spec = importlib.util.spec_from_file_location("build_walkthrough", NOTEBOOKS / "build_walkthrough.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_walkthrough_is_current() -> None:
    """After a change of a source file: `python notebooks/build_walkthrough.py`."""
    assert (NOTEBOOKS / "walkthrough.ipynb").read_text() == _builder().build()


def test_walkthrough_contains_every_file() -> None:
    builder = _builder()
    notebook = json.loads((NOTEBOOKS / "walkthrough.ipynb").read_text())
    cells = {"".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    names = builder.files()
    assert "src/hybrid/warmup.py" in names and "src/hybrid/nn/mlp.py" in names
    for name in names:
        text = (builder.HYBRID / name).read_text()
        assert f"%%source {name}\n{text}".strip("\n") in cells, name


def test_notebooks_have_no_outputs() -> None:
    for path in NOTEBOOKS.glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            assert not cell.get("outputs") and cell.get("execution_count") is None, path.name
            assert cell["id"]
