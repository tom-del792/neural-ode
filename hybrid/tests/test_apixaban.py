"""The apixaban case of `configs/`: its ids exist in the pinned model (downloads the model once)."""

from pathlib import Path

import numpy as np

from hybrid.config import read_config
from hybrid.data.dataset import generate_panel

from .conftest import write_case


def test_small_apixaban_panel(tmp_path: Path) -> None:
    config = read_config(
        write_case(tmp_path, "warmup_apixaban.toml", n_subjects=4, times="[0, 60, 240, 1440]")
    )
    panel = generate_panel(config)
    assert panel.x_names == ("PODOSE_api", "BW", "KI__f_renal_function", "f_cirrhosis")
    assert panel.y_names == ("Cve_api", "antiXa_activity", "PT", "aPTT")
    assert panel.y_clean.shape == (4, 4, 4) and np.all(np.isfinite(panel.y_clean))
    concentration, pt = panel.y_clean[..., 0], panel.y_clean[..., 2]
    # no drug before the dose; the drug prolongs the prothrombin time
    assert np.all(concentration[:, 0] == 0) and np.all(concentration[:, 1] > 0)
    assert np.all(pt[:, 1] > pt[:, 0])
