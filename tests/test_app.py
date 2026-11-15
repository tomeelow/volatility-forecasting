"""Smoke test: every dashboard page renders without raising, from both data sources."""

import pytest

from volcast import bundle, walkforward
from volcast.config import ROOT

PAGES = ["Overview", "Forecast explorer", "Model comparison", "Vol targeting", "Latest forecasts"]
SOURCES = {
    # Fresh pipeline output on this machine.
    "local": walkforward.FORECASTS_PATH.exists(),
    # The committed bundle: what a hosted deployment (and CI) sees.
    "bundle": bundle.FORECASTS.exists(),
}


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, source, monkeypatch):
    if not SOURCES[source]:
        pytest.skip(f"no {source} data")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VOLCAST_DATA", source)
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
    assert app.title[0].value


def test_bundle_matches_pipeline_output():
    """The bundle must reproduce the pipeline's forecasts (to float32 precision)."""
    if not all(SOURCES.values()):
        pytest.skip("needs both local artifacts and the bundle")
    import numpy as np

    keys = ["horizon", "ticker", "date"]
    local = walkforward.load_forecasts().sort_values(keys).reset_index(drop=True)
    bundled = bundle.forecasts().sort_values(keys).reset_index(drop=True)
    assert len(local) == len(bundled)
    assert (local[keys].to_numpy() == bundled[keys].to_numpy()).all()
    np.testing.assert_allclose(bundled["gbm"], local["gbm"], rtol=1e-6)
    np.testing.assert_allclose(bundled["target"], local["target"], rtol=1e-6, equal_nan=True)
