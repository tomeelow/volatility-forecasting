"""Smoke test: every dashboard page renders without raising (needs pipeline artifacts)."""

import pytest

from volcast import walkforward
from volcast.config import ROOT

pytestmark = pytest.mark.skipif(
    not walkforward.FORECASTS_PATH.exists(), reason="run `volcast all` first to create artifacts"
)

PAGES = ["Overview", "Forecast explorer", "Model comparison", "Vol targeting", "Latest forecasts"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
    assert app.title[0].value
