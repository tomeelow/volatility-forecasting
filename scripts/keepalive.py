"""Visit the hosted dashboard in a real browser so it never goes to sleep.

Streamlit Community Cloud hibernates an app after 12 hours without traffic and
shows visitors a "wake up" button instead. A plain HTTP request does not count
as traffic - the app only sees a visit once a browser opens a session - so this
script loads the page in headless Chromium, presses the wake-up button if the
app is asleep, and waits until the dashboard has actually rendered.

Usage:  APP_URL=https://<name>.streamlit.app python scripts/keepalive.py
Exits non-zero if the dashboard does not come up, so a scheduled run fails loudly.
"""

from __future__ import annotations

import os
import re
import sys
import time

from playwright.sync_api import Error, sync_playwright

MARKER = "Volatility Forecasting Lab"          # text that only the running dashboard shows
WAKE_BUTTON = re.compile("get this app back up", re.IGNORECASE)
DEADLINE_SECONDS = 300                         # waking a sleeping app takes a minute or two


def _visible(frame, locator_factory) -> bool:
    try:
        return locator_factory(frame).first.is_visible()
    except Error:  # frame navigated or detached while we were looking
        return False


def main() -> int:
    url = os.environ.get("APP_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not url:
        print("Set APP_URL or pass the dashboard address as the first argument.")
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        started, woke = time.monotonic(), False
        while time.monotonic() - started < DEADLINE_SECONDS:
            for frame in page.frames:  # the hosted app renders inside an iframe
                if _visible(frame, lambda f: f.get_by_role("button", name=WAKE_BUTTON)):
                    print("App was asleep - waking it up.")
                    frame.get_by_role("button", name=WAKE_BUTTON).first.click()
                    woke = True
                elif _visible(frame, lambda f: f.get_by_text(MARKER)):
                    page.wait_for_timeout(5_000)  # stay long enough to register as a visit
                    print(f"Dashboard is up{' (woken from sleep)' if woke else ''}: {url}")
                    browser.close()
                    return 0
            page.wait_for_timeout(3_000)
        print(f"Dashboard did not render within {DEADLINE_SECONDS}s: {page.url}")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
