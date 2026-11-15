"""Command line interface: `volcast <step>`.

    volcast download    fetch daily bars from Yahoo Finance
    volcast features    build the modelling panel
    volcast forecast    walk-forward forecasts for all models (the slow step)
    volcast evaluate    loss tables, Diebold-Mariano tests, model confidence set
    volcast backtest    volatility-targeting backtest
    volcast figures     static charts for the README
    volcast all         everything above, in order
    volcast bundle      export the compact data files the hosted dashboard reads
    volcast app         launch the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from .config import ROOT

STEPS = ["download", "features", "forecast", "evaluate", "backtest", "figures"]


def _run_step(step: str, force: bool) -> None:
    from . import data, evaluate, features, walkforward

    if step == "download":
        data.download(force=force)
    elif step == "features":
        features.build_and_save(data.load_prices(), data.load_aux())
    elif step == "forecast":
        walkforward.run(features.load_panel())
    elif step == "evaluate":
        tables = evaluate.run(walkforward.load_forecasts())
        print(tables["leaderboard"].round(4).to_string())
    elif step == "backtest":
        from . import voltarget

        for name, table in voltarget.run_report().items():
            print(f"\n{name}\n{table.round(4).to_string()}")
    elif step == "figures":
        from . import figures

        figures.make_all()
    elif step == "bundle":
        from . import bundle

        bundle.export()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="volcast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all", "bundle", "app"])
    parser.add_argument("--force", action="store_true", help="re-download data even if cached")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.step == "app":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py")]))
    for step in STEPS if args.step == "all" else [args.step]:
        logging.info("=== %s ===", step)
        _run_step(step, args.force)


if __name__ == "__main__":
    main()
