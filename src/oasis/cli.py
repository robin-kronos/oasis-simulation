"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .components import available_kinds
from .config import ConfigError, Parameters
from .metrics import History
from .scenarios import BUILTIN, Scenario, load
from .simulation.engine import Simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oasis",
        description="Project Oasis: prototype 2-D urban cooling simulator. "
                    "Synthetic forcing and effective parameters; not a "
                    "validated street-scale model.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--scenario", default="combined", choices=sorted(BUILTIN),
        help="built-in scenario to run (default: combined)",
    )
    source.add_argument(
        "--scenario-file", metavar="PATH",
        help="JSON scenario file; used instead of --scenario",
    )
    parser.add_argument("--hours", type=float, help="simulated duration in hours")
    parser.add_argument("--dt", type=float, help="time step in seconds")
    parser.add_argument("--grid", type=int, metavar="N", help="cells per side")
    parser.add_argument("--dx", type=float, help="cell size in metres")
    parser.add_argument(
        "--plot", metavar="PATH", nargs="?", const="oasis_results.png",
        help="write the four-panel figure (default: oasis_results.png)",
    )
    parser.add_argument("--show", action="store_true", help="display the figure")
    parser.add_argument(
        "--csv", metavar="PATH", help="write the scalar time series as CSV"
    )
    parser.add_argument(
        "--json", metavar="PATH", help="write the run summary as JSON"
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="also run the no-intervention case and report the difference",
    )
    parser.add_argument("--progress", action="store_true", help="print each sample")
    parser.add_argument(
        "--list-components", action="store_true",
        help="list registered component kinds and exit",
    )
    return parser


def _apply_overrides(params: Parameters, args: argparse.Namespace) -> None:
    if args.hours is not None:
        params.duration_hours = args.hours
    if args.dt is not None:
        params.dt = args.dt
    if args.grid is not None:
        params.n = args.grid
    if args.dx is not None:
        params.dx = args.dx


def _write_csv(history: History, path: str) -> None:
    import csv

    series = history.to_dict()
    keys = list(series)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for row in zip(*(series[k] for k in keys)):
            writer.writerow(["{0:.6g}".format(v) for v in row])


def _load_scenario(args: argparse.Namespace) -> Scenario:
    if args.scenario_file:
        return Scenario.from_json(args.scenario_file)
    return load(args.scenario)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_components:
        print("Registered component kinds:")
        for kind in available_kinds():
            print("  " + kind)
        return 0

    try:
        scenario = _load_scenario(args)
        _apply_overrides(scenario.params, args)
        sim = Simulator(scenario.params, scenario.components)
    except ConfigError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print("error: could not load scenario: {0}".format(exc), file=sys.stderr)
        return 2

    print("Scenario: {0}".format(scenario.name))
    if scenario.description:
        print("  " + scenario.description)
    print("\n".join(sim.describe()))
    print("")

    history, resources = sim.run(progress=args.progress)

    summary = {
        "scenario": scenario.name,
        "final_air_mean_c": history.air_mean[-1],
        "final_air_max_c": history.air_max[-1],
        "final_surface_mean_c": history.surface_mean[-1],
        "final_rh_mean_pct": history.rh_mean[-1],
        "peak_air_mean_c": max(history.air_mean),
        "water_litres": resources.water_litres,
        "energy_kwh": resources.energy_kwh,
        "peak_power_w": resources.peak_power_w,
    }

    print("Results")
    print("  Peak mean air temperature:  {0:.2f} C".format(summary["peak_air_mean_c"]))
    print("  Final mean air temperature: {0:.2f} C".format(summary["final_air_mean_c"]))
    print("  Final mean surface temp:    {0:.2f} C".format(
        summary["final_surface_mean_c"]))
    print("  Final mean RH:              {0:.1f} %".format(
        summary["final_rh_mean_pct"]))
    for line in resources.summary_lines():
        print("  " + line)

    if args.compare_baseline and scenario.components:
        base_params = Parameters(**scenario.params.to_dict())
        base_hist, _ = Simulator(base_params, []).run()
        delta_peak = max(history.air_mean) - max(base_hist.air_mean)
        summary["baseline_peak_air_mean_c"] = max(base_hist.air_mean)
        summary["delta_peak_air_mean_c"] = delta_peak
        print("")
        print("Versus no-intervention baseline")
        print("  Baseline peak mean air: {0:.2f} C".format(max(base_hist.air_mean)))
        print("  Change from scenario:   {0:+.2f} C".format(delta_peak))

    if args.csv:
        _write_csv(history, args.csv)
        print("")
        print("Wrote time series to {0}".format(args.csv))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("Wrote summary to {0}".format(args.json))
    if args.plot or args.show:
        try:
            from .viz import plot_results

            plot_results(
                scenario.params, history, save_path=args.plot, show=args.show
            )
            if args.plot:
                print("Wrote figure to {0}".format(args.plot))
        except ImportError:
            print(
                "warning: matplotlib is not installed; skipping the figure",
                file=sys.stderr,
            )

    print("")
    print("NOTE: synthetic forcing and uncalibrated parameters; "
          "not a real forecast for any location.")
    return 0
