import json

import pytest

from oasis.cli import main


# Coarse 10x10 cells at 8 m keeps the 80 m domain the scenarios assume,
# so runs stay fast without moving any component outside the grid.
SHORT = ["--hours", "0.5", "--grid", "10", "--dx", "8"]


def test_list_components_exits_cleanly(capsys):
    assert main(["--list-components"]) == 0
    out = capsys.readouterr().out
    assert "mister" in out and "canopy" in out


def test_default_run_reports_results(capsys):
    assert main(SHORT) == 0
    out = capsys.readouterr().out
    assert "Scenario: combined" in out
    assert "Final mean air temperature:" in out
    assert "NOTE:" in out


def test_each_builtin_scenario_runs(capsys):
    for name in ("baseline", "shade-only", "mist-only", "combined"):
        assert main(["--scenario", name] + SHORT) == 0
        assert "Scenario: {0}".format(name) in capsys.readouterr().out


def test_unknown_scenario_is_rejected_by_argparse():
    with pytest.raises(SystemExit) as exc:
        main(["--scenario", "does-not-exist"])
    assert exc.value.code == 2


def test_missing_scenario_file_returns_an_error(capsys):
    assert main(["--scenario-file", "no-such-file.json"]) == 2
    assert "error" in capsys.readouterr().err


def test_malformed_scenario_file_returns_an_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"components": [{"kind": "teleporter"}]}', encoding="utf-8")
    assert main(["--scenario-file", str(bad)]) == 2
    assert "teleporter" in capsys.readouterr().err


def test_scenario_file_round_trips(tmp_path, capsys):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({
        "name": "from-file",
        "parameters": {"n": 10, "duration_hours": 0.5},
        "components": [
            {"kind": "mister", "x_m": 10, "y_m": 10, "radius_m": 4, "strength": 1.0}
        ],
    }), encoding="utf-8")
    assert main(["--scenario-file", str(path)]) == 0
    assert "Scenario: from-file" in capsys.readouterr().out


def test_csv_and_json_outputs_are_written(tmp_path, capsys):
    csv_path = tmp_path / "series.csv"
    json_path = tmp_path / "summary.json"
    assert main(SHORT + ["--csv", str(csv_path), "--json", str(json_path)]) == 0
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("hour,")
    assert len(lines) > 1
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["scenario"] == "combined"
    assert summary["water_litres"] > 0


def test_compare_baseline_reports_a_difference(capsys):
    assert main(SHORT + ["--compare-baseline"]) == 0
    out = capsys.readouterr().out
    assert "Versus no-intervention baseline" in out
    assert "Change from scenario:" in out


def test_invalid_parameter_override_is_reported(capsys):
    assert main(["--grid", "1"]) == 2
    assert "error" in capsys.readouterr().err


def test_shrinking_the_domain_below_a_component_is_reported(capsys):
    # --grid without --dx shrinks the domain; built-in components sit at 40 m.
    assert main(["--scenario", "combined", "--grid", "10", "--hours", "0.5"]) == 2
    assert "outside the 20 m domain" in capsys.readouterr().err
