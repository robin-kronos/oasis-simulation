# Project Oasis

A prototype 2-D urban cooling simulator. It models how **shade**, **misting**
and **reflective pavement** change near-ground air temperature, surface
temperature and humidity across a small urban plot over a simulated day.

> **This is a prototype, not a validated model.** Weather forcing is a
> synthetic analytic cycle, and every coefficient is an effective bulk
> parameterization rather than a calibrated street-scale value. Output is
> useful for comparing interventions against each other, not for predicting
> conditions at any real location.

## Install

```bash
python -m pip install -e ".[dev]"
```

Runtime needs only `numpy`; `matplotlib` is required for figures and is pulled
in by the `plot` and `dev` extras.

## Run

```bash
oasis --scenario combined --plot
python -m oasis --scenario combined --plot        # without installing
```

Useful flags:

| Flag | Effect |
|---|---|
| `--scenario NAME` | one of `baseline`, `shade-only`, `mist-only`, `combined` |
| `--scenario-file PATH` | load a JSON scenario instead (see `scenarios/plaza.json`) |
| `--compare-baseline` | also run the no-intervention case and report the delta |
| `--hours`, `--dt`, `--grid`, `--dx` | override duration, time step and grid |
| `--plot [PATH]`, `--show` | write / display the four-panel figure |
| `--csv PATH`, `--json PATH` | export the time series and the run summary |
| `--list-components` | list registered component kinds |

Note that `--grid` alone changes the **domain size** (`n * dx`), which can move
built-in components outside the grid. Pass `--dx` alongside it to keep the
domain at 80 m.

## Library use

```python
from oasis import Parameters, Simulator, Canopy, Mister

sim = Simulator(Parameters(duration_hours=24.0), [
    Canopy(x_m=40, y_m=40, radius_m=12, strength=0.85),
    Mister(x_m=40, y_m=40, radius_m=8, strength=1.0),
])
history, resources = sim.run()
print(history.air_mean[-1], resources.water_litres)
```

## Layout

```
src/oasis/
  config.py          Parameters, validation, stability warnings
  psychrometrics.py  saturation pressure, humidity ratio <-> RH, dew point
  numerics.py        Laplacian, Gaussian footprints, CFL limits
  metrics.py         History recording and resource accounting
  scenarios.py       built-in scenarios and JSON loading
  viz.py             four-panel figure (matplotlib imported lazily)
  cli.py             command-line entry point
  weather/
    base.py          WeatherState and the WeatherSource protocol
    synthetic.py     synthetic diurnal forcing
  components/
    base.py          Component ABC, StaticFields, kind registry
    canopy.py        Canopy / Tree (shade)
    mister.py        Mister (evaporative cooling, draws power)
    pavement.py      ReflectivePavement (albedo)
  simulation/
    state.py         SimulationState: ta, ts, w
    physics.py       flux and tendency terms
    engine.py        Simulator: time integration
tests/               pytest suite
scenarios/           example JSON scenarios
```

### Model structure

Three prognostic fields evolve on a uniform grid with edge-replicated
(zero-normal-gradient) boundaries:

- `ts` — surface temperature, driven by absorbed shortwave, convective
  exchange with the air, linearized longwave loss, and lateral spreading.
- `ta` — near-ground air temperature over an effective 2 m layer, driven by
  surface exchange, relaxation toward background weather, evaporative cooling
  and horizontal mixing.
- `w` — humidity ratio, driven by evaporated mist, relaxation toward
  background, and horizontal mixing.

Components are static: each imprints itself once onto the `shade`, `albedo`
and `mist` maps at setup, and those maps do not change during a run.

### Adding a component type

Subclass `Component`, set `kind`, implement `apply`, and decorate with
`@register`:

```python
from oasis.components.base import Component, StaticFields, register

@register
@dataclass
class GreenRoof(Component):
    kind: str = "green_roof"

    def apply(self, fields: StaticFields, n: int, dx: float) -> None:
        k = self.footprint(n, dx)
        fields.albedo = fields.albedo * (1 - k) + 0.25 * k
```

It is then available to `--list-components`, `build()` and JSON scenarios.

## Tests

```bash
python -m pytest
```

The suite covers psychrometric round-trips, the Laplacian against analytic
fields, component registry and footprint behaviour, conservation and symmetry
properties of a run, directional checks that each intervention cools, and the
CLI surface.

## Known issues

These are inherited from the original prototype and are **not** fixed here,
because fixing them changes model output and that is a calibration decision:

- **Near-ground air runs ~8 K above background.** With `surface_air_h = 10`
  and `ventilation_tau = 60 s`, the air layer equilibrates well above the
  driving weather — peak mean air reaches ~51 °C against a 43 °C background.
  Real urban canopy air typically sits 1–3 K above the reference. The
  numerical clamps are *not* binding, so this is the parameterization's
  genuine solution, not clipping.
- **`solar_absorptivity` is unused.** For an opaque surface `1 - albedo`
  already is the absorbed fraction, so applying both would double-count. The
  parameter is retained in `Parameters` but deliberately not applied; see the
  comment in `simulation/physics.py`.
- **The RH figure panel is fixed to 0–100 %.** Actual values span roughly
  15–28 %, so the misting signal is invisible on that scale. Auto-scaling the
  panel would show it.
- **Component centres do not snap to cell centres.** A component at
  `(40, 40)` on a 2 m grid falls between cell centres at 39 m and 41 m, so its
  peak footprint weight is ~0.94 rather than 1.0 and its effective strength is
  slightly under-applied. Placing components on odd coordinates avoids this.
- **Mist water is assumed fully evaporated** at the configured rate regardless
  of ambient humidity. There is no saturation limit, so in humid conditions
  the model will over-predict evaporative cooling.

## Status

The package reproduces the original `src/oasis_simulator.py` prototype
**bit-identically** (all fields and series agree to 0.0) while splitting it
into testable modules. The earlier single-file prototype and the
`src/simulation/` and `src/components/` sketches are superseded by
`src/oasis/` and can be deleted once you are happy with the replacement.

Planned next (not yet implemented): thermal-comfort index, PV generation
offsetting mister draw, wind advection, and multi-day runs.
