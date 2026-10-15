"""The cooling-hotspot concept, as specified in the MCCR-5 midterm report.

Concept selected in the trade-off table (score 43): a designated cool spot
with concentrated cooling infrastructure, aimed at reducing thermal stress on
people rather than city-wide air temperature.

The layout below is derived from the report's requirements, not invented:

  COL-1  a publicly accessible space at most 28 C      -> the shaded core
  WOR-2  peak capacity 225 people, 45 with workspace   -> sizes the core
  LEI-1  open exercise space of at least 100 m2        -> the lawn
  LEI-4  cooled seating for at least 10% of visitors   -> within the core
  NAT-1  trees                                         -> the tree groves
  SAF-5  water misting as a sustainable cooling method -> the misting ring
  COL-3/4 shading                                      -> canopies, cool paths
  NAT-3  carbon neutral                                -> passive-first layout

The COL-1 vs NAT-3 conflict resolution in section 6.1 is what drives the
design: cooling is passive wherever possible, and the only powered element is
the misting ring.

Success measures reported here are the ones in the System Definition Matrix
that this model can actually produce. UTCI is NOT among them - see the note
at the end of the output.
"""
import numpy as np

from components import (
    Canopy,
    Mister,
    Park,
    ReflectivePavement,
    WaterFeature,
)
from config import Parameters
from data import plot_results
from simulation import Simulation
from simulation.weather import PhoenixHeatwaveWeather, PhoenixSummerWeather

DAYS = 31
HOURS = DAYS * 24.0

# Site geometry. 80 m x 80 m = 6400 m2, which at the WOR-2 peak capacity of
# 225 people is about 28 m2 per visitor - generous enough for seating,
# workspace and circulation without being a crowd.
SITE_M = 80.0
PEAK_CAPACITY = 225        # WOR-2
WORKSPACE_USERS = 45       # WOR-2
SEATING_USERS = 23         # LEI-4, 10% of 225
EXERCISE_MIN_M2 = 100.0    # LEI-1

# Misting rate per unit area. Kept deliberately low: NAT and COT-4 both
# penalise water use, and section 6.1 prefers passive cooling.
MIST_RATE = 4e-6           # kg/(m2 s)


def cooling_hotspot():
    """The component layout, laid down in the order the ground is built up."""
    return [
        # Cool paving across the whole approach, so the walk in is not the
        # hottest part of the visit (COL-4). Laid first; everything else
        # overwrites it where they overlap.
        ReflectivePavement(x_m=40, y_m=40, radius_m=40, strength=0.55),

        # LEI-1: open exercise lawn. Grass evapotranspires, so it cools the
        # air as well as providing the space.
        Park(x_m=62, y_m=20, radius_m=11),

        # A shallow water basin on the prevailing-wind side, so air reaching
        # the core has already passed over evaporating water.
        WaterFeature(x_m=18, y_m=62, radius_m=7),

        # NAT-1: tree groves on the three approach corners. Passive, no
        # water or power beyond irrigation, and they shade the paths.
        Canopy(x_m=16, y_m=20, radius_m=9, strength=0.80),
        Canopy(x_m=64, y_m=58, radius_m=9, strength=0.80),
        Canopy(x_m=20, y_m=42, radius_m=8, strength=0.80),

        # COL-1 / LEI-4: the shaded core. Heaviest shade on the site - this
        # is the sheltered space holding the seating and workspaces.
        Canopy(x_m=42, y_m=38, radius_m=14, strength=0.92),

        # SAF-5: misting ring inside the core, under the canopy. Misting
        # under shade is far more effective than misting in full sun, and it
        # is the only powered element on the site.
        Mister(x_m=42, y_m=38, radius_m=11, strength=MIST_RATE),
    ]


def zone_mask(params, x_m, y_m, radius_m):
    n, dx = params.n, params.dx
    rows, cols = np.ogrid[:n, :n]
    y = (rows + 0.5) * dx
    x = (cols + 0.5) * dx
    return (x - x_m) ** 2 + (y - y_m) ** 2 <= radius_m ** 2


def zone_series(record, mask, field):
    """Domain-masked mean of one field over time."""
    return np.array([getattr(s, field)[mask].mean() for s in record.snapshots])


def daily_extremes(hours, values):
    hours = np.asarray(hours)
    day = ((hours - hours[0]) // 24).astype(int)
    highs, lows = [], []
    for d in range(day.max() + 1):
        sel = day == d
        if sel.any():
            highs.append(values[sel].max())
            lows.append(values[sel].min())
    return np.array(highs), np.array(lows)


def to_f(c):
    return c * 9.0 / 5.0 + 32.0


def report(name, params, weather, gif_path=None):
    components = cooling_hotspot()
    site = Simulation(params=params, components=components, weather=weather)
    bare = Simulation(params=params, weather=weather)   # same site, untreated

    print("=" * 76)
    print(name)
    print("=" * 76)
    print("\n".join(site.describe()))
    print("")

    record = site.run(HOURS)
    base_record = bare.run(HOURS)

    core = zone_mask(params, 42, 38, 14)      # the sheltered core
    lawn = zone_mask(params, 62, 20, 11)      # LEI-1 exercise space
    whole = np.ones((params.n, params.n), dtype=bool)

    print("Success measures (System Definition Matrix)")
    print("-" * 76)

    # --- thermal stress reduction ---
    for label, mask in (("Sheltered core", core), ("Exercise lawn", lawn),
                        ("Whole site", whole)):
        air = zone_series(record, mask, "air_c")
        base_air = zone_series(base_record, mask, "air_c")
        surf = zone_series(record, mask, "surface_c")
        base_surf = zone_series(base_record, mask, "surface_c")
        a_hi, a_lo = daily_extremes(record.hour, air)
        ba_hi, ba_lo = daily_extremes(base_record.hour, base_air)
        s_hi, _ = daily_extremes(record.hour, surf)
        bs_hi, _ = daily_extremes(base_record.hour, base_surf)
        print("")
        print("  %s" % label)
        print("    Air, daily high       %5.1f C (%5.1f F)   untreated %5.1f C   "
              "reduction %5.2f K" % (a_hi.mean(), to_f(a_hi.mean()),
                                     ba_hi.mean(), ba_hi.mean() - a_hi.mean()))
        print("    Air, daily low        %5.1f C (%5.1f F)   untreated %5.1f C   "
              "reduction %5.2f K" % (a_lo.mean(), to_f(a_lo.mean()),
                                     ba_lo.mean(), ba_lo.mean() - a_lo.mean()))
        print("    Surface, daily high   %5.1f C (%5.1f F)   untreated %5.1f C   "
              "reduction %5.2f K" % (s_hi.mean(), to_f(s_hi.mean()),
                                     bs_hi.mean(), bs_hi.mean() - s_hi.mean()))

    # --- sustainability ---
    water_kg_s = site.total_water_kg_s()
    power_w = site.total_power_w()
    print("")
    print("  Sustainability")
    print("    Water (continuous)    %7.0f L/day   %9.0f L/year" % (
        water_kg_s * 86400.0, water_kg_s * 86400.0 * 365.0))
    print("    Water (12 h/day only) %7.0f L/day   %9.0f L/year" % (
        water_kg_s * 43200.0, water_kg_s * 43200.0 * 365.0))
    print("    Power usage           %7.0f W       %9.1f kWh/year" % (
        power_w, power_w * 8760.0 / 1000.0))

    # --- availability / capacity checks ---
    dx2 = params.dx ** 2
    core_m2 = core.sum() * dx2
    lawn_m2 = lawn.sum() * dx2
    print("")
    print("  Capacity against requirements")
    print("    Sheltered core area   %6.0f m2   %.1f m2/person at %d peak (WOR-2)"
          % (core_m2, core_m2 / PEAK_CAPACITY, PEAK_CAPACITY))
    print("    Exercise lawn area    %6.0f m2   LEI-1 needs >= %.0f m2   %s"
          % (lawn_m2, EXERCISE_MIN_M2,
             "PASS" if lawn_m2 >= EXERCISE_MIN_M2 else "FAIL"))

    if gif_path:
        plot_results(
            params=params, record=record, snapshot_hour=HOURS - 10.0,
            save_path=gif_path.replace(".gif", ".png"), show=False,
            create_gif=True, gif_path=gif_path, gif_interval=1,
        )
        print("")
        print("  Wrote %s and %s" % (gif_path, gif_path.replace(".gif", ".png")))
    print("")
    return record


if __name__ == "__main__":
    params = Parameters(
        n=40, dx=2.0, dt=60.0,
        output_dt=1800.0,     # 30 min sampling, 48 frames per simulated day
        diffusion_dt=30.0,
    )

    report("Peak summer (July normals)", params, PhoenixSummerWeather())
    report("Heatwave (July 2023 scale)", params, PhoenixHeatwaveWeather(),
           gif_path="hotspot_heatwave.gif")

    print("=" * 76)
    print("NOT YET MEASURABLE")
    print("=" * 76)
    print("  UTCI (the SDM's primary success measure) needs mean radiant")
    print("  temperature, which this model does not compute - there is no")
    print("  sky view factor and no radiant exchange with surrounding")
    print("  surfaces. Air temperature, humidity and wind are all available,")
    print("  so UTCI is reachable once MRT is added.")
    print("  People reached and walking travel time are siting questions,")
    print("  not simulation outputs.")
