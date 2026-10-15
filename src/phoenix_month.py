"""A month of Phoenix summer, against the published climate record.

Runs two 31-day scenarios - a normal peak summer and a July-2023-scale
heatwave - and prints the model's day and night temperatures next to the
observed NWS Sky Harbor figures.

Two different claims are checked here, and they should not be confused:
  1. the FORCING, which should reproduce the published normals exactly;
  2. the MODEL RESPONSE, which should sit above the forcing, because the
     domain is urban surface and the station figure is the background air.
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

# NWS Phoenix Sky Harbor. Normals are 1991-2020; 2023 is the observed record.
OBSERVED = {
    "Peak summer (July normals)": {"high_c": 41.4, "low_c": 29.2},
    "Heatwave (July 2023)": {"high_c": 45.9, "low_c": 32.7},
}


def to_f(celsius):
    return celsius * 9.0 / 5.0 + 32.0


def daily_extremes(hours, values, start_hour=0.0):
    """Per-day max and min of a time series sampled at `hours`."""
    hours = np.asarray(hours)
    values = np.asarray(values)
    day_index = ((hours - start_hour) // 24).astype(int)
    highs, lows = [], []
    for day in range(day_index.max() + 1):
        in_day = day_index == day
        if in_day.any():
            highs.append(values[in_day].max())
            lows.append(values[in_day].min())
    return np.array(highs), np.array(lows)


def plaza_components():
    return [
        ReflectivePavement(x_m=20, y_m=20, radius_m=12, strength=0.55),
        Park(x_m=60, y_m=58, radius_m=16),
        WaterFeature(x_m=18, y_m=62, radius_m=8),
        Canopy(x_m=40, y_m=40, radius_m=14, strength=0.85),
        Mister(x_m=40, y_m=40, radius_m=9, strength=5e-6),
    ]


def forcing_extremes(weather, days=DAYS):
    """Daily high and low of the background forcing itself."""
    highs, lows = [], []
    for day in range(days):
        series = [
            weather.at(day * 24 + minute / 60.0).temperature_c
            for minute in range(0, 24 * 60, 10)
        ]
        highs.append(max(series))
        lows.append(min(series))
    return np.array(highs), np.array(lows)


def run_scenario(name, weather, params, components, gif_path=None):
    print("=" * 74)
    print(name)
    print("=" * 74)

    sim = Simulation(params=params, components=components, weather=weather)
    record = sim.run(HOURS)

    observed = OBSERVED[name]
    f_high, f_low = forcing_extremes(weather)
    a_high, a_low = daily_extremes(record.hour, record.air_mean)
    s_high, s_low = daily_extremes(record.hour, record.surface_mean)

    print("")
    print("%-34s %-20s %-20s" % ("", "model", "observed"))
    print("%-34s %-20s %-20s" % ("Forcing, daily high",
                                 fmt(f_high.mean()), fmt(observed["high_c"])))
    print("%-34s %-20s %-20s" % ("Forcing, daily low",
                                 fmt(f_low.mean()), fmt(observed["low_c"])))
    print("%-34s %+.2f C" % ("  forcing error, high",
                             f_high.mean() - observed["high_c"]))
    print("%-34s %+.2f C" % ("  forcing error, low",
                             f_low.mean() - observed["low_c"]))
    print("")
    print("%-34s %-20s %s" % ("Modelled air, daily high", fmt(a_high.mean()),
                              "+%.2f K above forcing" % (a_high.mean() - f_high.mean())))
    print("%-34s %-20s %s" % ("Modelled air, daily low", fmt(a_low.mean()),
                              "+%.2f K above forcing" % (a_low.mean() - f_low.mean())))
    print("%-34s %-20s" % ("Modelled surface, daily high", fmt(s_high.mean())))
    print("%-34s %-20s" % ("Modelled surface, daily low", fmt(s_low.mean())))
    print("%-34s %-20s" % ("Hottest surface cell, any time",
                           fmt(max(s.surface_c.max() for s in record.snapshots))))

    if name.startswith("Heatwave"):
        print("")
        print("%-34s %d / %d days" % ("Forcing days at or above 110 F",
                                      int((f_high >= 43.3).sum()), DAYS))
        print("%-34s %d / %d nights" % ("Forcing nights at or above 90 F",
                                        int((f_low >= 32.2).sum()), DAYS))

    if gif_path:
        plot_results(
            params=params,
            record=record,
            snapshot_hour=HOURS - 10.0,
            save_path=gif_path.replace(".gif", ".png"),
            show=False,
            create_gif=True,
            gif_path=gif_path,
            gif_interval=6,       # one frame per 6 h: 124 frames for a month
        )
        print("")
        print("Wrote %s and %s" % (gif_path, gif_path.replace(".gif", ".png")))
    print("")
    return record


def fmt(celsius):
    return "%5.1f C (%5.1f F)" % (celsius, to_f(celsius))


if __name__ == "__main__":
    params = Parameters(
        n=40, dx=2.0, dt=60.0,
        output_dt=3600.0,
        # The explicit-diffusion limit is 0.25*dx^2/K = 100 s; 30 s is well
        # inside it and 3x faster than the old 2 s, for a 0.004 K difference.
        diffusion_dt=30.0,
    )

    run_scenario(
        "Peak summer (July normals)",
        PhoenixSummerWeather(),
        params,
        plaza_components(),
    )
    run_scenario(
        "Heatwave (July 2023)",
        PhoenixHeatwaveWeather(),
        params,
        plaza_components(),
        gif_path="phoenix_heatwave_month.gif",
    )
    print("NOTE: the forcing is fitted to published Phoenix normals, but the "
          "surface physics is uncalibrated. Treat the response, not the "
          "forcing, as the uncertain part.")
