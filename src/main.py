"""Project Oasis - run a scenario and plot it.

Prototype: synthetic weather forcing and effective parameters. Useful for
comparing interventions against each other, not for predicting conditions at
any real location.
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
from simulation.map.citymap import CityMap
from simulation.weather import SyntheticDiurnalWeather


def plaza_components(params):
    """A shaded plaza: trees over a misted centre, a park and a pond beside it."""
    return [
        # Laid first so later interventions can sit on top of it.
        ReflectivePavement(x_m=20, y_m=20, radius_m=12, strength=0.55),
        Park(x_m=60, y_m=58, radius_m=16),
        WaterFeature(x_m=18, y_m=62, radius_m=8),
        Canopy(x_m=40, y_m=40, radius_m=14, strength=0.85),
        # 5e-6 kg/(m2 s) is about 18 g per square metre per hour. Raising this
        # buys more cooling but the water bill rises with it - see the summary.
        Mister(x_m=40, y_m=40, radius_m=9, strength=5e-6),
    ]


def striped_base_map(params):
    """Alternating asphalt and grass strips, as a textured starting surface."""
    city_map = CityMap.initial(params)
    strip_width = 5
    for j in range(params.n):
        if (j // strip_width) % 2 == 0:
            city_map.surfacetype[:, j] = params.Asphalt
            city_map.albedo[:, j] = 0.12
        else:
            city_map.surfacetype[:, j] = params.Vegetation
            city_map.albedo[:, j] = 0.23
    return city_map


def summarise(label, record, sim, hours):
    air = np.array(record.air_mean)
    print("{0}: peak mean air {1:.2f} C, final {2:.2f} C".format(
        label, air.max(), air[-1]
    ))
    water_l = sim.total_water_kg_s() * hours * 3600.0
    energy_kwh = sim.total_power_w() * hours * 3600.0 / 3.6e6
    print("  water {0:.0f} L over {1:.0f} h, electricity {2:.1f} kWh".format(
        water_l, hours, energy_kwh
    ))
    return air.max()


if __name__ == "__main__":
    params = Parameters(n=40, dx=2.0, dt=60.0, output_dt=1800.0)
    hours = 48.0

    weather = SyntheticDiurnalWeather()
    components = plaza_components(params)

    sim = Simulation(
        params=params,
        components=components,
        map=striped_base_map(params),
        weather=weather,
    )
    print("\n".join(sim.describe()))
    print("")

    record = sim.run(hours)
    print(record)
    print("")

    peak = summarise("With interventions", record, sim, hours)

    # The same domain with nothing placed on it, to measure against.
    bare = Simulation(
        params=params, map=striped_base_map(params), weather=weather
    )
    bare_record = bare.run(hours)
    bare_peak = summarise("Bare surface", bare_record, bare, hours)
    print("")
    print("Change in peak mean air temperature: {0:+.2f} K".format(
        peak - bare_peak
    ))

    plot_results(
        params=params,
        record=record,
        snapshot_hour=14.0,
        save_path="oasis_run.png",
        show=False,
        create_gif=True,
        gif_path="oasis_run.gif",
        gif_interval=1,
    )
    print("")
    print("NOTE: synthetic forcing and uncalibrated parameters; "
          "not a real forecast for any location.")
