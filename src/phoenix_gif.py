"""A smooth month-long animation of the Phoenix heatwave scenario.

Samples every 30 simulated minutes and keeps every frame, so one day is 48
frames - about two seconds at 24 fps, which reads as a continuous day-night
cycle rather than a slideshow. The file is large; that is the trade.
"""
from components import (
    Canopy,
    Mister,
    Park,
    ReflectivePavement,
    WaterFeature,
)
from config import Parameters
from data.viz import make_gif
from simulation import Simulation
from simulation.weather import PhoenixHeatwaveWeather

DAYS = 31
HOURS = DAYS * 24.0

if __name__ == "__main__":
    params = Parameters(
        n=40, dx=2.0, dt=60.0,
        output_dt=1800.0,     # a sample every 30 simulated minutes
        diffusion_dt=30.0,
    )
    components = [
        ReflectivePavement(x_m=20, y_m=20, radius_m=12, strength=0.55),
        Park(x_m=60, y_m=58, radius_m=16),
        WaterFeature(x_m=18, y_m=62, radius_m=8),
        Canopy(x_m=40, y_m=40, radius_m=14, strength=0.85),
        Mister(x_m=40, y_m=40, radius_m=9, strength=5e-6),
    ]

    sim = Simulation(
        params=params, components=components, weather=PhoenixHeatwaveWeather()
    )
    print("\n".join(sim.describe()))
    record = sim.run(HOURS)
    print(record)
    print("")
    print("Rendering {0} frames at 48 per simulated day...".format(
        len(record.snapshots)
    ))

    make_gif(
        params=params,
        record=record,
        save_path="phoenix_heatwave_smooth.gif",
        interval=1,      # keep every sample
        fps=24,
    )
    print("Wrote phoenix_heatwave_smooth.gif")
