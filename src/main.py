from config import Parameters
from simulation import Simulation
import numpy as np

from src.data import plot_results
from src.simulation.map.citymap import CityMap

if __name__ == "__main__":
    p = Parameters()
    # TODO Components
    # components = [
    #     Component("canopy", x_m=40, y_m=40, radius_m=12, strength=0.85),
    #     Component("mister", x_m=40, y_m=40, radius_m=8, strength=1.0),
    #     Component("reflective_pavement", x_m=20, y_m=20, radius_m=10, strength=0.55),
    # ]

    city_map = np.zeros((p.n, p.n), dtype=int)

    city_map = CityMap.initial(p)

    strip_width = 5

    for j in range(p.n):
        if (j // strip_width) % 2 == 0:
            # Asphalt
            city_map.surfacetype[:, j] = p.Asphalt
            city_map.albedo[:, j] = 0.12

        else:
            # Grass
            city_map.surfacetype[:, j] = p.Vegetation
            city_map.albedo[:, j] = 0.23

    # Create simulation
    sim = Simulation(
        params=p,
        map=city_map
    )

    # Run simulation
    record = sim.run(48)

    print(record)

    plot_results(
        params=p,
        record=record,
        snapshot_hour=14.0,
        save_path="oasis_runJaspertry.png",
        show=True,
        create_gif=True,
        gif_path="oasis_runJaspertry.gif",
        gif_interval=60
    )

    # TODO Plots Data