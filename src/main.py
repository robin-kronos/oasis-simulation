from simulation.config import Parameters
from simulation.engine import Simulation

if __name__ == "__main__":
    p = Parameters()
    # TODO Components
    # components = [
    #     Component("canopy", x_m=40, y_m=40, radius_m=12, strength=0.85),
    #     Component("mister", x_m=40, y_m=40, radius_m=8, strength=1.0),
    #     Component("reflective_pavement", x_m=20, y_m=20, radius_m=10, strength=0.55),
    # ]

    # TODO add components parameter
    sim = Simulation(p )
    #TODO return history and data
    sim.run()
    # TODO Plots Data