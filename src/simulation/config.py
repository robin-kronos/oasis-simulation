from dataclasses import dataclass


@dataclass
class Parameters:
    # Simulation Parameters
    n: int = 40                  # 40 x 40 cells, each 2 m x 2 m => 80 m x 80 m
    dx: float = 2.0              # metres per cell
    dt: float = 30.0             # seconds; explicit time step
    start_time: float =  0.0     # Start Time
    duration_hours: float = 24.0

    # Physics Parameters
    surface_heat_capacity: float = 150_000.0  # J / (m2 K), effective layer
    surface_air_h: float = 10.0  # W / (m2 K)
    surface_radiative_h: float = 5.5  # W / (m2 K), linearized longwave loss
    surface_diffusivity: float = 0.0005  # m2/s, effective lateral spreading
    air_diffusivity: float = 0.01        # m2/s, effective horizontal mixing
    moisture_diffusivity: float = 0.005   # m2/s, effective horizontal mixing
    air_density: float = 1.15            # kg/m3
    air_cp: float = 1005.0                # J/(kg K)
    air_mixing_height: float = 2.0        # m, effective near-ground layer
    ventilation_tau: float = 60.0       # s, relaxation toward background weather
    base_albedo: float = 0.20
    solar_absorptivity: float = 0.80
    mist_water_kg_m2_s: float = 1.0e-5    # effective evaporated water rate
    latent_heat: float = 2.43e6           # J/kg
