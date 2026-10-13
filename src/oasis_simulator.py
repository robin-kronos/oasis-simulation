"""Project Oasis: simplified 2-D urban cooling simulator.

Prototype only: synthetic weather forcing and effective parameterizations.
Not a validated street-scale weather or UTCI model.

Install: pip install numpy matplotlib
Run:     python oasis_simulator.py
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Component:
    kind: str
    x_m: float
    y_m: float
    radius_m: float
    strength: float = 1.0
    # strength means: misting intensity multiplier; shade coverage multiplier;
    # or albedo override for reflective pavement.


@dataclass
class Parameters:
    n: int = 40                  # 40 x 40 cells, each 2 m x 2 m => 80 m x 80 m
    dx: float = 2.0              # metres per cell
    dt: float = 30.0             # seconds; explicit time step
    duration_hours: float = 24.0
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


def laplacian(a: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian with zero-normal-gradient (edge-replicated) boundaries."""
    p = np.pad(a, 1, mode="edge")
    return (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4*a) / dx**2


def gaussian_kernel(n: int, dx: float, component: Component) -> np.ndarray:
    """Dimensionless spatial influence field, peak at component centre."""
    xs = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(xs, xs)
    r = np.sqrt((xx-component.x_m)**2 + (yy-component.y_m)**2)
    # Smooth Gaussian that is practically zero beyond the stated radius.
    sigma = max(component.radius_m / 2.0, dx / 2.0)
    k = np.exp(-0.5 * (r/sigma)**2)
    k[r > 2.5*component.radius_m] = 0.0
    return k


def weather(hour: float):
    """Synthetic Phoenix-like summer day, for code testing—not measured data."""
    # Smooth daily cycle: warm overnight, hot afternoon.
    phase = 2*np.pi*(hour - 9.0)/24.0
    air_c = 36.0 + 7.0*np.sin(phase)  # ~29 to 43 C
    daylight = max(0.0, np.sin(np.pi*(hour-6.0)/13.0)) if 6.0 <= hour <= 19.0 else 0.0
    solar_w_m2 = 900.0 * daylight
    rh_pct = 22.0 - 7.0*np.sin(phase)  # illustrative background RH
    rh_pct = float(np.clip(rh_pct, 8.0, 45.0))
    wind_m_s = 2.0
    return air_c, solar_w_m2, rh_pct, wind_m_s


def saturation_vapor_pressure_pa(temp_c: np.ndarray) -> np.ndarray:
    """Magnus approximation over liquid water."""
    return 610.94 * np.exp(17.625*temp_c/(temp_c+243.04))


def humidity_ratio_from_rh(temp_c: float, rh_pct: float, pressure_pa: float = 101325.0) -> float:
    e = (rh_pct/100.0) * float(saturation_vapor_pressure_pa(np.array(temp_c)))
    return 0.622*e/(pressure_pa-e)


def rh_from_humidity_ratio(temp_c: np.ndarray, w: np.ndarray, pressure_pa: float = 101325.0) -> np.ndarray:
    e = pressure_pa*w/(0.622+w)
    rh = 100.0*e/saturation_vapor_pressure_pa(temp_c)
    return np.clip(rh, 0.0, 100.0)


class OasisSimulator:
    def __init__(self, params=None, components=None):
        self.p = params or Parameters()
        self.components = components or []
        self.n = self.p.n
        self.area = self.p.dx**2
        # Start near the background air conditions at hour 0.
        ta0, _, rh0, _ = weather(0.0)
        self.ta = np.full((self.n, self.n), ta0, dtype=float)  # air temp, C
        self.ts = self.ta + 3.0                                  # surface temp, C
        self.w = np.full((self.n, self.n), humidity_ratio_from_rh(ta0, rh0), dtype=float)
        self.albedo = np.full((self.n, self.n), self.p.base_albedo, dtype=float)
        self.shade = np.zeros((self.n, self.n), dtype=float)
        self.mist = np.zeros((self.n, self.n), dtype=float)
        self.power_w = 0.0
        self.water_kg_s = 0.0
        self._prepare_components()

    def _prepare_components(self):
        for c in self.components:
            k = gaussian_kernel(self.n, self.p.dx, c)
            if c.kind in ("canopy", "tree"):
                self.shade = np.maximum(self.shade, np.clip(c.strength*k, 0.0, 1.0))
            elif c.kind == "reflective_pavement":
                self.albedo = self.albedo*(1-k) + np.clip(c.strength, 0.0, 0.9)*k
            elif c.kind == "mister":
                self.mist += np.clip(c.strength, 0.0, 2.0)*k
            else:
                raise ValueError(f"Unknown component kind: {c.kind}")

    def step(self, hour: float):
        p = self.p
        ta_bg, solar, rh_bg, wind = weather(hour)
        w_bg = humidity_ratio_from_rh(ta_bg, rh_bg)
        # Solar flux absorbed by the surface (W/m2).
        q_solar = (1.0-self.albedo) * solar * (1.0-self.shade)
        # Exchange between surface and near-ground air (W/m2); positive warms surface.
        q_surface_air = p.surface_air_h*(self.ta-self.ts)
        # Evaporative cooling is limited by the configured mist water rate.
        water_flux = self.mist * p.mist_water_kg_m2_s  # kg/(m2 s), assumed evaporated
        q_evap = water_flux * p.latent_heat            # W/m2 removed from surface/near-ground layer
        # Surface state: energy balance + effective lateral heat spreading.
        q_longwave = p.surface_radiative_h*(ta_bg-self.ts)  # linearized net longwave exchange
        dts = (q_solar + q_surface_air + q_longwave)/p.surface_heat_capacity
        dts += p.surface_diffusivity*laplacian(self.ts, p.dx)
        ts_new = self.ts + p.dt*dts
        # Near-ground air energy balance: surface exchange, background ventilation,
        # evaporation cooling and horizontal mixing. Effective 2 m air layer.
        air_capacity_per_area = p.air_density*p.air_cp*p.air_mixing_height
        q_air_from_surface = -q_surface_air
        q_air_vent = air_capacity_per_area*(ta_bg-self.ta)/p.ventilation_tau
        q_air_evap = q_evap
        dta = (q_air_from_surface + q_air_vent - q_air_evap)/air_capacity_per_area
        dta += p.air_diffusivity*laplacian(self.ta, p.dx)
        ta_new = self.ta + p.dt*dta
        # Moisture balance in humidity ratio (kg water / kg dry air).
        # Assume the configured evaporated water enters the represented air layer.
        dry_air_mass_per_area = p.air_density*p.air_mixing_height
        dw = water_flux/dry_air_mass_per_area + (w_bg-self.w)/p.ventilation_tau
        dw += p.moisture_diffusivity*laplacian(self.w, p.dx)
        w_new = np.maximum(self.w + p.dt*dw, 0.0)
        # Keep states finite and in a broad numerical safety range.
        self.ts = np.clip(ts_new, -20.0, 90.0)
        self.ta = np.clip(ta_new, -20.0, 65.0)
        self.w = np.clip(w_new, 0.0, 0.05)
        self.power_w = sum(250.0 for c in self.components if c.kind == "mister")
        self.water_kg_s = float(np.sum(water_flux)*self.area)
        return ta_bg, solar, rh_bg, wind

    def run(self):
        p = self.p
        steps = int(p.duration_hours*3600/p.dt)
        record_every = max(1, int(600/p.dt))  # output every ~10 minutes
        history = {"hour": [], "air_mean": [], "air_max": [], "surface_mean": [],
                   "rh_mean": [], "rh_max": [], "water_l": [], "net_power_w": []}
        snapshots = []
        for step in range(steps):
            hour = step*p.dt/3600.0
            self.step(hour)
            if step % record_every == 0 or step == steps-1:
                rh = rh_from_humidity_ratio(self.ta, self.w)
                history["hour"].append(hour)
                history["air_mean"].append(float(np.mean(self.ta)))
                history["air_max"].append(float(np.max(self.ta)))
                history["surface_mean"].append(float(np.mean(self.ts)))
                history["rh_mean"].append(float(np.mean(rh)))
                history["rh_max"].append(float(np.max(rh)))
                snapshots.append((hour, self.ta.copy(), rh.copy(), self.ts.copy()))
        # Cumulative resource use from constant assumed operation.
        hours = np.array(history["hour"])
        total_seconds = steps*p.dt
        total_water_l = self.water_liters_for_seconds(total_seconds)
        energy_kwh = self.power_w*total_seconds/3.6e6
        history["water_l"] = [total_water_l*(h/p.duration_hours) for h in hours]
        history["net_power_w"] = [self.net_power_w()]*len(hours)
        return history, snapshots, total_water_l, energy_kwh

    def water_liters_for_seconds(self, seconds):
        # Total mist flux (kg/m2/s) times cell area and duration; 1 kg water ~ 1 L.
        return float(np.sum(self.mist*self.p.mist_water_kg_m2_s)*self.area*seconds)

    def net_power_w(self):
        # Prototype: misting station draw; no solar-electric generation modeled yet.
        return self.power_w


def plot_results(sim, history, snapshots, save_path="oasis_results.png"):
    p = sim.p
    # Pick a midday snapshot nearest 14:00.
    snap = min(snapshots, key=lambda s: abs(s[0]-14.0))
    hour, air_map, rh_map, surface_map = snap
    extent = [0, p.n*p.dx, 0, p.n*p.dx]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    im0 = axes[0, 0].imshow(air_map, origin="lower", extent=extent, cmap="inferno")
    axes[0, 0].set_title(f"Outdoor air temperature at {hour:.1f} h (°C)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)
    im1 = axes[0, 1].imshow(rh_map, origin="lower", extent=extent, cmap="YlGnBu", vmin=0, vmax=100)
    axes[0, 1].set_title("Relative humidity (%)")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)
    im2 = axes[1, 0].imshow(surface_map, origin="lower", extent=extent, cmap="magma")
    axes[1, 0].set_title("Surface temperature (°C)")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)
    axes[1, 1].plot(history["hour"], history["air_mean"], label="Mean air temperature")
    axes[1, 1].plot(history["hour"], history["surface_mean"], label="Mean surface temperature")
    axes[1, 1].set_xlabel("Hour of simulated day")
    axes[1, 1].set_ylabel("Temperature (°C)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    for ax in axes.flat:
        if ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
    fig.suptitle("Project Oasis — prototype output (synthetic weather)", fontsize=14)
    fig.savefig(save_path, dpi=160)
    plt.show()


def main():
    p = Parameters()
    # Example configuration: a shaded canopy and a misting station.
    # Coordinates are metres from the lower-left corner of the 80 m x 80 m domain.
    components = [
        Component("canopy", x_m=40, y_m=40, radius_m=12, strength=0.85),
        Component("mister", x_m=40, y_m=40, radius_m=8, strength=1.0),
        Component("reflective_pavement", x_m=20, y_m=20, radius_m=10, strength=0.55),
    ]
    sim = OasisSimulator(p, components)
    history, snapshots, water_l, energy_kwh = sim.run()
    plot_results(sim, history, snapshots)
    print("\nProject Oasis prototype summary")
    print(f"Domain: {p.n*p.dx:.0f} m x {p.n*p.dx:.0f} m; cell size: {p.dx:.1f} m")
    print(f"Simulated duration: {p.duration_hours:.1f} h; time step: {p.dt:.0f} s")
    print(f"Final mean outdoor air temperature: {history['air_mean'][-1]:.2f} °C")
    print(f"Final mean relative humidity: {history['rh_mean'][-1]:.1f} %")
    print(f"Estimated water use: {water_l:.1f} L/day (assumed evaporation rate)")
    print(f"Estimated electricity use: {energy_kwh:.2f} kWh/day (placeholder 250 W per mister)")
    print("NOTE: Synthetic forcing and uncalibrated parameters; not a real Phoenix forecast.")


if __name__ == "__main__":
    main()
