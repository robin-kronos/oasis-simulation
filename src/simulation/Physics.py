import numpy as np

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant
RHO = 1.2               # Air density, kg/m³
CP = 1005.0             # Air heat capacity, J/(kg K)
LV = 2.45e6             # Latent heat of vaporization, J/kg
P = 101325.0            # Atmospheric pressure, Pa

#TODO are these actual good values?
h = 10.0  # Air layer height, m
cs = 200000.0  # Surface heat capacity, J/(m² K)


G_COEFFICIENTS = { ##Todo make this less dik where this came from
    0: 0.70,  # Asphalt: simplified OHM
    1: 0.10,  # Vegetation: FAO-56 daytime
    2: 0.10,  # Wet soil: provisional assumption
    3: 0.26,  # Gravel roof: simplified OHM
}

def diffuse_air(ta, dx, dt=2.0, K=0.25):
    """
    Explicit horizontal temperature diffusion.

    ta: 2D temperature field (°C)
    dx: grid spacing (m)
    dt: diffusion timestep (s)
    K: turbulent thermal diffusivity (m²/s)

    Uses zero-flux boundaries.
    """

    if dx <= 0 or dt <= 0 or K < 0:
        raise ValueError("Invalid diffusion parameters")

    if K * dt / dx**2 > 0.25:
        raise ValueError(
            "Diffusion timestep too large for stability"
        )

    padded = np.pad(ta, 1, mode="edge")

    laplacian = (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * ta
    ) / dx**2

    return ta + K * dt * laplacian

def calculate_G(rn, surface_type):
    return G_COEFFICIENTS[surface_type] * rn

def saturation_pressure(T):
    return 610.8 * np.exp(17.27 * T / (T + 237.3))

def penman_monteith(ta, w, rn, g, ra, rs):
    es = saturation_pressure(ta)
    ea = P * w / (0.62198 + w)
    vpd = max(0.0, es - ea)
    # Slope of saturation vapor pressure curve, Pa/K
    delta = 4098.0 * es / (ta + 237.3)**2
    # Psychrometric constant, Pa/K
    gamma = CP * P / (0.622 * LV)

    le = (
        delta * (rn - g)
        + RHO * CP * vpd / ra
    ) / (
        delta + gamma * (1.0 + rs / ra)
    )

    return max(0.0, le)

def ra_water(w, zm=2.0, z0=0.0002):
    return 4.72 * np.log(zm / z0)**2 / (1 + 0.536 * w)

def calcDelta(T_city, I, state, map):
    windspeed =5 ##assumption idk if we implement this
    mapshape = map.surfacetype.shape
    dta = np.zeros(mapshape)
    dts = np.zeros(mapshape)
    dw = np.zeros(mapshape)
    emissivity = 0.95 ## assumption for city for now

    ldown = 0.8 * SIGMA * (T_city + 273.15) ** 4 ##radiation reflectivity atmospheer

    for (i, j), value in np.ndenumerate(map.surfacetype):
        shade = np.clip(map.shade_fraction[i, j], 0.0, 1.0)
        I_effective = I * (1.0 - shade)
        rn = I_effective * (1 - map.albedo[i, j]) + emissivity * ldown - emissivity * SIGMA * (
                    state.ts[i, j] + 273.15) ** 4  # Net Radiation
        g =  calculate_G(rn,map.surfacetype[i, j])
        ra = 100 #TODO no actual valuefor asphalt and roof
        le = 0.0
        match map.surfacetype[i, j]:
                case 0:
                    pass
                    #todo asphalt

                case 1:
                    #todo vegatation
                    ra = 208/windspeed
                    le = penman_monteith(state.ta[i,j], state.w[i,j], rn,g, ra, 70.0)  ## rs = 0 because water
                case 2:
                    ra = ra_water()
                    le = penman_monteith(state.ta[i,j], state.w[i,j], rn, g, ra, 0.0) ## rs = 0 because water
                case 3:
                    pass
                    #todo roof
        # Sensible heat flux (W/m²)
        H = RHO * CP * (state.ts[i, j] - state.ta[i,j]) / ra

        # Surface temperature change
        dts[i, j] = (rn - H - le - g)  / cs

        # Air temperature change
        dta[i, j] = H  / (RHO * CP * h)

        # Change in humidity ratio
        evaporation = le / LV  # kg/(m² s)
        dw[i, j] = evaporation  / (RHO * h)







    return dta, dts, dw


def saturation_pressure(T):
    return 610.8 * np.exp(17.27 * T / (T + 237.3))


def saturation_humidity_ratio(T):
    es = saturation_pressure(T)
    return 0.62198 * es / np.maximum(P - es, 1.0)


def apply_misting(state, city_map, dt, h=10.0):
    """
    Apply evaporative mist cooling.

    state.ta: air temperature (°C)
    state.w: humidity ratio (kg/kg)
    city_map.mist_rate: potential evaporation (kg/m²/s)
    dt: timestep (s)
    h: air mixing height (m)

    Returns evaporated water mass per area (kg/m²).
    """

    rate = np.maximum(city_map.mist_rate, 0.0)

    # Maximum water available for evaporation
    available = rate * dt

    # Air mass above each square meter
    air_mass = RHO * h

    # Find how much water can evaporate without
    # exceeding saturation after cooling
    low = np.zeros_like(available)
    high = available.copy()

    for _ in range(30):
        evaporated = (low + high) / 2.0

        T_new = (
            state.ta
            - LV * evaporated / (air_mass * CP)
        )

        w_new = state.w + evaporated / air_mass

        can_evaporate = (
            w_new <= saturation_humidity_ratio(T_new)
        )

        low = np.where(can_evaporate, evaporated, low)
        high = np.where(can_evaporate, high, evaporated)

    # Apply local cooling and humidification
    state.ta -= LV * low / (air_mass * CP)
    state.w += low / air_mass

    return low

def apply_ventilation(
    state,
    background_temp,
    background_w,
    dt,
    tau=1800.0
):
    """
    Exchange local air with background atmosphere.

    dt: simulation timestep (s)
    tau: ventilation timescale (s)

    Updates air temperature and humidity.
    """

    if tau <= 0:
        raise ValueError("tau must be positive")

    # Exact solution for first-order relaxation
    exchange = 1.0 - np.exp(-dt / tau)

    state.ta += exchange * (
        background_temp - state.ta
    )

    state.w += exchange * (
        background_w - state.w
    )