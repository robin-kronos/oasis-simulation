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
        rn = I * (1 - map.albedo[i, j]) + emissivity * ldown - emissivity * SIGMA * (
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