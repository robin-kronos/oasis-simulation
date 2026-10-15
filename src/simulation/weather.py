from dataclasses import dataclass
import numpy as np

@dataclass
class Weather:
    temperature: float  # °C
    solar: float        # W/m²
    humidity: float     # Relative humidity, 0–1
    wind_speed: float   # m/s

    @classmethod
    def at_time(cls, hour):
        """Generate example weather for time t in hour."""



        # Solar radiation: sunrise 6, sunset 18
        solar = 700 * max(
            0.0, np.sin(np.pi * (hour - 6) / 12)
        )

        # Air temperature: peak around 15:00
        temperature = 30 + 7 * np.cos(
            2 * np.pi * (hour - 15) / 24
        )

        # Humidity: higher at night
        humidity = 0.60 - 0.20 * np.cos(
            2 * np.pi * (hour - 15) / 24
        )

        return temperature, solar