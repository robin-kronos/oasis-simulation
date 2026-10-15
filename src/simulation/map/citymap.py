from dataclasses import dataclass
import numpy as np


@dataclass
class CityMap:
    surfacetype: np.ndarray
    albedo: np.ndarray
    shade_fraction: np.ndarray
    mist_rate: np.ndarray


    def __post_init__(self):
        if self.surfacetype.shape != self.albedo.shape:
            raise ValueError(
                "Surfacetype and albedo must have the same shape"
            )

    @classmethod
    def initial(cls, params):
        n = params.n

        return cls(
            surfacetype=np.full(
                (n, n), params.Asphalt, dtype=int
            ),
            albedo=np.full(
                (n, n), 0.12, dtype=float
            ),
            shade_fraction=np.full((n,n), 0, dtype=float),
            mist_rate=np.full((n,n), 0, dtype=float),
        )

    def add_mister(self, row, col, radius, rate=0.000002):
        """
        Add a circular misting area.

        row, col: center position in grid cells
        radius: radius in grid cells
        rate: potential evaporation in kg/m²/s
        """

        rows, cols = np.ogrid[
            :self.surfacetype.shape[0],
            :self.surfacetype.shape[1]
        ]

        distance_squared = (
                (rows - row) ** 2 +
                (cols - col) ** 2
        )

        mask = distance_squared <= radius ** 2

        self.mist_rate[mask] = rate