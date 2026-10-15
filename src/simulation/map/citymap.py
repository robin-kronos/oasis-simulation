from dataclasses import dataclass
import numpy as np


@dataclass
class CityMap:
    surfacetype: np.ndarray
    albedo: np.ndarray

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
            )
        )