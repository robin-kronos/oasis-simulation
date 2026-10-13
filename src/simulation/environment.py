import numpy as np

class environment: 
    def __init__(self, gridSize):
        self.grid = np.array([np.array([0 for x in range(gridSize)]) for y in range(gridSize)])

    def toString(self):
        print(self.grid)