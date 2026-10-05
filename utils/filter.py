import numpy as np


class CertaintyKalmanFilter:
    def __init__(self, shape: tuple, R: float = 0.5):
        self.R = R
        self.P = np.ones(shape, dtype=np.float32)
        self.x = np.full(shape, 0.5, dtype=np.float32)
    
    def step(self, rows, cols, z: float):
        P = self.P[rows, cols]
        K = P / (P + self.R)
        self.x[rows, cols] += K * (z - self.x[rows, cols])
        self.P[rows, cols] = (1 - K) * P
        return self.x[rows, cols]