import numpy as np


class CertaintyKalmanFilter:
    def __init__(self, shape: tuple, R: float = 0.5):
        self.shape = shape
        self.R = R
        self.reset()
    
    def step(self, rows, cols, z: float, R=None):
        R = self.R if R is None else R
        P = self.P[rows, cols]
        K = P / (P + R)
        self.x[rows, cols] += K * (z - self.x[rows, cols])
        self.P[rows, cols] = (1 - K) * P
        return self.x[rows, cols]
    
    def reset(self):
        self.P = np.ones(self.shape, dtype=np.float32)
        self.x = np.full(self.shape, 0.5, dtype=np.float32)