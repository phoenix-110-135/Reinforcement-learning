import numpy as np 
import matplotlib.pyplot as plt 


class SnakeEnv:
    def __init__(self, H:int, W:int):
        self.H  = H
        self.W = W
    def Reset(self):
        self.Map = np.zeros((self.H,self.W),dtype=np.int32)