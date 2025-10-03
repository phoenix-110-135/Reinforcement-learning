import numpy as np 
import matplotlib.pyplot
import tensorflow as tf 
import keras.models as mod 
import tensorflow as tf
import keras.layers as lay 
import keras.activations as act 
import keras.optimizers as opt 
import keras.losses as los 

class SnakeAgent:
    def __init__(self,
                 nEpsilon:int=100,
                 Epsilon1:float=0.99,
                 Epsilon2:float=0.01):
        self.nEpsilon = nEpsilon
        self.Epsilon1 = Epsilon1
        self.Epsilon2 = Epsilon2
        self.Epsilons = np.linspace(start=Epsilon1,
                                    stop=Epsilon2,
                                    num=nEpsilon)