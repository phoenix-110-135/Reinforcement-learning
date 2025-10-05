import numpy as np 
import matplotlib.pyplot
import tensorflow as tf 
import keras.models as mod 
import tensorflow as tf
import keras.layers as lay 
import keras.activations as act 
import keras.optimizers as opt 
import keras.losses as los 
import Env as env 
class SnakeAgent:
    def __init__(self,
                 Hv:int,
                 Wv:int,
                 Env:env.Env,
                 nEpsilon:int=100,
                 Epsilon1:float=0.99,
                 Epsilon2:float=0.01):
        self.Hv = Hv
        self.Wv = Wv
        self.sState = 4 * ((2 * Hv + 1) * (2 * Wv + 1) -1) + 2 + 2 
        self.InputShape = (self.sState,)
        self.Env = Env 
        self.nEpsilon = nEpsilon
        self.Epsilon1 = Epsilon1
        self.Epsilon2 = Epsilon2
        self.Epsilons = np.linspace(start=Epsilon1,
                                    stop=Epsilon2,
                                    num=nEpsilon)
    def CreateModel(self,
                    nDense:list=[512,256],
                    Activation:str="gelu"):
        self.nDense = nDense
        self.Activation = getattr(act,Activation)
        self.Model = mod.Sequential()
        self.Model.add(lay.InputLayer(input_tensor=self.InputShape))
        for i in nDense:
            self.Model.add(lay.Dense(units=i,activation=self.Activation))
        self.Model.add(lay.Dense(units=self.Env.nAction, activation=getattr(act,'linear')))