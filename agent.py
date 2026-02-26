import os 
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
                 Env:env.SnakeEnv,
                 nEpsilon:int=100,
                 Epsilon1:float=0.99,
                 Epsilon2:float=0.01,
                 LR:float=1e-3,
                 Save:bool=True,
                 Load:bool=True):
        self.LR =LR
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
        
        self.LR = LR
        self.Save = Save
        self.Load =Load

    def CreateModel(self,
                    nDense:list=[512,256],
                    Activation:str="gelu"):
        self.nDense = nDense
        self.Activation = getattr(act,Activation)
        if os.path.exists('Model') and self.Load:
            self.Model = mod.load_model('Model')
        else:
            self.Model = mod.Sequential()
            self.Model.add(lay.InputLayer(input_tensor=self.InputShape))
            for i in nDense:
                self.Model.add(lay.Dense(units=i,activation=self.Activation))
            self.Model.add(lay.Dense(units=self.Env.nAction, activation=getattr(act,'linear')))
        
    def CompileModel(self,
                     Optimizer:str='Adam',
                     Loss:str='MSE'):
        Optimizer = Optimizer.lower()
        Loss = Loss.lower()
        if Optimizer == 'adam':
            self.Optimizer = opt.Adam(learning_rate=self.LR)
        elif Optimizer == 'sgd':
            self.Optimizer = opt.SGD(learning_rate=self.LR,
                                     momentum=0.9)
        elif Optimizer == 'rmsprop':
            self.Optimizer = opt.RMSprop(learning_rate=self.LR)
        elif Loss in ['mse','mean squared error','mean_squared_error']:
            self.Loss = los.MeanSquaredError()
        elif Loss in ['mae','mean absolute error','mean_absolute_error']:
            self.Loss = los.MeanAbsoluteError()
        elif Loss in 'huber':
            self.Loss = los.Huber()
        self.Model.compile(optimizer=self.Optimizer,
                           loss=self.Loss)
    
    def SummaryModel(self):
        print(60 * '_')
        print('Model Summary')
        self.Model.summary()
        print(60 * '_')
    def SaveModel(self):
        self.Model.save(filepath='Model')
        print('Model Saved Succesfully.')
    def PredictQ(self,
                 X:np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            Q = self.Model.predict(np.expand_dims(X, axis=0),
                                   verbose=0)[0]
        else:
            Q = self.Model.predict(X,
                                   verbose=0)
        return Q