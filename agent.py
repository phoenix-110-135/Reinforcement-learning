import os as os
import Env as env
import numpy as np
import tensorflow as tf
import collections as col
import keras.models as mod
import keras.layers as lay
import keras.losses as los
import keras.optimizers as opt
import keras.activations as act
import matplotlib.pyplot as plt

class SnakeAgent:
    def __init__(self,
                 Env:env.SnakeEnv,
                 Hv:int,
                 Wv:int,
                 nEpisode:int=100,
                 mStep:int=128,
                 Epsilon1:float=0.99, 
                 Epsilon2:float=0.01,
                 Temperature1:float=10,
                 Temperature2:float=0.1,
                 LR:float=1e-3,
                 Gamma:float=0.95,
                 sMemory:int=1024,
                 sBatch:int=64,
                 TrainOn:int=32,
                 Save:bool=True,
                 Load:bool=True,
                 trShow:bool=True,
                 teShow:bool=True):
        self.Env = Env
        self.Hv = Hv
        self.Wv = Wv
        self.sState = 4 * ((2 * Hv + 1) * (2 * Wv + 1) - 1) + 2 + 2
        self.InputShape = (self.sState, )
        self.nEpisode = nEpisode
        self.mStep = mStep
        self.Epsilon1 = Epsilon1
        self.Epsilon2 = Epsilon2
        self.Epsilons = np.linspace(start=Epsilon1,
                                    stop=Epsilon2,
                                    num=nEpisode)
        self.Temperature1 = Temperature1
        self.Temperature2 = Temperature2
        self.Temperatures = np.logspace(start=np.log(Temperature1),
                                        stop=np.log(Temperature2),
                                        num=nEpisode,
                                        base=np.e)
        self.LR = LR
        self.Gamma = Gamma
        self.sMemory = sMemory
        self.Memory = col.deque([], maxlen=sMemory)
        self.sBatch = sBatch
        self.TrainOn = TrainOn
        self.Counter = 0
        self.Save = Save
        self.Load = Load
        self.trShow = trShow
        self.teShow = teShow
        self.Episode = -1
        self.Step = -1
        self.MinReward = min(list(Env.Rewards.values()))
        self.MaxReward = max(list(Env.Rewards.values()))
        self.MinQ = 1.2 * self.MinReward * (1 + self.Gamma)
        self.MaxQ = 1.2 * self.MaxReward * (1 + self.Gamma)
        self.ActionLog = []
        self.EpisodeLog = np.zeros(shape=(nEpisode, ))
        self.Training = False
        self.SaveOn = 4
        self.SaveOnCounter = 0
    def CreateModel(self,
                    nDense:list=[512, 256],
                    Activation:str='gelu'):
        self.nDense = nDense
        self.Activation = getattr(act, Activation)
        if os.path.exists('Model') and self.Load:
            self.Model = mod.load_model('Model.keras')
        else:
            self.Model = mod.Sequential()
            self.Model.add(lay.InputLayer(input_shape=self.InputShape))
            for i in nDense:
                self.Model.add(lay.Dense(units=i,
                                        activation=self.Activation))
            self.Model.add(lay.Dense(units=self.Env.nAction,
                                    activation=getattr(act, 'linear')))
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
        if Loss in ['mse', 'mean squared error', 'mean_squared_error']:
            self.Loss = los.MeanSquaredError()
        elif Loss in ['mae', 'mean absolute error', 'mean_absolute_error']:
            self.Loss = los.MeanAbsoluteError()
        elif Loss in 'huber':
            self.Loss = los.Huber()
        self.Model.compile(optimizer=self.Optimizer,
                           loss=self.Loss)
    def SummaryModel(self):
        print(60 * '_')
        print('Model Sumary: ')
        self.Model.summary()
        print(60 * '_')
    def SaveModel(self):
        self.Model.save(filepath='Model.keras')
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
    def GetState(self) -> np.ndarray:
        State = []
        for i in range(-self.Hv, self.Hv + 1):
            for j in range(-self.Wv, self.Wv + 1):
                if i != 0 or j != 0:
                    x = self.Env.Object2Code['Out']
                    if (0 <= self.Env.Head[0] + i < self.Env.H) and (0 <= self.Env.Head[1] + j < self.Env.W):
                        x = self.Env.Map[self.Env.Head[0] + i, self.Env.Head[1] + j]
                    for k in self.Env.t1:
                        if x == k:
                            State.append(+1)
                        else:
                            State.append(-1)
        State.append(2 * self.Env.Head[0] / (self.Env.H - 1) - 1)
        State.append(2 * self.Env.Head[1] / (self.Env.W - 1) - 1)
        State.append(2 * self.Env.Food[0] / (self.Env.H - 1) - 1)
        State.append(2 * self.Env.Food[1] / (self.Env.W - 1) - 1)
        State = np.array(State)
        return State
    def Decide(self,
               Policy:str) -> int:
        if Policy == 'R':
            Action = np.random.randint(low=0,
                                       high=self.Env.nAction)
        elif Policy == 'G':
            Q = self.PredictQ(self.State)
            Action = np.argmax(Q)
        elif Policy == 'EG':
            r = np.random.rand()
            if r < self.Epsilon:
                Action = self.Decide('R')
            else:
                Action = self.Decide('G')
        elif Policy == 'B':
            Q = self.PredictQ(self.State)
            Q = Q / self.Temperature
            Q[Q > 20] = 20
            Q = np.exp(Q)
            P = Q / Q.sum()
            Action = np.random.choice(a=self.Env.nAction, p=P)
        return Action
    def NextEpisode(self):
        self.Episode += 1
        self.Step = -1
        self.Epsilon = self.Epsilons[self.Episode]
        self.Temperature = self.Temperatures[self.Episode]
        self.Env.Reset()
    def NextStep(self):
        self.Step += 1
        if self.Training and self.trShow:
            self.Env.OnlineShow()
        elif not self.Training and self.teShow:
            self.Env.OnlineShow()
    def PlotEpsilons(self):
        Episodes = np.arange(start=1,
                             stop=self.nEpisode + 1,
                             step=1)
        plt.plot(Episodes,
                 self.Epsilons,
                 ls='-',
                 lw=1,
                 c='crimson')
        plt.title('Epsilon Value Over Training Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.show()
    def PlotTemperatures(self):
        Episodes = np.arange(start=1,
                             stop=self.nEpisode + 1,
                             step=1)
        plt.plot(Episodes,
                 self.Temperatures,
                 ls='-',
                 lw=1,
                 c='crimson')
        plt.title('Temperature Value Over Training Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Temperature')
        plt.show()
    def Save2Memory(self,
                    Action:int,
                    Reward:float,
                    State2:np.ndarray,
                    Done:bool):
        self.Memory.append([self.State, Action, Reward, State2, Done])
        self.Counter += 1
        if self.Counter == self.TrainOn:
            self.TrainModel()
            self.Counter = 0
    def TrainModel(self):
        nD = len(self.Memory)
        Selecteds = np.random.choice(a=nD,
                                     size=min(nD, self.sBatch),
                                     replace=False)
        X = []
        Y = []
        for i in Selecteds:
            s1, a, r, s2, done = self.Memory[i]
            X.append(s1)
            q1 = self.PredictQ(s1)
            if done:
                q1[a] = r
                Y.append(q1)
            else:
                q2 = self.PredictQ(s2)
                q1[a] = r + self.Gamma * q2.max()
                Y.append(q1)
        X = np.array(X, dtype=np.float16)
        Y = np.array(Y, dtype=np.float16)
        Y = np.clip(Y, self.MinQ, self.MaxQ)
        self.Model.train_on_batch(x=X,
                                  y=Y)
        print('Model Trained On Batch Succesfully.')
        self.SaveOnCounter += 1
        if self.SaveOnCounter == self.SaveOn:
            self.SaveModel()
            self.SaveOnCounter = 0
    def Do(self,
           Action:int) -> tuple[float,
                                np.ndarray,
                                bool]:
        Reward = 0
        Done = False
        h0, w0 = self.Env.Head
        dh, dw = self.Env.Code2Trans[Action]
        h, w = h0 + dh, w0 + dw
        Distance1 = np.linalg.norm(self.Env.Head - self.Env.Food, ord=2)
        if (0 <= h < self.Env.H) and (0 <= w < self.Env.W):
            if len(self.Env.Snake) == 0:
                if self.Env.Map[h, w] == self.Env.Object2Code['Free']:
                    self.Env.Head = np.array([h, w])
                    self.Env.Map[h0, w0] = self.Env.Object2Code['Free']
                    self.Env.Map[h, w] = self.Env.Object2Code['Head']
                    Distance2 = np.linalg.norm(self.Env.Head - self.Env.Food, ord=2)
                    if Distance2 < Distance1:
                        print('Free +')
                        Reward += self.Env.Rewards['Closer']
                    else:
                        print('Free -')
                        Reward -= self.Env.Rewards['Closer']
                elif self.Env.Map[h, w] == self.Env.Object2Code['Food']:
                    print('Food')
                    self.Env.Head = np.array([h, w])
                    self.Env.Map[h, w] = self.Env.Object2Code['Head']
                    self.Env.Map[h0, w0] = self.Env.Object2Code['Snake']
                    self.Env.Snake.append(np.array([h0, w0]))
                    self.Env.ResetFood()
                    Reward += self.Env.Rewards['Closer']
                    Reward += self.Env.Rewards['Food']
            else:
                if self.Env.Map[h, w] == self.Env.Object2Code['Free']:
                    self.Env.Map[h, w] = self.Env.Object2Code['Head']
                    self.Env.Map[h0, w0] = self.Env.Object2Code['Snake']
                    self.Env.Map[self.Env.Snake[-1][0], self.Env.Snake[-1][1]] = self.Env.Object2Code['Free']
                    self.Env.Head = np.array([h, w])
                    self.Env.Snake = [np.array([h0, w0])] + self.Env.Snake[:-1]
                    Distance2 = np.linalg.norm(self.Env.Head - self.Env.Food, ord=2)
                    if Distance2 < Distance1:
                        print('Free +')
                        Reward += self.Env.Rewards['Closer']
                    else:
                        print('Free -')
                        Reward -= self.Env.Rewards['Closer']
                elif self.Env.Map[h, w] == self.Env.Object2Code['Snake']:
                    if (np.array([h, w]) == self.Env.Snake[0]).all():
                        print('Reverse')
                        Reward += self.Env.Rewards['Reverse']
                    else:
                        print('Snake')
                        Reward += self.Env.Rewards['Snake']
                        Done = True
                elif self.Env.Map[h, w] == self.Env.Object2Code['Food']:
                    print('Food')
                    self.Env.Map[h, w] = self.Env.Object2Code['Head']
                    self.Env.Map[h0, w0] = self.Env.Object2Code['Snake']
                    self.Env.Head = np.array([h, w])
                    self.Env.Snake = [np.array([h0, w0])] + self.Env.Snake
                    self.Env.ResetFood()
                    Reward += self.Env.Rewards['Closer']
                    Reward += self.Env.Rewards['Food']
        else:
            print('Out')
            Reward += self.Env.Rewards['Out']
            Done = True
        State2 = self.GetState()
        return Reward, State2, Done
    def Train(self,
              Policy:str):
        self.Training = True
        for i in range(self.nEpisode):
            print(60 * '_')
            self.NextEpisode()
            print(f'Episode {i + 1}: [Epsilon: {self.Epsilon:.2f}] - [Temperature: {self.Temperature:.2f}]')
            self.State = self.GetState()
            while True:
                self.NextStep()
                Action = self.Decide(Policy)
                Reward, State2, Done = self.Do(Action)
                self.ActionLog.append(Reward)
                self.EpisodeLog[i] += Reward
                self.Save2Memory(Action, Reward, State2, Done)
                self.State = State2
                if Done or self.Step == self.mStep - 1:
                    break
            print(f'Total Reward: {self.EpisodeLog[i]:.0f}')
        self.ActionLog = np.array(self.ActionLog,
                                  dtype=np.float16)
        plt.close()
        self.Training = False
        self.SaveModel()
        self.SaveOnCounter = 0
    def Test(self,
             Policy:str,
             nEpisode:int):
        self.Epsilon = self.Epsilons[-1]
        self.Temperature = self.Temperatures[-1]
        for i in range(nEpisode):
            print(60 * '_')
            print(f'Test Episode {i + 1}: [Epsilon: {self.Epsilon:.2f}] - [Temperature: {self.Temperature:.2f}]')
            self.Step = -1
            self.Env.Reset()
            self.State = self.GetState()
            TotalReward = 0
            while True:
                self.NextStep()
                Action = self.Decide(Policy)
                Reward, State2, Done = self.Do(Action)
                TotalReward += Reward
                self.State = State2
                if Done or self.Step == self.mStep - 1:
                    break
            print(f'Total Reward: {self.EpisodeLog[i]:.0f}')
        plt.close()
    def SMA(self,
            S:np.ndarray,
            L:int) -> np.ndarray:
        nD0 = S.size
        nD = nD0 - L + 1
        M = np.zeros(shape=(nD, ),
                     dtype=np.float32)
        for i in range(nD):
            M[i] = S[i:i + L].mean()
        return M
    def PlotActionLog(self,
                      L:int):
        Actions = np.arange(start=1,
                            stop=self.ActionLog.size + 1,
                            step=1)
        M = self.SMA(self.ActionLog, L)
        plt.plot(Actions,
                 self.ActionLog,
                 ls='-',
                 lw=1,
                 c='teal',
                 label='Reward')
        plt.plot(Actions[-M.size:],
                 M,
                 ls='-',
                 lw=1.2,
                 c='crimson',
                 label=f'SMA({L})')
        plt.title('Agent Actions Reward Over Training Episodes')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
    def PlotEpisodeLog(self,
                       L:int):
        Episodes = np.arange(start=1,
                             stop=self.EpisodeLog.size + 1,
                             step=1)
        M = self.SMA(self.EpisodeLog, L)
        plt.plot(Episodes,
                 self.EpisodeLog,
                 ls='-',
                 lw=1,
                 c='teal',
                 label='Reward')
        plt.plot(Episodes[-M.size:],
                 M,
                 ls='-',
                 lw=1.2,
                 c='crimson',
                 label=f'SMA({L})')
        plt.title('Agent Episodes Reward Over Training Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
