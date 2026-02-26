import numpy as np
import matplotlib.pyplot as plt

class SnakeEnv:
    def __init__(self, H:int, W:int):
        self.H = H
        self.W = W
        self.Object2Code = {'Free': 0,
                            'Snake': 1,
                            'Food': 2,
                            'Head': 3,
                            'Out': 4}
        self.t1 = [0, 1, 2, 4]
        self.Action2Code = {'Up': 0,
                            'Right': 1,
                            'Down': 2,
                            'Left': 3}
        self.Code2Trans = {0: np.array([-1, 0]),
                           1: np.array([0, +1]),
                           2: np.array([+1, 0]),
                           3: np.array([0, -1])}
        self.Rewards = {'Food': +12,
                        'Snake': -11,
                        'Out': -10,
                        'Closer': +1.5,
                        'Reverse': -1}
        self.nAction = len(self.Action2Code)
        self.Reset()
    def Reset(self):
        self.Map = np.zeros((self.H, self.W), dtype=np.int8)
        self.ResetFood()
        self.ResetSnake()
    def ResetFood(self):
        m = self.Map == self.Object2Code['Food']
        self.Map[m] = self.Object2Code['Free']
        h = np.random.randint(low=0, high=self.H)
        w = np.random.randint(low=0, high=self.W)
        while self.Map[h, w] != self.Object2Code['Free']:
            h = np.random.randint(low=0, high=self.H)
            w = np.random.randint(low=0, high=self.W)
        self.Map[h, w] = self.Object2Code['Food']
        self.Food = np.array([h, w])
    def ResetSnake(self):
        m1 = self.Map == self.Object2Code['Snake']
        m2 = self.Map == self.Object2Code['Head']
        self.Map[m1] = self.Object2Code['Free']
        self.Map[m2] = self.Object2Code['Free']
        h = np.random.randint(low=0, high=self.H)
        w = np.random.randint(low=0, high=self.W)
        while self.Map[h, w] != self.Object2Code['Free']:
            h = np.random.randint(low=0, high=self.H)
            w = np.random.randint(low=0, high=self.W)
        self.Map[h, w] = self.Object2Code['Head']
        self.Head = np.array([h, w])
        self.Snake = []
    def Show(self):
        plt.imshow(self.Map)
        plt.show()
    def OnlineShow(self):
        plt.cla()
        plt.imshow(self.Map)
        plt.pause(0.05)
