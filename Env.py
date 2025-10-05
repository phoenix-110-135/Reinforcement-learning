import numpy as np 
import matplotlib.pyplot  as plt  
from matplotlib.colors import  ListedColormap  


class SnakeEnv:
    def __init__(self, H:int, W:int):
        self.H  = H
        self.W = W
        self.object2code = {'Free':0,
                            'Snake':1,
                            'Food':2,
                            'Head':3,
                            'Out':4}
        self.Action2Code = {'Up':0,
                            'Right':1,
                            'Down':2,
                            'Left':3}
        self.Action2Trans = {"Up":np.array([-1,0]),
                             "Right":np.array([0,+1]),
                             "Down":np.array([+1,0]),
                             "Left":np.array([0,-1])}
        
        self.nAction = len(self.Action2Code)
        self.Reset()


    def Reset(self):
        self.Map = np.zeros((self.H,self.W),dtype=np.int8)
        self.ResetFood()
        self.ResetSnake()

    def ResetFood(self):
        m = self.Map == self.object2code['Food']
        self.Map[m] = self.object2code['Free']
        h = np.random.randint(low=0, high=self.H)
        w = np.random.randint(low=0, high=self.W)
        while self.Map[h,w] != self.object2code['Free']:
            h = np.random.randint(low=0, high=self.H)
            w = np.random.randint(low=0, high=self.W)
        self.Map[h,w] = self.object2code['Food']
        self.Food = np.array([w,h])

    def ResetSnake(self):
        m1 = self.Map == self.object2code['Snake']
        m2 = self.Map == self.object2code['Head']
        self.Map[m1] = self.object2code['Free']
        self.Map[m2] = self.object2code['Free']
        h = np.random.randint(low=0, high=self.H)
        w = np.random.randint(low=0, high=self.W)
        while self.Map[h,w] != self.object2code['Free']:
            h = np.random.randint(low=0, high=self.H)
            w = np.random.randint(low=0, high=self.W)
        self.Map[h,w] = self.object2code['Head']
        self.Head = np.array([w,h])
        self.Snake = []

    def show(self):
        colors = ['black', 'orange','yellow','purple']  

        cmap = ListedColormap(colors)

        plt.imshow(self.Map, cmap=cmap)
        plt.show()

Env = SnakeEnv(9,14)
Env.show()

print("end")