import Env as en
import Agent as ag

Env = en.SnakeEnv(6, 8)

Agent = ag.SnakeAgent(Env,
                      2,
                      2,
                      nEpisode=400,
                      mStep=150,
                      sMemory=1024,
                      sBatch=64,
                      Temperature1=10,
                      Temperature2=1.5,
                      TrainOn=32)

Agent.CreateModel(nDense=[1024], Activation='leaky_relu')

Agent.CompileModel()

Agent.SummaryModel()

# Agent.PlotEpsilons()
# Agent.PlotTemperatures()

# Agent.Train('B')

# Agent.PlotActionLog(400)

# Agent.PlotEpisodeLog(10)

Agent.Test('G', 5)