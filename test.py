from solver import Solver
from problems import Mueller2d, LLGC
import torch as pt

problem = Mueller2d()
#problem = LLGC(d=2)
sol = Solver('test', problem, K=5)
sol.train()

