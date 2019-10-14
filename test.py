from solver import Solver
from problems import Mueller2d, LLGC, DoubleWell
import torch as pt
from utilities import do_importance_sampling

problem = DoubleWell(T=1.0, delta_t = 0.005)
#problem = LLGC(d=2)
sol = Solver('test', problem, K=5)
#sol.train()
pt.manual_seed(50)
do_importance_sampling(problem, sol, 50000, control='true', verbose=True, delta_t=0.005)


