from solver import Solver
from problems import Mueller2d, LLGC, DoubleWell1D
import torch as pt
from utilities import do_importance_sampling

#problem = Mueller2d(T=1.0)
problem = DoubleWell1D(T=1.0, delta_t = 0.005)
#problem = LLGC(d=2)
sol = Solver('test', problem, K=5, L=3, time_approx='outer',approx_method='control',loss_method='moment')
#pt.manual_seed(42)
#sol.train()
sol.load_networks('output/test_2019-10-28.pt')

#do_importance_sampling(problem, sol, 50000, control='true', verbose=True, delta_t=0.005)
