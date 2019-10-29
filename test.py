from solver import Solver
from problems import Mueller2d, LLGC, DoubleWell1D
import torch as pt
from utilities import do_importance_sampling, plot_solution_for_DoubleWell1d

#problem = Mueller2d(T=1.0)
problem = DoubleWell1D(T=1.0, delta_t = 0.005)
#problem = LLGC(d=2)
sol = Solver('test', problem, K=100, L=500, time_approx='outer',approx_method='control',lr = 0.05, loss_method='cross_entropy')
#pt.manual_seed(42)
#sol.train_LSE_with_reference()
#sol.train()

sol.load_networks('output/test_2019-10-29.pt')

plot_solution_for_DoubleWell1d(sol, 'output/sol_doublewell1d.eps')

#do_importance_sampling(problem, sol, 50000, control='true', verbose=True, delta_t=0.005)
