#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-instance-attributes, not-callable, no-else-return
#pylint: disable=inconsistent-return-statements


from datetime import date
import json
import numpy as np
import os
import time
import torch as pt

from function_space import Linear, NN, SingleParam


device = pt.device('cpu')


# to do:
# - flexible learning rate, line search?
# - automatic logging
# - flexible optimizer


class Solver():

    def __init__(self, name, problem, lr=0.001, L=10000, K=50, delta_t=0.05,
                 loss_method='variance', learn_Y_0=False, adaptive_forward_process=True,
                 early_stopping_time=10000, seed=42, save_results=False):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d
        self.T = problem.T

        # hyperparameters
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size

        # function approximation
        self.Y_0 = SingleParam(lr=self.lr).to(device)
        self.Z_n = [NN(d=self.d, lr=self.lr) for i in range(self.N)]
        #self.Z_n = [Linear(d=self.d, B=problem.B, Q=problem.Q, lr=self.lr).to(device)
        #            for i in range(self.N)]
        self.Y_0.train()
        for z_n in self.Z_n:
            z_n.train()

        # learning properties
        self.loss_method = loss_method
        self.learn_Y_0 = learn_Y_0
        self.adaptive_forward_process = adaptive_forward_process
        self.early_stopping_time = early_stopping_time

        if self.loss_method == 'moment':
            self.learn_Y_0 = True
        if self.loss_method == 'functional':
            self.adaptive_forward_process = True

        # logging
        self.Y_0_log = []
        self.loss_log = []
        self.u_L2_loss = []
        self.times = []

        # printing
        self.print_every = 100
        self.save_results = save_results

    def b(self, x):
        return self.problem.b(x)

    def sigma(self, x):
        return self.problem.sigma(x)

    def h(self, t, x, y, z):
        return self.problem.h(t, x, y, z)

    def g(self, x):
        return self.problem.g(x)

    def u_true(self, x, t):
        return self.problem.u_true(x, t)

    def loss_function(self, X, Y, Z_sum):
        if self.loss_method == 'moment':
            return (Y - self.g(X)).pow(2).mean()
        elif self.loss_method == 'variance':
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
        elif self.loss_method == 'variance_red':
            return ((-u_int - self.g(X)).pow(2).mean() - 2 * ((-u_int - self.g(X)) * u_W_int).mean()
                    + 2 * u_int.mean() - (-u_int - self.g(X)).mean().pow(2))
        elif self.loss_method == 'variance_red_2':
            return ((-u_int - self.g(X)).pow(2).mean() + 2 * (self.g(X) * u_W_int).mean()
                    - double_int.mean() + 2 * u_int.mean() - (-u_int - self.g(X)).mean().pow(2))
        elif self.loss_method == 'functional':
            return ((Z_sum + self.g(X))).mean()

    def initialize_training_data(self):
        X = pt.zeros([self.K, self.d]).to(device)
        Y = pt.zeros(self.K).to(device)
        if self.learn_Y_0 is True:
            Y = self.Y_0(X)
        Z_sum = pt.zeros(self.K).to(device)
        u_L2 = pt.zeros(self.K).to(device)
        u_int = pt.zeros(self.K).to(device)
        u_W_int = pt.zeros(self.K).to(device)
        double_int = pt.zeros(self.K).to(device)
        xi = pt.randn(self.K, self.d, self.N + 1).to(device)
        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi

    def zero_grad(self):
        self.Y_0.adam.zero_grad()
        for z_n in self.Z_n:
            z_n.adam.zero_grad()

    def optimization_step(self):
        self.Y_0.adam.step()
        for z_n in self.Z_n:
            z_n.adam.step()

    def gradient_descent(self, X, Y, Z_sum):
        self.zero_grad()
        loss = self.loss_function(X, Y, Z_sum)
        loss.backward()
        self.optimization_step()
        return loss

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            sd_list[name] = sd[name].numpy().tolist()
        return sd_list

    def save_logs(self):
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'seed': self.seed, 'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Y_0_state_dict': self.state_dict_to_list(self.Y_0.state_dict()),
                'Z_n_state_dict': [self.state_dict_to_list(z_n.state_dict()) for z_n in self.Z_n]}

        path_name = 'logs/%s_%s.json' % (self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%d.json' % (self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f)

    def train(self):

        pt.manual_seed(self.seed)

        print('d = %d, L = %d, K = %d, delta_t = %.3f, lr = %.2e, %s, %s'
              % (self.d, self.L, self.K, self.delta_t_np, self.lr, self.loss_method,
                 'adaptive' if self.adaptive_forward_process else ''))

        for l in range(self.L):
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi = self.initialize_training_data()

            for i in range(self.N):
                Z = self.Z_n[i](X)
                c = pt.zeros(self.d, 1).to(device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                X = (X + (self.b(X) + pt.mm(self.sigma(X), c)[:, 0]) * self.delta_t
                     + pt.mm(xi[:, :, i + 1], self.sigma(X).t()) * self.sq_delta_t)
                Y = (Y + (self.h(self.delta_t * i, X, Y, Z) + pt.mm(Z, c)[:, 0]) * self.delta_t
                     + pt.sum(Z * xi[:, :, i+1], dim=1) * self.sq_delta_t)
                if self.loss_method == 'functional':
                    Z_sum += 0.5 * pt.sum(Z**2, 1) * self.delta_t

                u_L2 += pt.sum((-Z - pt.tensor(self.u_true(X, i * self.delta_t_np)).float())**2
                               * self.delta_t, 1)

            loss = self.gradient_descent(X, Y, Z_sum)

            self.loss_log.append(loss.item())
            self.u_L2_loss.append(pt.mean(u_L2).item())

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.learn_Y_0 is True:
                self.Y_0_log.append(Y[0].item())

            if l % self.print_every == 0:
                print('%d - loss: %.4e - u-L2 loss: %.4e - time per iteration: %.2fs'
                      % (l, self.loss_log[-1], self.u_L2_loss[-1], np.mean(self.times[-100:])))

            if self.early_stopping_time is not None:
                if ((l > self.early_stopping_time) and
                        (np.std(self.u_L2_loss[-self.early_stopping_time:])
                         / self.u_L2_loss[-1] < 0.02)):
                    break

        if self.save_results is True:
            self.save_logs()
