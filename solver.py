from datetime import date
import json
import numpy as np
import os
import time
import torch as pt

from function_space import DenseNet, Linear, NN, SingleParam
from utilities import do_importance_sampling

device = pt.device('cpu')

class Solver():

    def __init__(self, name, problem, lr=0.001, L=10000, K=50, delta_t=0.01, approx_method='control', loss_method='variance',
            time_approx='outer', adaptive_forward_process=False, random_X_0=False, metastability_logs=None, print_every=100, save_results=True, u_l2_error_flag=False):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        # dimension of the problem
        self.d = problem.d
        # time interval: [0, T]
        self.T = problem.T
        # starting state
        self.X_0 = problem.X_0

        # hyperparameters
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps 
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size
        # whether x0 is randomized or fixed 
        self.random_X_0 = random_X_0

        # learning properties
        self.loss_method = loss_method
        self.approx_method = approx_method
        self.adaptive_forward_process = adaptive_forward_process
        self.learn_Y_0 = False
        self.u_l2_error_flag = u_l2_error_flag
        # Y0 will be learned when we use 2nd moment as loss function
        if self.loss_method == 'moment':
            self.learn_Y_0 = True
        if self.loss_method == 'relative_entropy':
            self.adaptive_forward_process = True

        # function approximation
        self.Phis = []
        self.time_approx = time_approx
        # if we learn control 
        if self.approx_method == 'control':
            self.y_0 = SingleParam(lr=self.lr).to(device)
            # if different neural networks are used for different time 
            if self.time_approx == 'outer':
                self.z_n = [DenseNet(d_in=self.d, d_out=self.d, lr=self.lr) for i in range(self.N)]
            # if a single neural network is used to learn control u(x,t) 
            elif self.time_approx == 'inner':
                self.z_n = DenseNet(d_in=self.d + 1, d_out=self.d, lr=self.lr)
        # if we learn value function
        elif self.approx_method == 'value_function':
            # if different neural networks are used for different time 
            if self.time_approx == 'outer':
                self.y_n = [DenseNet(d_in=self.d, d_out=1, lr=self.lr) for i in range(self.N)]
            # if a single neural network is used to learn value function phi(x,t) 
            elif self.time_approx == 'inner':
                self.y_n = [DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr)]

        # putting all netwroks together
        self.update_Phis()

        for phi in self.Phis:
            phi.train()

        # number of parameters in the NNs that will be learned 
        self.p = sum([np.prod(params.size()) for params in filter(lambda params: params.requires_grad, self.Phis[0].parameters())])
        if self.time_approx == 'outer':
            print ('%d neural network(NN), %d parameters in each network, total parameters in NNs = %d' % (self.N, self.p, self.p * self.N))
        else:
            print ('%d neural network(NN), %d parameters in each network, total parameters in NNs = %d' % (1, self.p, self.p))

        # logging
        self.Y_0_log = []
        self.loss_log = []
        self.u_L2_loss = []
        self.times = []
        self.particles_close_to_target = []

        # printing and logging
        self.print_every = print_every
        self.save_results = save_results
        self.metastability_logs = metastability_logs

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

    def v_true(self, x, t):
        return self.problem.v_true(x, t)

    # put all NNs together 
    def update_Phis(self):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                self.Phis = self.z_n + [self.y_0]
            elif self.time_approx == 'inner':
                self.Phis = [self.z_n, self.y_0]
        elif self.approx_method == 'value_function':
            self.Phis = self.y_n

    # different loss functions
    def loss_function(self, X, Y, Z_sum):
        if self.loss_method == 'moment':
            return (Y - self.g(X)).pow(2).mean()
        elif self.loss_method == 'variance':
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
        elif self.loss_method == 'relative_entropy':
            return (Z_sum + self.g(X)).mean()
        elif self.loss_method == 'cross_entropy':
            if self.adaptive_forward_process is True:
                return (Y * pt.exp(-self.g(X) + Y)).mean()
            return (Y * pt.exp(-self.g(X))).mean()

    def initialize_training_data(self):
        # shape of tensor X: k x d
        if self.random_X_0 is True: # in case we use random initial states
            X = pt.randn(self.K, self.d).to(device)
        else: # otherwise, use X_0
            X = self.X_0.repeat(self.K, 1).to(device)

        if self.learn_Y_0 is True: 
            # in this case, the loss is 2nd moment.
            # Y0 is either a constant (fixed X0) or a function (random X0)
            # here index 0 means at time 0
            if self.approx_method == 'control': 
                Y = self.y_0(X)
            else:
                if self.time_approx == 'outer': # Y0 is a function of x 
                    Y = self.y_n[0](X)
                else: # Y0 is a function of (x,t) 
                    t_X = pt.cat([pt.zeros([X.shape[0], 1]), X], 1)
                    #print ('t_X:', t_X)
                    return self.y_n[0](t_X)

            self.Y_0_log.append(Y[0].item())
        else:
            Y = pt.zeros(self.K).to(device)

        Z_sum = pt.zeros(self.K).to(device)
        u_L2 = pt.zeros(self.K).to(device)
        u_int = pt.zeros(self.K).to(device)
        u_W_int = pt.zeros(self.K).to(device)
        double_int = pt.zeros(self.K).to(device)

        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int 

    # clear the gradient
    def zero_grad(self):
        for phi in self.Phis:
            phi.adam.zero_grad()

    def gradient_descent(self):
        for phi in self.Phis:
            phi.adam.step()

    # Wei: this function seems not used?
    def flatten_gradient(self, k, grads, grads_flat):
        i = 0
        for grad in grads:
            grad_flat = grad.reshape(-1)
            j = len(grad_flat)
            grads_flat[k, i:i + j] = grad_flat
            i += j
        return grads_flat

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            sd_list[name] = sd[name].numpy().tolist()
        return sd_list

    def save_logs(self):
        # currently does not work for all modi
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Z_n_state_dict': [self.state_dict_to_list(z.state_dict()) for z in self.z_n]}

        path_name = 'logs/%s_%s.json' % (self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%d.json' % (self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f)

    # when neural network represents value function, i.e., self.approx_method = 'value_function', we computing control by taking gradient 
    def compute_grad_Y(self, X, n):
        Y_n_eval = self.Y_n(X, n).squeeze(1).sum() # compare to Jacobi-Vector trick
        Y_n_eval.backward(retain_graph=True) # do we need this?
        Z, = pt.autograd.grad(Y_n_eval, X, create_graph=True)
        Z = pt.mm(self.sigma(X), Z.t()).t()
        return Z

    # Compute value function at given time.
    # This function should be called only when neural network represents value function, i.e., self.approx_method = 'value_function'
    def Y_n(self, X, n):
        if self.time_approx == 'outer':
            return self.y_n[n](X)  # n is the time index 
        elif self.time_approx == 'inner':
            # prepare input by putting time and position together
            t_X = pt.cat([pt.ones([X.shape[0], 1]) * n * self.delta_t, X], 1)
            print ('t_X:', t_X)
            return self.y_n[0](t_X)

    # compute control at given time
    def Z_n(self, X, n):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                return self.z_n[n](X)
            elif self.time_approx == 'inner':
                t_X = pt.cat([pt.ones([X.shape[0], 1]) * n * self.delta_t, X], 1)
                return self.z_n(t_X)
        if self.approx_method == 'value_function': 
            # need to compute gradient 
            return self.compute_grad_Y(X, n)

    def train(self):

        print('d = %d, L = %d, K = %d, delta_t = %.2e, N = %d, lr = %.2e, %s, %s, %s, %s'
              % (self.d, self.L, self.K, self.delta_t_np, self.N, self.lr, self.approx_method, self.time_approx, self.loss_method, 'adaptive' if self.adaptive_forward_process else ''))

        # stochastic gradient descent (SGD) steps
        for l in range(self.L):
            # get current time
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int = self.initialize_training_data()

            additional_loss = pt.zeros(self.K)
            for n in range(self.N):
                # Gaussian random variables 
                xi = pt.randn(self.K, self.d).to(device)
                if self.approx_method == 'value_function':
                    if n > 0:
                        additional_loss += (self.Y_n(X, n)[:, 0] - Y).pow(2)
                Z = self.Z_n(X, n)
                c = pt.zeros(self.K, self.d).to(device)
                if self.adaptive_forward_process is True:
                    c = -Z
                X = X + (self.b(X) + pt.bmm(self.sigma(X), c.unsqueeze(2)).squeeze(2)) * self.delta_t + pt.bmm(self.sigma(X), xi.unsqueeze(2)).squeeze(2) * self.sq_delta_t
                Y = Y + (self.h(self.delta_t * n, X, Y, Z) + pt.sum(Z*c, 1)) * self.delta_t + pt.sum(Z*xi, 1) * self.sq_delta_t
                if 'relative_entropy' in self.loss_method:
                    Z_sum += 0.5 * pt.sum(Z**2, 1) * self.delta_t

                if self.u_l2_error_flag == True:
                    u_L2 += pt.sum((-Z - pt.tensor(self.u_true(X, n * self.delta_t_np)).t().float())**2 * self.delta_t, 1)

            # total loss function
            loss = self.loss_function(X, Y, Z_sum) + additional_loss.mean()
            self.zero_grad()
            loss.backward()
            self.gradient_descent()

            self.loss_log.append(loss.item())
            self.u_L2_loss.append(pt.mean(u_L2).item())

            if self.metastability_logs is not None:
                target, epsilon = self.metastability_logs
                self.particles_close_to_target.append(pt.mean((pt.sqrt(pt.sum((X - target)**2, 1)) < epsilon).float()))

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.4e - u L2: %.4e - time/iter: %.2fs' % (l, self.loss_log[-1], self.u_L2_loss[-1], np.mean(self.times[-self.print_every:])))
                print(string)

        if self.save_results is True:
            self.save_logs()
