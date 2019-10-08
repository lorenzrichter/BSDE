#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-instance-attributes, not-callable, no-else-return
#pylint: disable=inconsistent-return-statements, too-many-locals, too-many-return-statements
#pylint: disable=too-many-statements


from datetime import date
import json
import numpy as np
import os
import time
import torch as pt

from function_space import DenseNet, Linear, NN, SingleParam
from utilities import do_importance_sampling


device = pt.device('cpu')


# to do:
# - flexible learning rate, line search?
# - automatic logging
# - flexible optimizer


class Solver():

    def __init__(self, name, problem, lr=0.001, L=10000, K=50, delta_t=0.05,
                 approx_method='control', loss_method='variance', time_approx='outer',
                 learn_Y_0=False, adaptive_forward_process=True, early_stopping_time=10000,
                 random_X_0=False, compute_gradient_variance=0, IS_variance_K=0,
                 metastability_logs=None, print_every=100, seed=42, save_results=False):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d
        self.T = problem.T
        self.X_0 = problem.X_0
        self.Y_0 = pt.tensor([0.0])

        # hyperparameters
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size
        self.random_X_0 = random_X_0

        # learning properties
        self.loss_method = loss_method
        self.approx_method = approx_method
        self.learn_Y_0 = learn_Y_0
        self.adaptive_forward_process = adaptive_forward_process
        self.early_stopping_time = early_stopping_time
        if self.loss_method == 'moment':
            self.learn_Y_0 = True
        if self.loss_method == 'relative_entropy':
            self.adaptive_forward_process = True
        if self.loss_method == 'cross_entropy':
            self.learn_Y_0 = False

        # function approximation
        self.Phis = []
        self.time_approx = time_approx
        pt.manual_seed(self.seed)
        if self.approx_method == 'control':
            self.y_0 = SingleParam(lr=self.lr).to(device)
            if self.time_approx == 'outer':
                self.z_n = [DenseNet(d_in=self.d, d_out=self.d, lr=self.lr) for i in range(self.N)]
            elif self.time_approx == 'inner':
                self.z_n = DenseNet(d_in=self.d + 1, d_out=self.d, lr=self.lr)

        elif self.approx_method == 'value_function':
            if self.time_approx == 'outer':
                self.y_n = [DenseNet(d_in=self.d, d_out=1, lr=self.lr) for i in range(self.N)]
            elif self.time_approx == 'inner':
                self.y_n = [DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr)]

        self.update_Phis()

        for phi in self.Phis:
            phi.train()

        self.p = sum([np.prod(params.size()) for params in filter(lambda params:
                                                                  params.requires_grad,
                                                                  self.Phis[0].parameters())])

        # logging
        self.Y_0_log = []
        self.loss_log = []
        self.u_L2_loss = []
        self.times = []
        self.grads_rel_error_log = []
        self.particles_close_to_target = []

        # printing and logging
        self.print_every = print_every
        self.save_results = save_results
        self.compute_gradient_variance = compute_gradient_variance
        self.IS_variance_K = IS_variance_K
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

    def update_Phis(self):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                self.Phis = self.z_n + [self.y_0]
            elif self.time_approx == 'inner':
                self.Phis = [self.z_n, self.y_0]
        elif self.approx_method == 'value_function':
            self.Phis = self.y_n

    def loss_function(self, X, Y, Z_sum, l):
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
        elif self.loss_method == 'relative_entropy':
            return (Z_sum + self.g(X)).mean()
        elif self.loss_method == 'cross_entropy':
            if self.adaptive_forward_process is True:
                return (Y * pt.exp(-self.g(X) + Y)).mean()
            return (Y * pt.exp(-self.g(X))).mean()
        elif self.loss_method == 'relative_entropy_variance':
            if l < 1000:
                return ((Z_sum + self.g(X))).mean()
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
            #return ((Z_sum + self.g(X))).mean() + (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)

    def initialize_training_data(self):
        X = self.X_0.repeat(self.K, 1).to(device)
        if self.random_X_0 is True:
            X = pt.randn(self.K, self.d).to(device)
        Y = self.Y_0.repeat(self.K).to(device)
        if self.approx_method == 'value_function':
            X = pt.autograd.Variable(X, requires_grad=True)
            Y = self.Y_n(X, 0)[:, 0]
        elif self.learn_Y_0 is True:
            Y = self.y_0(X)
            self.Y_0_log.append(Y[0].item())
        Z_sum = pt.zeros(self.K).to(device)
        u_L2 = pt.zeros(self.K).to(device)
        u_int = pt.zeros(self.K).to(device)
        u_W_int = pt.zeros(self.K).to(device)
        double_int = pt.zeros(self.K).to(device)

        xi = pt.randn(self.K, self.d, self.N + 1).to(device)
        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi

    def zero_grad(self):
        for phi in self.Phis:
            phi.adam.zero_grad()

    def optimization_step(self):
        for phi in self.Phis:
            phi.adam.step()

    def gradient_descent(self, X, Y, Z_sum, l, additional_loss):
        self.zero_grad()
        loss = self.loss_function(X, Y, Z_sum, l) + additional_loss
        loss.backward()
        self.optimization_step()
        return loss

    def flatten_gradient(self, k, grads, grads_flat):
        i = 0
        for grad in grads:
            grad_flat = grad.reshape(-1)
            j = len(grad_flat)
            grads_flat[k, i:i + j] = grad_flat
            i += j
        return grads_flat

    def get_gradient_variances(self, X, Y):
        grads_mean = pt.zeros(self.N, self.p)
        grads_var = pt.zeros(self.N, self.p)

        for n in range(self.N):

            grads_Y_flat = pt.zeros(self.K, self.p)

            for k in range(self.K):
                self.zero_grad()
                Y[k].backward(retain_graph=True)

                grad_Y = [params.grad for params in list(filter(lambda params:
                                                                params.requires_grad,
                                                                self.z_n[n].parameters()))
                          if params.grad is not None]

                grads_Y_flat = self.flatten_gradient(k, grad_Y, grads_Y_flat)

            grads_g_X_flat = pt.zeros(self.K, self.p)

            if self.adaptive_forward_process is True:

                for k in range(self.K):
                    self.zero_grad()
                    self.g(X[0, :].unsqueeze(0)).backward(retain_graph=True)

                    grad_g_X = [params.grad for params in list(filter(lambda params:
                                                                      params.requires_grad,
                                                                      self.z_n[n].parameters()))
                                if params.grad is not None]

                    grads_g_X_flat = self.flatten_gradient(k, grad_g_X, grads_g_X_flat)

            if self.loss_method == 'moment':
                grads_flat = 2 * (Y - self.g(X)).unsqueeze(1) * (grads_Y_flat - grads_g_X_flat)
            elif self.loss_method == 'variance':
                grads_flat = 2 * (((Y - self.g(X)).unsqueeze(1)
                                   - pt.mean((Y - self.g(X)).unsqueeze(1), 0).unsqueeze(0))
                                  * (grads_Y_flat - grads_g_X_flat
                                     - pt.mean(grads_Y_flat - grads_g_X_flat, 0).unsqueeze(0)))

            grads_mean[n, :] = pt.mean(grads_flat, dim=0)
            grads_var[n, :] = pt.var(grads_flat, dim=0)

        grads_rel_error = pt.sqrt(grads_var) / grads_mean
        grads_rel_error[grads_rel_error != grads_rel_error] = 0
        return grads_rel_error

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            sd_list[name] = sd[name].numpy().tolist()
        return sd_list

    def save_logs(self):
        # currently does not work for all modi
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'seed': self.seed, 'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Y_0_state_dict': self.state_dict_to_list(self.Y_0.state_dict()),
                'Z_n_state_dict': [self.state_dict_to_list(z.state_dict()) for z in self.z_n]}

        path_name = 'logs/%s_%s.json' % (self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%d.json' % (self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f)

    def compute_grad_Y(self, X, n):
        Y_n_eval = self.Y_n(X, n).squeeze(1).sum() # compare to Jacobi-Vector trick
        Y_n_eval.backward(retain_graph=True) # do we need this?
        Z, = pt.autograd.grad(Y_n_eval, X, create_graph=True)
        Z = pt.mm(self.sigma(X), Z.t()).t()
        return Z

    def Y_n(self, X, n):
        if self.time_approx == 'outer':
            return self.y_n[n](X)
        elif self.time_approx == 'inner':
            t_X = pt.cat([pt.ones([X.shape[0], 1]) * n * self.delta_t, X], 1)
            return self.y_n[0](t_X)

    def Z_n(self, X, n):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                return self.z_n[n](X)
            elif self.time_approx == 'inner':
                t_X = pt.cat([pt.ones([X.shape[0], 1]) * n * self.delta_t, X], 1)
                return self.z_n(t_X)
        if self.approx_method == 'value_function':
            return self.compute_grad_Y(X, n)

    def train(self):

        pt.manual_seed(self.seed)

        print('d = %d, L = %d, K = %d, delta_t = %.2e, lr = %.2e, %s, %s, %s, %s'
              % (self.d, self.L, self.K, self.delta_t_np, self.lr, self.approx_method,
                 self.time_approx, self.loss_method,
                 'adaptive' if self.adaptive_forward_process else ''))

        for l in range(self.L):
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi = self.initialize_training_data()
            additional_loss = pt.zeros(self.K)

            for n in range(self.N):
                if self.approx_method == 'value_function':
                    if n > 0:
                        additional_loss += (self.Y_n(X, n)[:, 0] - Y).pow(2)
                Z = self.Z_n(X, n)
                c = pt.zeros(self.d, 1).to(device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                X = (X + (self.b(X) + pt.mm(self.sigma(X), c)[:, 0]) * self.delta_t
                     + pt.mm(self.sigma(X), xi[:, :, n + 1].t()).t() * self.sq_delta_t)
                Y = (Y + (self.h(self.delta_t * n, X, Y, Z) + pt.mm(Z, c)[:, 0]) * self.delta_t
                     + pt.sum(Z * xi[:, :, n + 1], dim=1) * self.sq_delta_t)
                if 'relative_entropy' in self.loss_method:
                    Z_sum += 0.5 * pt.sum(Z**2, 1) * self.delta_t

                if self.u_true(X, n * self.delta_t_np) is not None:
                    u_L2 += pt.sum((-Z
                                    - pt.tensor(self.u_true(X, n * self.delta_t_np)).t().float())**2
                                   * self.delta_t, 1)

            if self.compute_gradient_variance > 0 and l % self.compute_gradient_variance == 0:
                self.grads_rel_error_log.append(pt.mean(self.get_gradient_variances(X, Y)).item())

            loss = self.gradient_descent(X, Y, Z_sum, l, additional_loss.mean())

            self.loss_log.append(loss.item())
            self.u_L2_loss.append(pt.mean(u_L2).item())
            if self.metastability_logs is not None:
                target, epsilon = self.metastability_logs
                self.particles_close_to_target.append(pt.mean((pt.sqrt(pt.sum((X - target)**2, 1)) <
                                                               epsilon).float()))

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.4e - u L2: %.4e - time/iter: %.2fs'
                          % (l, self.loss_log[-1], self.u_L2_loss[-1],
                             np.mean(self.times[-self.print_every:])))
                if self.IS_variance_K > 0:
                    variance_naive, variance_IS = do_importance_sampling(self.problem, self, self.IS_variance_K,
                                                                         control='approx', verbose=False)
                    string += ' - var naive: %.4e - var IS: %.4e' % (variance_naive, variance_IS)
                print(string)

            if self.early_stopping_time is not None:
                if ((l > self.early_stopping_time) and
                        (np.std(self.u_L2_loss[-self.early_stopping_time:])
                         / self.u_L2_loss[-1] < 0.02)):
                    break

        if self.save_results is True:
            self.save_logs()
