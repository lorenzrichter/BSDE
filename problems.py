#pylint: disable=invalid-name, no-member, too-many-arguments, unused-argument, missing-docstring, too-many-instance-attributes

import numpy as np
import torch as pt

from scipy.linalg import expm


device = pt.device('cpu')


# to do:
# - elliptic problems, e.g. hitting time problems (control not explicitly time-dependent)


class LLGC():
    def __init__(self, name='LQGC', d=1, off_diag=0, T=5):
        self.name = name
        self.d = d
        self.T = T
        self.A = -pt.eye(self.d).to(device) + off_diag * pt.randn(self.d, self.d)
        self.B = pt.eye(self.d).to(device) + off_diag * pt.randn(self.d, self.d)
        self.alpha = pt.ones(self.d, 1).to(device)

        if ~np.all(np.linalg.eigvals(self.A.numpy()) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def h(self, t, x, y, z):
        return 0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return pt.mm(x, self.alpha)[:, 0]

    def u_true(self, x, t):
        return -self.sigma(x).numpy().T.dot(expm(self.A.numpy().T * (self.T - t)).dot(
            self.alpha.numpy())[:, 0])


class LQGC():
    def __init__(self, name='LQGC', delta_t=0.05, d=1, off_diag=0, T=5):
        self.name = name
        self.d = d
        self.T = T
        self.A = -pt.eye(self.d).to(device) + off_diag * pt.randn(self.d, self.d)
        self.B = pt.eye(self.d).to(device) + off_diag * pt.randn(self.d, self.d)
        self.delta_t = delta_t
        self.N = int(np.floor(self.T / self.delta_t))

        if ~np.all(np.linalg.eigvals(self.A.numpy()) < 0):
            print('not all EV of A are negative')

        self.P = pt.zeros(self.d, self.d).to(device)
        self.Q = 0.5 * pt.eye(self.d).to(device)
        self.R = pt.eye(self.d).to(device)
        self.F = pt.zeros([self.N + 1, self.d, self.d]).to(device)
        self.F[self.N, :, :] = self.R
        for n in range(self.N, 0, -1):
            self.F[n - 1, :, :] = (self.F[n, :, :]
                                   + (pt.mm(self.A.t(), self.F[n, :, :])
                                      + pt.mm(self.F[n, :, :], self.A)
                                      - pt.mm(pt.mm(pt.mm(pt.mm(self.F[n, :, :], self.B),
                                                          self.Q.inverse()), self.B.t()),
                                              self.F[n, :, :]) + self.P) * self.delta_t)
        self.G = pt.zeros([self.N + 1])
        for n in range(self.N, 0, -1):
            self.G[n - 1] = self.G[n] - pt.trace(pt.mm(pt.mm(self.B, self.F[n, :, :]), self.B)) * self.delta_t

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def h(self, t, x, y, z):
        return 0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return pt.sum(x.t() * pt.mm(self.R, x.t()), 0)

    def u_true(self, x, t):
        n = int(np.ceil(t / self.delta_t))
        return -pt.mm(pt.mm(pt.mm(self.Q.inverse(), self.B.t()), self.F[n, :, :]), x.t()).t()

    def v_true(self, x, t):
        n = int(np.ceil(t / self.delta_t))
        return -pt.mm(x, pt.mm(self.F[n, :, :], x.t())).t() + problem.G[n]


class DoubleWell():
    def __init__(self, name='Double well', d=1, T=5, alpha=1, beta=1):
        self.name = name
        self.d = d
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.B = pt.eye(self.d).to(device)

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

    def V(self, x):
        return self.beta * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.beta * x * (x**2 - 1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B

    def h(self, t, x, y, z):
        return 0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return self.alpha * (x - 1)**2

    def u_true(self, x, t):
        return pt.tensor([[0.0]])

class HeatEquation():
    def __init__(self, name='Heat equation', d=1, T=5):
        self.name = name
        self.d = d
        self.T = T
        self.A = pt.zeros(self.d).to(device)
        self.B = pt.eye(self.d).to(device)
        self.alpha = pt.ones(self.d, 1).to(device)

        if ~np.all(np.linalg.eigvals(self.A.numpy()) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def h(self, t, x, y, z):
        return 0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return pt.mm(x, self.alpha)[:, 0]

    def u_true(self, x, t):
        return -self.sigma(x).numpy().T.dot(expm(self.A.numpy().T * (self.T - t))
                                            .dot(self.alpha.numpy())[:, 0])
