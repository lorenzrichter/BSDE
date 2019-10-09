#pylint: disable=invalid-name, no-member, too-many-arguments, unused-argument, missing-docstring, too-many-instance-attributes

import numpy as np
import torch as pt

from numpy import exp, log

from scipy.linalg import expm, inv, solve_banded


device = pt.device('cpu')


# to do:
# - elliptic problems, e.g. hitting time problems (control not explicitly time-dependent)


class LLGC():
    def __init__(self, name='LLGC', d=1, off_diag=0, T=5):
        self.name = name
        self.d = d
        self.T = T
        self.A = (-pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.B = (pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.alpha = pt.ones(self.d, 1).to(device)
        self.X_0 = pt.zeros(self.d).to(device)

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
            self.alpha.numpy()) * np.ones(x.shape).T)

    def v_true(self, x, t):
        Sigma_n = (0.5 * inv(self.A.numpy()).dot(expm(self.A.numpy() * self.T))
                   .dot(self.sigma(np.zeros([self.d, self.d])))
                   .dot(self.sigma(np.zeros([self.d, self.d])).t())
                   .dot(expm(self.A.numpy().T * self.T))
                   -0.5 * inv(self.A.numpy()).dot(expm(self.A.numpy() * t))
                   .dot(self.sigma(np.zeros([self.d, self.d])))
                   .dot(self.sigma(np.zeros([self.d, self.d])).t())
                   .dot(expm(self.A.numpy().T * t)))
        return ((expm(self.A.numpy() * (self.T - t)).dot(x.t()).T).dot(self.alpha.numpy())
                - 0.5 * self.alpha.numpy().T.dot(Sigma_n.dot(self.alpha)))


class LQGC():
    def __init__(self, name='LQGC', delta_t=0.05, d=1, off_diag=0, T=5):
        self.name = name
        self.d = d
        self.T = T
        self.A = (-pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.B = (pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.delta_t = delta_t
        self.N = int(np.floor(self.T / self.delta_t))
        self.X_0 = pt.zeros(self.d)

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
            self.G[n - 1] = (self.G[n] - pt.trace(pt.mm(pt.mm(self.B, self.F[n, :, :]), self.B))
                             * self.delta_t)

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
        return -pt.mm(pt.mm(pt.mm(self.Q.inverse(), self.B.t()), self.F[n, :, :]), x.t()).detach().numpy()#.t()

    def v_true(self, x, t):
        n = int(np.ceil(t / self.delta_t))
        return -pt.mm(x, pt.mm(self.F[n, :, :], x.t())).t() + self.G[n]


class DoubleWell():
    def __init__(self, name='Double well', d=1, T=5, delta_t=0.01, alpha=1, beta=1):
        self.name = name
        self.d = d
        self.T = T
        self.delta_t = delta_t
        self.alpha = alpha
        self.beta = beta
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

        # range of x, [-xb, xb]
        self.xb = 5
        # number of discrete interval
        self.nx = 2500
        dx = 2.0 * self.xb / self.nx

        beta = 1

        xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):
            x = -self.xb + (i + 0.5) * dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * dx
                x1 = -self.xb + i * dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / (dx**2)
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * dx
                x1 = -self.xb + (i + 1) * dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / dx**2

        A = -A
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g(xvec))

        for n in range(N - 1, -1, -1):
            band = - delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                          np.diagonal(A, offset=0) - N / self.T,
                                          np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));

        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -self.B * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / dx
        self.u = pt.tensor(self.u)

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
        i = pt.floor((x.squeeze() + self.xb) * 250).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return self.u[n, i].unsqueeze(0)


class HeatEquation():
    def __init__(self, name='Heat equation', d=1, T=5):
        self.name = name
        self.d = d
        self.T = T
        self.A = pt.zeros(self.d).to(device)
        self.B = pt.eye(self.d).to(device)
        self.alpha = pt.ones(self.d, 1).to(device)
        self.X_0 = pt.zeros(self.d)

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

class Mueller2d():
    def __init__(self, name='Mueller potential', d=2, T=5, alpha=1, beta=1):

        self.name = name
        self.d = d
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.X_0 = pt.zeros(self.d).to(device)
        self.B = pt.eye(self.d).to(device)

        self.AA = [-20, -10, -17, 1.5] 
        self.coeff_a = [-1, -1, -6.5, 0.7]
        self.coeff_b = [0, 0, 11.0, 0.6]
        self.coeff_c = [-10, -10, -6.5, 0.7]
        self.xc = [1.0, 0.0, -0.5, -1.0]
        self.yc = [0.0, 0.5, 1.5, 1.0]

        if self.d != 2:
            print('The Muller example is only for d = 2.')

    def V(self, x):
      s = 0 
      for i in range(4):
        dx = x[:,0] - self.xc[i] 
        dy = x[:,1] - self.yc[i] 
        s += self.AA[i] * exp(self.coeff_a[i] * dx * dx + self.coeff_b[i] * dx * dy + self.coeff_c[i] * dy * dy)      
      return s

    def grad_V(self, x):
      grad = pt.zeros(x.shape[0], self.d) 
      for i in range(4):
        dx = x[:,0] - self.xc[i] 
        dy = x[:,1] - self.yc[i] 
        grad[:,0] += self.AA[i] * (2 * self.coeff_a[i] * dx + self.coeff_b[i] * dy) * exp(self.coeff_a[i] * dx * dx + self.coeff_b[i] * dx * dy + self.coeff_c[i] * dy * dy) 
        grad[:,1] += self.AA[i] * (2 * self.coeff_c[i] * dy + self.coeff_b[i] * dx) * exp(self.coeff_a[i] * dx * dx + self.coeff_b[i] * dx * dy + self.coeff_c[i] * dy * dy)

      return grad

    def b(self, x):
      return -1.0 * self.grad_V(x)

    def sigma(self, x):
        return self.B

    def h(self, t, x, y, z):
        return 0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return self.alpha * (x - 1)**2

    def u_true(self, x, t):
        return pt.tensor([[0.0]])

