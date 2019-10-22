#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring, arguments-differ, unused-argument


import torch as pt


class SingleParam(pt.nn.Module):
    def __init__(self, lr, initial=None, seed=42):
        super(SingleParam, self).__init__()
        pt.manual_seed(seed)
        if initial is None:
            self.Y_0 = pt.nn.Parameter(pt.randn(1), requires_grad=True)
        else:
            self.Y_0 = pt.nn.Parameter(pt.tensor([initial]), requires_grad=True)
        self.register_parameter('param', self.Y_0)
        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.Y_0


class Constant(pt.nn.Module):
    def __init__(self, d, lr, seed=42):
        super(Constant, self).__init__()
        pt.manual_seed(seed)
        self.c = pt.nn.Parameter(pt.randn(d), requires_grad=True)
        self.register_parameter('param', self.c)
        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.c.repeat(x.shape[0], 1)


class Linear(pt.nn.Module):
    def __init__(self, d, B, Q, lr, seed=42):
        super(Linear, self).__init__()
        pt.manual_seed(seed)
        self.F = pt.nn.Parameter(pt.randn(d, d), requires_grad=True)
        self.B = B
        self.Q = Q
        self.register_parameter('param', self.F)
        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return pt.mm(self.Q.inverse(), pt.mm(self.B.t(), pt.mm(self.F, x.t()))).t()


class Affine(pt.nn.Module):
    def __init__(self, d, lr, seed=42):
        super(Affine, self).__init__()
        pt.manual_seed(seed)
        self.A = pt.nn.Parameter(pt.randn(d, d), requires_grad=True)
        self.b = pt.nn.Parameter(pt.randn(1, d), requires_grad=True)
        self.register_parameter('param A', self.A)
        self.register_parameter('param b', self.b)
        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return pt.mm(self.A, x.t()).t() + self.b


class Sines(pt.nn.Module):
    # linear comibation of M sine functions with frequencies omega
    # only works for d = 1
    def __init__(self, d, lr, M=10, seed=42):
        super(Sines, self).__init__()
        pt.manual_seed(seed)
        self.alpha = pt.nn.Parameter(pt.randn(M, 1), requires_grad=True)
        self.omega = pt.linspace(1, M, M).unsqueeze(0)

        self.register_parameter('param alpha', self.alpha)
        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return pt.mm(pt.sin(pt.mm(x, self.omega)), self.alpha)


class NN(pt.nn.Module):
    def __init__(self, d_in, d_out, lr, seed=42):
        super(NN, self).__init__()
        pt.manual_seed(seed)
        self.nn_dims = [d_in, 20, d_out] # [d, 40, 30, 30, 40, d]
        self.W = [item for sublist in
                  [[pt.nn.Parameter(pt.randn(self.nn_dims[i], self.nn_dims[i + 1],
                                             requires_grad=True)),
                    pt.nn.Parameter(pt.randn(self.nn_dims[i + 1], requires_grad=True))] for
                   i in range(len(self.nn_dims) - 1)]
                  for item in sublist]
        for i, w in enumerate(self.W):
            self.register_parameter('param %d' % i, w)

        self.BN1 = pt.nn.BatchNorm1d(self.nn_dims[0])
        self.BN2 = pt.nn.BatchNorm1d(self.nn_dims[1])
        self.BN3 = pt.nn.BatchNorm1d(self.nn_dims[2])
        #self.BN4 = pt.nn.BatchNorm1d(self.nn_dims[3])
        #self.BN5 = pt.nn.BatchNorm1d(self.nn_dims[4])
        #self.BN6 = pt.nn.BatchNorm1d(self.nn_dims[5])
        self.BN = [self.BN1, self.BN2, self.BN3]#, self.BN5, self.BN6]

        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.BN[0](x)
        for i in range(len(self.nn_dims) - 1):
            x = pt.matmul(x, self.W[2 * i]) # + self.W[2 * i + 1]
            x = self.BN[i + 1](x)
            if i != len(self.nn_dims) - 2:
                x = pt.nn.functional.relu(x)
        return x


class DenseNet(pt.nn.Module):
    def __init__(self, d_in, d_out, lr, seed=42):
        super(DenseNet, self).__init__()
        pt.manual_seed(seed)
        self.nn_dims = [d_in, 30, 30, d_out]
        self.W = [item for sublist in
                  [[pt.nn.Parameter(pt.randn(sum(self.nn_dims[:i + 1]), self.nn_dims[i + 1],
                                             requires_grad=True) * 0.1),
                    pt.nn.Parameter(pt.zeros(self.nn_dims[i + 1], requires_grad=True))] for
                   i in range(len(self.nn_dims) - 1)]
                  for item in sublist]
        for i, w in enumerate(self.W):
            self.register_parameter('param %d' % i, w)

        self.adam = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                x = pt.matmul(x, self.W[2 * i]) + self.W[2 * i + 1]
            else:
                x = pt.cat([x, pt.nn.functional.relu(pt.matmul(x, self.W[2 * i])
                                                     + self.W[2 * i + 1]) ** 2], dim=1)
        return x
