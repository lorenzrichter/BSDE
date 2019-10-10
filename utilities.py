#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-branches


import matplotlib.pyplot as plt
import numpy as np
import torch as pt


device = pt.device('cpu')


def plot_loss_logs(experiment_name, models):
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle('%s, d = %d' % (experiment_name, models[0].d))

    for model in models:
        if 'entropy' in model.loss_method:
            ax[0].plot(np.array(model.loss_log) - np.min(np.array(model.loss_log)),
                       label=model.name)
            ax[0].set_yscale('log')
        else:
            ax[0].plot(model.loss_log, label=model.name)
            ax[0].set_yscale('log')
        ax[1].plot(model.u_L2_loss, label=model.name)
    ax[1].set_yscale('log')
    ax[0].legend()
    ax[0].set_title('loss')
    ax[1].set_title(r'$\mathbb{E}\left[\|u - u^* \|^2_{L_2}\right]$')
    return fig

def plot_solution(model, x, t, components, ylims=None):

    n = int(np.ceil(t / model.delta_t_np))
    t_range = np.linspace(0, model.T, model.N)
    x_val = pt.linspace(-3, 3, 100)

    for phi in model.Phis:
        phi.eval()
        phi.to(pt.device('cpu'))

    if model.approx_method == 'control':
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    elif model.approx_method == 'value_function':
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))

    fig.suptitle(model.name)

    X = pt.autograd.Variable(x_val.unsqueeze(1).repeat(1, model.d), requires_grad=True)

    ax[0].set_title('control, t = %.2f' % t)
    for j in components:
        if model.u_true(x_val.unsqueeze(1).repeat(1, model.d), t) is not None:
            ax[0].plot(x_val.numpy(), model.u_true(x_val.unsqueeze(1).repeat(1, model.d), t)[j, :],
                       label='true x_%d' % j)
        ax[0].plot(x_val.numpy(), -model.Z_n(X, n).detach().numpy()[:, j], '--',
                   label='approx x_%d' % j)
    if ylims is not None:
        ax[0].set_ylim(ylims[0][0], ylims[0][1])
    ax[0].legend()

    X = pt.autograd.Variable(pt.tensor([[x] * model.d]), requires_grad=True)

    ax[1].set_title('control, x = %.2f' % x)
    for j in components:
        if model.u_true(X.detach(), n * model.delta_t_np) is not None:
            ax[1].plot(t_range, [model.u_true(X.detach(), n * model.delta_t_np)[j].item() for n in
                                 range(model.N)], label='true x_%d' % j)
        ax[1].plot(t_range, [-model.Z_n(X, n)[0, j].item() for n in range(model.N)], '--',
                   label='approx x_%d' % j)
    if ylims is not None:
        ax[1].set_ylim(ylims[1][0], ylims[1][1])

    if model.approx_method == 'value_function':

        X = pt.autograd.Variable(x_val.unsqueeze(1).repeat(1, model.d), requires_grad=True)

        ax[2].set_title('value function, t = %.2f' % t)
        if model.v_true(x_val.unsqueeze(1).repeat(1, model.d), t) is not None:
            ax[2].plot(x_val.numpy(), model.v_true(x_val.unsqueeze(1).repeat(1, model.d), t))
        ax[2].plot(x_val.numpy(), model.Y_n(X, n)[:, 0].detach().numpy(), '--')
        if ylims is not None:
            ax[2].set_ylim(ylims[2][0], ylims[2][1])

        X = pt.autograd.Variable(pt.tensor([[x] * model.d]), requires_grad=True)

        ax[3].set_title('value function, x = %.2f' % x)
        if model.v_true(X.detach(), n * model.delta_t_np) is not None:
            ax[3].plot(t_range, [model.v_true(X.detach(), n * model.delta_t_np).item()
                                 for n in range(model.N)])
        ax[3].plot(t_range, [model.Y_n(X, n)[0, 0].detach().numpy() for n in range(model.N)], '--')
        if ylims is not None:
            ax[3].set_ylim(ylims[3][0], ylims[3][1])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for phi in model.Phis:
        phi.to(device)

    return fig

def do_importance_sampling(problem, model, K, control='approx', verbose=True, delta_t=0.01):

    X = problem.X_0.repeat(K, 1)
    X_u = problem.X_0.repeat(K, 1)
    ito_int = pt.zeros(K)
    riemann_int = pt.zeros(K)

    sq_delta_t = np.sqrt(delta_t)
    N = int(np.ceil(problem.T / delta_t))

    for n in range(N):
        xi = pt.randn(K, problem.d)
        X = (X + problem.b(X) * delta_t
             + pt.mm(problem.sigma(X), xi.t()).t() * sq_delta_t)
        if control == 'approx':
            ut = -model.Z_n(X_u, int(np.floor(n * delta_t / model.delta_t)))
        if control == 'true':
            ut = pt.tensor(problem.u_true(X_u, n * delta_t)).t().float()
        X_u = (X_u + (problem.b(X_u) + pt.mm(problem.sigma(X_u), ut.t()).t()) * delta_t
               + pt.mm(problem.sigma(X_u), xi.t()).t() * sq_delta_t)
        ito_int += pt.sum(ut * xi, 1) * sq_delta_t
        riemann_int += pt.sum(ut**2, 1) * delta_t

    girsanov = pt.exp(- ito_int - 0.5 * riemann_int)

    variance_naive = pt.var(pt.exp(-problem.g(X))).item()
    variance_IS = pt.var(pt.exp(-problem.g(X_u)) * girsanov).item()

    if verbose is True:
        print('variance of naive estimator: %.4e' % variance_naive)
        print('variance of importance sampling estimator: %.4e' % variance_IS)
    return variance_naive, variance_IS
