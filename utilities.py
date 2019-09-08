#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring


import matplotlib.pyplot as plt
import numpy as np
import torch as pt

def plot_loss_logs(experiment_name, models):
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle('%s, d = %d' % (experiment_name, models[0].d))

    for model in models:
        if 'functional' in model.name:
            ax[0].plot(np.array(model.loss_log) - np.min(np.array(model.loss_log)))
            ax[0].set_yscale('log')
        else:
            ax[0].plot(model.loss_log)
            ax[0].set_yscale('log')
        ax[1].plot(model.u_L2_loss, label=model.name)
        ax[1].set_yscale('log')
        ax[1].legend()
    ax[0].set_title('loss')
    ax[1].set_title(r'$\mathbb{E}\left[\|u - u^* \|^2_{L_2}\right]$')
    return fig

def plot_solution(model, x, t, r=1):
    model.Y_0.eval()
    for z_n in model.Z_n:
        z_n.eval()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    if t == 'all':
        t_range = np.linspace(0, model.T, model.N)
        for j in range(r):
            ax.plot(t_range, [model.u_true(x, i * model.delta_t_np)[0, j].item()
                              for i in range(model.N)], label='true x_%d' % j)
            ax.plot(t_range, [-model.Z_n[i].forward(x)[0, j].item() for i in range(model.N)], '--',
                    label='approx x_%d' % j)
    elif x == 'all':
        n = int(np.ceil(t / model.delta_t_np))
        x_range = pt.linspace(-3, 3, 100).repeat(model.d, 1).t()
        for j in range(r):
            ax.plot(x_range.numpy()[:, j], model.u_true(x_range, t)[:, j].numpy(),
                    label='true x_%d' % j);
            ax.plot(x_range.numpy()[:, j], -model.Z_n[n](x_range)[:, j].detach().numpy(), '--',
                    label='approx x_%d' % j)

    plt.legend()
    return fig