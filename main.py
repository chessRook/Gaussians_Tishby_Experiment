import numpy as np
import matplotlib.pyplot as plt


def lambda_i(i):
    lambda_ = 1 - 1. / i
    return lambda_


def main(dim=10):
    lambdas_lst = [lambda_i(i+1) for i in range(1, dim + 8)]
    x_axis_lower, y_axis_lower = [], []
    # lower graph
    for n in range(2, dim):
        i_tx, i_ty = equation_15(n, lambdas_lst)
        x_axis_lower.append(i_tx)
        y_axis_lower.append(i_ty)

    # upper graph
    x_axis_upper, y_axis_upper = [], []
    for n in range(2, dim):
        i_tx, i_ty = equation_15(n, lambdas_lst, upper=True)
        x_axis_upper.append(i_tx)
        y_axis_upper.append(i_ty)

    plotter(x_axis_lower, y_axis_lower, x_axis_upper, y_axis_upper, add_2_title='', dim=dim)


def plotter(x_axis_lower, y_axis_lower, x_axis_upper, y_axis_upper, add_2_title='', dim=None):
    x_axis_lower, y_axis_lower = x_axis_lower[30:], y_axis_lower[30:]
    x_axis_upper, y_axis_upper = x_axis_upper[30:], y_axis_upper[30:]
    plt.plot(x_axis_lower, y_axis_lower, label='Lower')
    plt.plot(x_axis_upper, y_axis_upper, label='Upper')
    plt.title(f'eq_15 graph in Chechick et al. {add_2_title}')
    plt.legend(loc="upper left")
    plt.show()

    ###########################
    # Log scale plotter
    lg_x_lower, lg_y_lower = np.log(x_axis_lower), np.log(y_axis_lower)
    lg_x_upper, lg_y_upper = np.log(x_axis_upper), np.log(y_axis_upper)
    plt.plot(lg_x_lower, lg_y_lower, label='Lower')
    plt.plot(lg_x_upper, lg_y_upper, label='Upper')
    plt.title(f'eq_15 graph in Chechick et al.::LOG SCALE!|  dim={dim}. {add_2_title}')
    plt.xlabel('I(T,X)')
    plt.ylabel('I(T,Y)')
    plt.legend(loc="upper left")
    plt.show()
    ###########################


def comp_geometric_mean(lambdas_lst, n_beta):
    comps = [1 - lambda_ for lambda_ in lambdas_lst]
    final = geometric_mean(comps, n_beta)
    return final


def geometric_mean(lambdas_lst, n_beta):
    lambdas_lg = np.log(lambdas_lst)
    lambdas_relevant_lg = lambdas_lg[:n_beta]
    lg_avg = np.mean(lambdas_relevant_lg)
    final = np.exp(lg_avg)
    return final


def get_itx_lower_bound(lambdas, beta):
    n_beta = beta
    lambda_NI = lambdas[n_beta - 1]
    c_nI = 0
    for i in range(2, n_beta - 2):
        lambda_i = lambdas[i]
        frac_i = lambda_NI * (1 - lambda_i) / (lambda_i * (1 - lambda_NI))
        term_i = np.log(frac_i)
        c_nI += term_i
    i_tx = c_nI
    return i_tx


def get_itx_upper_bound(lambdas, beta):
    itx = get_itx_lower_bound(lambdas, beta + 1)
    return itx


def equation_15(beta, lambdas, upper=False):
    complements_geometric_mean = comp_geometric_mean(lambdas, n_beta=beta - 1)
    geometric_mean_ = geometric_mean(lambdas, n_beta=beta - 1)
    if upper:
        i_tx = get_itx_upper_bound(lambdas, beta)
    else:
        i_tx = get_itx_lower_bound(lambdas, beta)
    inner_log_term = complements_geometric_mean + np.exp(2 * i_tx / beta) * geometric_mean_
    i_ty = i_tx - beta * np.log(inner_log_term) / 2.
    return i_tx, i_ty


if __name__ == '__main__':
    main(50_000)
