import math
import numpy as np
import matplotlib.pyplot as plt


def lambda_i(i):
    lambda_ = 1 - 1. / i
    return lambda_


def complementer(lambdas_dct):
    comps = dict()
    for lambda_ in lambdas_dct:
        lambda_comp = 1 - lambda_
        comps[lambda_comp] = lambdas_dct[lambda_]
    return comps


def main(dim=10):
    lambdas_dct = lambdas_builder(dim + 5)
    lambdas_complements = complementer(lambdas_dct)
    cumulative_lambdas_dct = accumulator(lambdas_dct)
    x_axis_lower, y_axis_lower = [], []
    # lower graph
    for n in range(5, dim):
        i_tx, i_ty = equation_15(n, lambdas_dct, lambdas_complements, cumulative_lambdas_dct)
        x_axis_lower.append(i_tx)
        y_axis_lower.append(i_ty)

    # upper graph
    x_axis_upper, y_axis_upper = [], []
    for n in range(5, dim):
        i_tx, i_ty = equation_15(n, lambdas_dct, lambdas_complements, cumulative_lambdas_dct, upper=True)
        x_axis_upper.append(i_tx)
        y_axis_upper.append(i_ty)

    plotter(x_axis_lower, y_axis_lower, x_axis_upper, y_axis_upper, add_2_title='', dim=dim)


def accumulator(lambdas_dct):
    cumulative = dict()
    total_mult = 0
    for lambda_ in lambdas_dct:
        total_mult += lambdas_dct[lambda_]
        cumulative[lambda_] = total_mult
    return cumulative


def lambdas_builder(dim):
    lambdas_multiplicity = dict()
    for n in range(3, dim):
        lambda_n = lambda_i(n)
        itx_wanted = n ** 2
        itx_now = itx_calculator_mult(lambdas_multiplicity, lambda_n)
        delta_itx = itx_wanted - itx_now
        lambda_to_add = lambda_i(n - 1)
        quant_will_add = itx_termer(lambda_to_add, lambda_n)
        mult_2_add = math.floor(delta_itx / quant_will_add)
        if lambda_to_add not in lambdas_multiplicity:
            lambdas_multiplicity[lambda_to_add] = 0
        lambdas_multiplicity[lambda_to_add] = mult_2_add
    return lambdas_multiplicity


def quant(x):
    return x / (1 - x)


def itx_termer(lambda_, lambda_max):
    quant_now = quant(lambda_)
    quant_max = quant(lambda_max)
    frac = quant_max / quant_now
    term = np.log(frac)
    return term


def itx_calculator_mult(lambdas_multiplicity, lambda_max):
    itx = 0
    for lambda_, mult in lambdas_multiplicity.items():
        term = itx_termer(lambda_, lambda_max)
        itx += term * mult

    return itx


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


def get_itx_lower_bound(lambdas, beta):
    n_beta = beta
    lambda_max = lambda_i(n_beta - 1)
    c_nI = 0
    for i in range(3, n_beta - 2):
        c_nI += itx_termer(lambda_i(i), lambda_max) * lambdas[lambda_i(i)]
    i_tx = c_nI
    return i_tx


def get_itx_upper_bound(lambdas, beta):
    itx = get_itx_lower_bound(lambdas, beta + 1)
    return itx


def equation_15(beta, lambdas, lambdas_comps, cumulative_lambdas_dct, upper=False):
    lambdas_lg = {np.log(lambda_): mult for lambda_, mult in lambdas.items()}
    lambdas_comps_lg = {np.log(lambda_): mult for lambda_, mult in lambdas_comps.items()}
    comps_geometric_mean_ = comp_geometric_mean(lambdas_comps_lg, n_beta=beta - 1)
    geometric_mean_ = geometric_mean(lambdas_lg, n_beta=beta - 1)
    if upper:
        i_tx = get_itx_upper_bound(lambdas, beta)
    else:
        i_tx = get_itx_lower_bound(lambdas, beta)
    i_ty = ity_calcer(comps_geometric_mean_, geometric_mean_, cumulative_lambdas_dct, i_tx, beta)
    return i_tx, i_ty


def ity_calcer(comps_geometric_mean_, geometric_mean_, cumulative_lambdas_dct, i_tx, beta):
    n_beta = n_I_(beta, cumulative_lambdas_dct)
    inner_log_term = comps_geometric_mean_ + np.exp(2 * i_tx / n_beta) * geometric_mean_
    i_ty = i_tx - n_beta * np.log(inner_log_term) / 2.
    return i_ty


def n_I_(beta, cumulative_lambdas_dct):
    accumulation = cumulative_lambdas_dct[lambda_i(beta)]
    return accumulation


def comp_geometric_mean(lambdas_dct, n_beta):
    geo_sum = total_elem = 0
    for i in range(3, n_beta + 1):
        lambda_ = np.log(1 - lambda_i(i))
        geo_sum += lambdas_dct[lambda_] * lambda_
        total_elem += lambdas_dct[lambda_]
    mean = geo_sum / total_elem
    final = np.exp(mean)
    return final


def geometric_mean(lambdas_dct, n_beta):
    geo_sum = total_elem = 0
    for i in range(3, n_beta + 1):
        lambda_ = np.log(lambda_i(i))
        geo_sum += lambdas_dct[lambda_] * lambda_
        total_elem += lambdas_dct[lambda_]
    mean = geo_sum / total_elem
    final = np.exp(mean)
    return final


if __name__ == '__main__':
    main(10_000)
