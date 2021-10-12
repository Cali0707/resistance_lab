import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def fit_to_linear(x_data, y_data, x_err, y_err, title):
    def linear_function(x, m, b):
        return (m * x) + b

    popt, pcov = optimize.curve_fit(linear_function, x_data, y_data)

    m_opt = popt[0]
    u_m = pcov[0, 0]
    b_opt = popt[1]
    u_b = pcov[1, 1]

    print(f'm = {m_opt}, u_m = {u_m}, b = {b_opt}, u_b = {u_b}')

    start = min(x_data)
    end = max(x_data)
    x_values = np.arange(start, end, (end - start) / 1000)
    curve = linear_function(x_values, m_opt, b_opt)

    plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='.', label="Actual (I,V) points measured")
    plt.plot(x_values, curve, label="Linearly fitted curve")
    plt.xlabel("I[mA]")
    plt.ylabel("V[V]")
    plt.title(title)
    plt.legend()
    plt.show()


def calc_ra(v, i, r):
    return (v / (i * (10 ** 3))) - r


def calc_rv(r, v, i):
    return v / (i - (v / r))


if __name__ == '__main__':
    data = np.loadtxt("data_1.txt", skiprows=1, unpack=True)
    r = data[0]
    u_r = data[1]
    v = data[2]
    u_v = data[3]
    i = data[4]
    u_i = data[5]
    fit_to_linear(i, v, u_i, u_v, "Graph of I vs V for Circuit 1")
    r_a = calc_ra(v, i, r)
    print(f'Calculated Ra = {r_a}, mean = {np.mean(r_a)}')

    data = np.loadtxt("data_2.txt", skiprows=1, unpack=True)
    r = data[0]
    u_r = data[1]
    v = data[2]
    u_v = data[3]
    i = data[4]
    u_i = data[5]
    fit_to_linear(i, v, u_i, u_v, "Graph of I vs V for Circuit 2")
    r_v = calc_rv(r, v, i)
    print(f'Calculated Rv = {r_v}, mean = {np.mean(r_v)}')
