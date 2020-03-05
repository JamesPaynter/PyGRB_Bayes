import numpy as np
from scipy.signal import convolve
import sys,os
MIN_FLOAT = sys.float_info[3]








def gaussian_pulse(times, start, scale, sigma):
    """ start = mode = mu, sigma is std dev = width of pulse.
        scale is height."""
    return scale * np.exp(- np.power(times - start, 2.) / (
                        2 * np.power(sigma, 2.) + MIN_FLOAT))


def exp_decay(times, tau, start):
    dex = np.where(times - start <= 0, MIN_FLOAT,
            np.exp( - (times - start) / (tau + MIN_FLOAT)))
    return dex

def exp_gauss(times, start, scale, sigma, tau):
    s = np.mean(times)
    conv  = convolve(gaussian_pulse(times, s, 1, sigma),
                    exp_decay(times,tau,start), 'same')
    conv /= np.max(conv) + MIN_FLOAT
    return scale * conv


def func(**kwargs):
    a = kwargs.get('a')
    b = kwargs.get('b', [])
    # print(b)
    for i in range(a):
        b.append(i)
    print(b)

if __name__ == '__main__':
    di = {}
    di['a'] = 4
    # di['b'] = None
    func(**di)
    di['a'] = 9
    func(**di)
    #
    # import matplotlib.pyplot as plt
    # x = np.linspace(-2, 20, 1000)
    # # y = FRED_pulse(x, start = -1, scale = 5, tau = 1, xi = 1)
    # y = exp_gauss(x, start = 1, scale = 5, sigma = .6, tau = 1)
    # plt.plot(x,y)
    # plt.show()
    pass
