import sys
import numpy as np
import scipy.special as special
from scipy.signal import convolve

MIN_FLOAT = sys.float_info[3]
# MIN_FLOAT = 1e-20



def gaussian_pulse(times, start, scale, sigma):
    """ start = mode = mu, sigma is std dev = width of pulse.
        scale is height."""
    return scale * np.exp(- np.power(times - start, 2.) / (
                        2 * np.power(sigma, 2.) + MIN_FLOAT))


def FRED_pulse(times, start, scale, tau, xi):
    return np.where(times - start <= 0, MIN_FLOAT, scale * np.exp( - xi * (
    (tau / np.where(times - start <= 0, times - start - MIN_FLOAT, times - start + MIN_FLOAT))
    + ((times - start) / (tau + MIN_FLOAT)) - 2.)))



def FREDx_pulse(times, start, scale, tau, xi, gamma, nu):
    X   = np.where(times - start <= 0, MIN_FLOAT, np.exp(
        - np.power(xi * (tau / (times - start)), gamma)
        - np.power(xi * ((times - start) / tau), nu)) )
    X  /= np.max(X)
    return X * scale



def sine_gaussian(times, res_begin, sg_A, sg_lambda, sg_omega, sg_phi):
    s = (np.exp(- np.square((times - res_begin) / sg_lambda)) *
         np.cos(sg_omega * times + sg_phi))
    s  /= np.max(np.abs(s))
    return sg_A * s


def modified_bessel(times, bes_A, bes_Omega, bes_s, res_begin, bes_Delta):
    """ Not Tested. """
    b = np.where(times > res_begin + bes_Delta / 2.,
            special.j0(bes_s * bes_Omega *
           (- res_begin + times - bes_Delta / 2.) ),
           (np.where(times < res_begin - bes_Delta / 2.,
            special.j0(bes_Omega *
           (res_begin - times - bes_Delta / 2.) ),
           1)))
    b  /= np.max(np.abs(b))
    return bes_A * b


def exp_decay(times, tau, start):
    dex = np.where(times - start <= 0, MIN_FLOAT,
            np.exp( - (times - start) / (tau + MIN_FLOAT)))
    return dex

def convolution_gaussian(times, start, scale, sigma, tau):
    s = np.mean(times)
    conv  = convolve(gaussian_pulse(times, s, 1, sigma),
                    exp_decay(times,tau,start), 'same')
    # conv /= np.max(conv) + MIN_FLOAT
    return scale * conv



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(-2, 5, 1000)
    # y = FRED_pulse(x, start = -1, scale = 5, tau = 1, xi = 1)
    y = FREDx_pulse(x, start = -1, scale = 5, tau = .12, xi = 1, gamma = 3, nu = 1)
    plt.plot(x,y)
    plt.show()
    pass
