import sys
import numpy as np
import scipy.special as special

MIN_FLOAT = sys.float_info[3]



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
    return np.where(times - start <= 0, MIN_FLOAT, scale * np.exp(
    - np.power(xi * (tau / (times - start)), gamma)
    - np.power(xi * ((times - start) / tau), nu)) )



def sine_gaussian(times, sg_A, res_begin, sg_lambda, sg_omega, sg_phi):
    return (sg_A * np.exp(- np.square((times - res_begin) / sg_lambda)) *
            np.cos(sg_omega * times + sg_phi) )



def modified_bessel(times, bes_A, bes_Omega, bes_s, res_begin, bes_Delta):
    return np.where(times > res_begin + bes_Delta / 2.,
            bes_A * special.j0(bes_s * bes_Omega *
           (- res_begin + times - bes_Delta / 2.) ),
           (np.where(times < res_begin - bes_Delta / 2.,
            bes_A * special.j0(bes_Omega *
           (res_begin - times - bes_Delta / 2.) ),
           bes_A)))



def convolution_gaussian(times, parameters):
    return parameters



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(-2, 5, 1000)
    # y = FRED_pulse(x, start = -1, scale = 5, tau = 1, xi = 1)
    y = FREDx_pulse(x, start = -1, scale = 5, tau = 1, xi = 1, gamma = 3, nu = 1)
    plt.plot(x,y)
    plt.show()
    pass
