import numpy as np

class RateFunctionWrapper(object):
    """docstring for RateFunctionWrapper."""

    def __init__(self):
        super(RateFunctionWrapper, self).__init__()

    @staticmethod
    def one_pulse_rate(     delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12

        rates = background + scale_1 * np.exp(- xi_1 * ((tau_1 / times_1)
                                                    + (times_1 / tau_1)) )
        return np.multiply(rates, widths)

    @staticmethod
    def one_FREDx_rate(     delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1, gamma_1, nu_1):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12

        rates = background + scale_1 * np.exp(
                    - np.power(xi_1 * (tau_1 / times_1), gamma_1)
                    - np.power(xi_1 * (times_1 / tau_1), nu_1)   )
        return np.multiply(rates, widths)

    @staticmethod
    def one_FREDx_one_Gauss_rate(     delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1, gamma_1, nu_1,
                            start_2, scale_2, sigma_2):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = (times - start_2) * np.heaviside(times - start_2, 0) + 1e-12

        rates = background + scale_1 * np.exp(
                    - np.power(xi_1 * (tau_1 / times_1), gamma_1)
                    - np.power(xi_1 * (times_1 / tau_1), nu_1)   )
        rates+= scale_2 * np.exp(- np.power(times_2 / sigma_2, 2) )
        return np.multiply(rates, widths)

    @staticmethod
    def one_FRED_lens_rate(    delta_t, t_0, background,
                                time_delay, magnification_ratio,
                                start_1, scale_1, tau_1, xi_1):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = ( times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_0 = ((times - start_1 - time_delay) * np.heaviside(
                                times - start_1 - time_delay, 0) + 1e-12 )
        rates  = scale_1 * np.exp(- xi_1 * ((tau_1 / times_1)
                                        + (times_1 / tau_1)))

        rates += magnification_ratio * scale_1 * np.exp(- xi_1 * (
                                    (tau_1 / times_0) + (times_0 / tau_1)) )
        rates += background
        return np.multiply(rates, widths)


    @staticmethod
    def two_FRED_rate(      delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1,
                            start_2, scale_2, tau_2, xi_2):

        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = (times - start_2) * np.heaviside(times - start_2, 0) + 1e-12

        rates =( background + scale_1 * np.exp(- xi_1 * ((tau_1 / times_1)
                                                     + (times_1 / tau_1)) )
                            + scale_2 * np.exp(- xi_2 * ((tau_2 / times_2)
                                                     + (times_2 / tau_2)) ) )
        return np.multiply(rates, widths)
