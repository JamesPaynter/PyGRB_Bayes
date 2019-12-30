import numpy as np
import scipy.special as special

class RateFunctionWrapper(object):
    """docstring for RateFunctionWrapper."""

    def __init__(self):
        super(RateFunctionWrapper, self).__init__()

    @staticmethod
    def single_FRED(        delta_t, t_0,
                            start, scale, tau, xi):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0

        times_1 = (times - start) * np.heaviside(times - start, 0) + 1e-12

        rates = scale * np.exp(- xi * ( (tau / times_1) + (times_1 / tau) - 2))
        return rates


    @staticmethod
    def one_FRED_rate(      delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1,
                            sg_A_1, sg_begin_1, sg_tau_1, sg_omega_1, sg_phi_1):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        # widths= np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12

        rates = background + scale_1 * np.exp(- xi_1 *
                                ( (tau_1 / times_1) + (times_1 / tau_1) - 2))

        rates+= (sg_A_1 * np.exp(- np.square((times - sg_begin_1) / sg_tau_1)) *
                        np.cos(sg_omega_1 * (times - sg_begin_1) + sg_phi_1) )
        # rates+= self.sine_gaussian(times, sg_A_1, sg_begin_1, sg_tau_1, sg_omega_1, sg_phi_1)
        if np.any(rates < 0.):
            return np.zeros(len(rates))
        else:
            return rates

    @staticmethod
    def one_FREDx_rate(     delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1, gamma_1, nu_1):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        # widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12

        rates = background + scale_1 * np.exp(
                    - np.power(xi_1 * (tau_1 / times_1), gamma_1)
                    - np.power(xi_1 * (times_1 / tau_1), nu_1)   )
        return rates
        # return np.multiply(rates, widths)

    @staticmethod
    def one_FRED_lens_rate( delta_t, t_0, background,
                            time_delay, magnification_ratio,
                            start_1, scale_1, tau_1, xi_1,
                            sg_A_1, sg_begin_1, sg_tau_1, sg_omega_1, sg_phi_1):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        # widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = ( times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_0 = ((times - start_1 - time_delay) * np.heaviside(
                                times - start_1 - time_delay, 0) + 1e-12 )
        rates  = scale_1 * np.exp(- xi_1 * ((tau_1 / times_1)
                                        + (times_1 / tau_1) - 2))

        rates += (sg_A_1 * np.exp(- np.square((times - sg_begin_1) / sg_tau_1))*
                  np.cos(sg_omega_1 * (times - sg_begin_1) + sg_phi_1) )

        rates += magnification_ratio * scale_1 * np.exp(- xi_1 * (
                                    (tau_1 / times_0) + (times_0 / tau_1) - 2) )
        rates += background
        if np.any(rates < 0.):
            return np.zeros(len(rates))
        else:
            return rates
        # return np.multiply(rates, widths)

    @staticmethod
    def two_pulse_contraints(parameters):
        parameters['constraint_2'] = (  parameters['start_2']
                                      - parameters['start_1'] )
        parameters['constraint_2_res'] = (  parameters['sg_begin_2']
                                      - parameters['sg_begin_1'] )
        return parameters

    @staticmethod
    def two_FRED_rate(      delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1,
                            start_2, scale_2, tau_2, xi_2,
                            sg_A_1, sg_begin_1, sg_tau_1, sg_omega_1, sg_phi_1,
                            sg_A_2, sg_begin_2, sg_tau_2, sg_omega_2, sg_phi_2):


        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        # widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = (times - start_2) * np.heaviside(times - start_2, 0) + 1e-12

        rates =( background + scale_1 * np.exp(- xi_1 * ((tau_1 / times_1)
                                                     + (times_1 / tau_1) - 2) )
                            + scale_2 * np.exp(- xi_2 * ((tau_2 / times_2)
                                                     + (times_2 / tau_2) - 2)) )
        rates += (sg_A_1 * np.exp(- np.square((times - sg_begin_1) / sg_tau_1))*
                  np.cos(sg_omega_1 * (times - sg_begin_1) + sg_phi_1) )
        rates += (sg_A_2 * np.exp(- np.square((times - sg_begin_2) / sg_tau_2))*
                  np.cos(sg_omega_2 * (times - sg_begin_2) + sg_phi_2) )
        if np.any(rates < 0.):
            return np.zeros(len(rates))
        else:
            return rates

    @staticmethod
    def three_FRED_rate(    delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1,
                            start_2, scale_2, tau_2, xi_2,
                            start_3, scale_3, tau_3, xi_3):

        times   = np.cumsum(delta_t)
        times   = np.insert(times, 0, 0.0)
        times  += t_0

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = (times - start_2) * np.heaviside(times - start_2, 0) + 1e-12
        times_3 = (times - start_3) * np.heaviside(times - start_3, 0) + 1e-12

        rates   =               (
        scale_1 * np.exp(- xi_1 * ((tau_1 / times_1) + (times_1 / tau_1) - 2))
      + scale_2 * np.exp(- xi_2 * ((tau_2 / times_2) + (times_2 / tau_2) - 2))
      + scale_3 * np.exp(- xi_3 * ((tau_3 / times_3) + (times_3 / tau_3) - 2))
                                )
        rates  += background
        return rates

    @staticmethod
    def two_FRED_lens_rate(     delta_t, t_0, background,
                                time_delay, magnification_ratio,
                                start_1, scale_1, tau_1, xi_1,
                                start_2, scale_2, tau_2, xi_2):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0

        times_1 = ( times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = ( times - start_2) * np.heaviside(times - start_2, 0) + 1e-12

        times_0a= ((times - start_1 - time_delay) * np.heaviside(
                                times - start_1 - time_delay, 0) + 1e-12 )
        times_0b= ((times - start_2 - time_delay) * np.heaviside(
                                times - start_2 - time_delay, 0) + 1e-12 )

        rates   =               (
        scale_1 * np.exp(- xi_1 * ((tau_1 / times_1) + (times_1 / tau_1) - 2))
      + scale_2 * np.exp(- xi_2 * ((tau_2 / times_2) + (times_2 / tau_2) - 2))
                                )

        rates += magnification_ratio *  (
        scale_1 * np.exp(- xi_1 * ((tau_1 / times_0a) + (times_0a / tau_1) - 2))
      + scale_2 * np.exp(- xi_2 * ((tau_2 / times_0b) + (times_0b / tau_2) - 2))
                                        )
        rates += background
        return rates

    @staticmethod
    def four_FRED_rate(     delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1,
                            start_2, scale_2, tau_2, xi_2,
                            start_3, scale_3, tau_3, xi_3,
                            start_4, scale_4, tau_4, xi_4):

        times   = np.cumsum(delta_t)
        times   = np.insert(times, 0, 0.0)
        times  += t_0

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = (times - start_2) * np.heaviside(times - start_2, 0) + 1e-12
        times_3 = (times - start_3) * np.heaviside(times - start_3, 0) + 1e-12
        times_4 = (times - start_4) * np.heaviside(times - start_4, 0) + 1e-12

        rates   =               (
        scale_1 * np.exp(- xi_1 * ((tau_1 / times_1) + (times_1 / tau_1) - 2))
      + scale_2 * np.exp(- xi_2 * ((tau_2 / times_2) + (times_2 / tau_2) - 2))
      + scale_3 * np.exp(- xi_3 * ((tau_3 / times_3) + (times_3 / tau_3) - 2))
      + scale_4 * np.exp(- xi_4 * ((tau_4 / times_4) + (times_4 / tau_4) - 2))
                                )
        rates  += background
        return rates



    @staticmethod
    def residuals_bessel(times, bes_A, bes_Omega, bes_s, bes_t_0, bes_Delta):
        return np.where(times > bes_t_0 + bes_Delta / 2,
                bes_A * special.j0(bes_s * bes_Omega *
               (- bes_t_0 + times - bes_Delta / 2) ),
               (np.where(times < bes_t_0 - bes_Delta / 2,
                bes_A * special.j0(bes_Omega *
               (bes_t_0 - times - bes_Delta / 2) ),
               bes_A)))

    @staticmethod
    def sine_gaussian(times, sg_A, sg_begin, sg_tau, sg_omega, sg_phi):
        return (sg_A * np.exp(- np.square((times - sg_begin) / sg_tau)) *
                np.cos(sg_omega * times + sg_phi) )


    @staticmethod
    def one_FREDx_one_Gauss_rate(     delta_t, t_0, background,
                            start_1, scale_1, tau_1, xi_1, gamma_1, nu_1,
                            start_2, scale_2, sigma_2):
        times = np.cumsum(delta_t)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        # widths = np.hstack((delta_t, delta_t[-1]))

        times_1 = (times - start_1) * np.heaviside(times - start_1, 0) + 1e-12
        times_2 = (times - start_2) * np.heaviside(times - start_2, 0) + 1e-12

        rates = background + scale_1 * np.exp(
                    - np.power(xi_1 * (tau_1 / times_1), gamma_1)
                    - np.power(xi_1 * (times_1 / tau_1), nu_1)   )
        rates+= scale_2 * np.exp(- np.power(times_2 / sigma_2, 2) )
        # return np.multiply(rates, widths)
        return rates
