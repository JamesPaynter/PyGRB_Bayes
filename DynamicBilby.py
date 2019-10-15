import re
import sys
import numpy as np

from scipy.signal           import savgol_filter
from scipy.special          import gammaln

import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec

import bilby
from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction
from bilby.core.likelihood  import Analytical1DLikelihood
from bilby.core.likelihood  import PoissonLikelihood as bilbyPoissonLikelihood


import BATSEpreprocess
from rate_functions import RateFunctionWrapper

class BilbyObject(RateFunctionWrapper):
    ''' Wrapper object for Bayesian analysis. '''

    def __init__(self,  trigger, times, datatype, satellite = 'BATSE',
                        model               = 'lens_model',
                        ## are your bins the right size in rate function ????
                        sampler             = 'dynesty',
                        verbose             = True,
                        nSamples            = 200,
                        priors_pulse_start  = -10,# IMPORTANT  ##cuts GRB rates
                        priors_pulse_end    =  50,# IMPORTANT ##cuts GRB rates
                        priors_bg_lo        = 1e-1, #
                        priors_bg_hi        = 3e3,  #
                        priors_td_lo        = 0,    # IMPORTANT
                        priors_td_hi        = 30,   # IMPORTANT
                        priors_mr_lo        = 0.2, #
                        priors_mr_hi        = 1,    #
                        priors_tau_lo       = 1,
                        priors_tau_hi       = 1e3,
                        priors_xi_lo        = 1,
                        priors_xi_hi        = 1e3,
                        priors_gamma_min    = 1e-1,
                        priors_gamma_max    = 1e1,
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_scale_min    = 1e3,
                        priors_scale_max    = 1e8):

        super(BilbyObject, self).__init__()

        print('DO THE PRIORS MAKE SENSE !! ??')
        (self.start, self.end)   = times
        self.colours             = ['red', 'orange', 'green', 'blue']
        self.clabels             = ['1', '2', '3', '4']
        self.datatype            = datatype
        self.satellite           = satellite
        self.sampler             = sampler
        self.nSamples            = nSamples
        self.model               = model
        self.trigger             = trigger
        self.priors_pulse_start  = priors_pulse_start
        self.priors_pulse_end    = priors_pulse_end
        self.priors_bg_lo        = priors_bg_lo
        self.priors_bg_hi        = priors_bg_hi
        self.priors_td_lo        = priors_td_lo
        self.priors_td_hi        = priors_td_hi
        self.priors_mr_lo        = priors_mr_lo
        self.priors_mr_hi        = priors_mr_hi
        self.priors_tau_lo       = priors_tau_lo
        self.priors_tau_hi       = priors_tau_hi
        self.priors_xi_lo        = priors_xi_lo
        self.priors_xi_hi        = priors_xi_hi
        self.priors_gamma_min    = priors_gamma_min
        self.priors_gamma_max    = priors_gamma_max
        self.priors_nu_min       = priors_nu_min
        self.priors_nu_max       = priors_nu_max
        self.priors_scale_min    = priors_scale_min
        self.priors_scale_max    = priors_scale_max

        self.verbose = verbose
        self.keys = []
        self.priors = bilbyPriorDict()
        # self.bilby_kwargs = self.priors.copy()
        self.make_priors(FRED = 1, FREDx = 0, gaussian = 0, lens = True)
        self.populate_priors()
        for key in self.priors:
            print(key)
        print(self.priors)


        self.GRB = BATSEpreprocess.BATSESignal(
            self.trigger, times = (self.start, self.end),
            datatype = self.datatype, bgs = False)

        self.model      = 'two pulse'
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)
        self.one_pulse_lens_main()
        self.plot_rates(priors = self.priors.copy(),
                        rate_function = self.one_pulse_lens_rate)

    def get_trigger_label(self):
        tlabel = str(self.trigger)
        if len(tlabel) < 4:
            tlabel = ''.join('0' for i in range(4-len(tlabel))) + tlabel
        return tlabel

    def get_directory_name(self):
        directory = self.tlabel + '_model_comparison_' + str(self.nSamples)
        if self.model == 'lens_model':
            directory += '/lens_model'
        else:
            directory += '/multipulse_model'
        return directory

    def get_file_string(self):
        file_string = ''
        if self.satellite == 'BATSE':
            file_string += 'B_'
        file_string += self.tlabel
        if   self.datatype == 'discsc':
            file_string += '__d'
        elif self.datatype == 'TTE':
            file_string += '__t'
        elif self.datatype == 'TTElist':
            file_string += '_tl'
        if 'lens' in self.model:
            file_string += '_YL'
        else:
            file_string +='_NL'
        file_string += str(self.nSamples) + '_'
        return file_string

    def make_priors(self, FRED, FREDx, gaussian, lens):
        self.add_background_prior()
        self.add_pulse_priors(FRED, FREDx)
        self.add_gaussian_priors(gaussian)
        if lens:
            self.add_lens_priors()

    def add_background_prior(self):
        list = ['background']
        keys = ['{}'.format(list[k]) for k in range(len(list))]
        self.keys += keys
        for key in keys:
            self.priors[key] = None

    def add_lens_priors(self):
        list = ['time_delay', 'magnification_ratio']
        keys = ['{}'.format(list[k]) for k in range(len(list))]
        self.keys += keys
        for key in keys:
            self.priors[key] = None

    def add_pulse_priors(self, count_FRED, count_FREDx):
        list = ['start', 'scale', 'tau', 'xi']
        for i in range(1, count_FRED + 1):
            keys = ['{}_{}'.format(list[k], i) for k in range(len(list))]
            self.keys += keys
            for key in keys:
                self.priors[key] = None

        list.append('gamma')
        list.append('nu')
        for i in range(count_FRED + 1, count_FREDx + count_FRED + 1):
            keys = ['{}_{}'.format(list[k], i) for k in range(len(list))]
            self.keys += keys
            for key in keys:
                self.priors[key] = None

    def add_gaussian_priors(self, count):
        list = ['gauss_strt', 'mu', 'sigma']
        for i in range(1, count + 1):
            keys = ['{}_{}'.format(list[k], i) for k in range(len(list))]
            self.keys += keys
            for key in keys:
                self.priors[key] = None

    def populate_priors(self):
        ''' initialise priors

            Pass in **kwargs, then overwrite pulse parameters as
            applicable. Otherwise take generic parameters defined in init.

            add kwargs to list ??
        '''
        for key in self.keys:
            for i in range(10):
                if str(i) in key:
                    n = str(i)
                else:
                    pass
            ## find integer in key and put in label
            if key == 'background':
                self.priors[key] = bilbyLogUniform(
                minimum = self.priors_bg_lo,
                maximum = self.priors_bg_hi,
                latex_label='Bg',
                unit = 'counts / sec')

            elif key == 'time_delay':
                self.priors[key] = bilbyUniform(
                minimum = self.priors_td_lo,
                maximum = self.priors_td_hi,
                latex_label='$\\Delta t$',
                unit = ' seconds ')

            elif key == 'magnification_ratio':
                self.priors[key] = bilbyUniform(
                minimum = self.priors_mr_lo,
                maximum = self.priors_mr_hi,
                latex_label='$\\Delta \\mu$',
                unit = ' ')

            elif 'start' in key:
                self.priors[key] = bilbyUniform(
                    minimum = self.priors_pulse_start,
                    maximum = self.priors_pulse_end,
                    latex_label = '$\\Delta_{}$'.format(n), unit = 'sec')

            elif 'scale' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_scale_min,
                    maximum = self.priors_scale_max,
                    latex_label='$A_{}$'.format(n), unit = 'counts / sec')

            elif 'tau' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_tau_lo,
                    maximum = self.priors_tau_hi,
                    latex_label='$\\tau_{}$'.format(n), unit = ' ')

            elif 'xi' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_xi_lo,
                    maximum = self.priors_xi_hi,
                    latex_label='$\\xi_{}$'.format(n), unit = ' ')

            elif 'gamma' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_gamma_min,
                    maximum = self.priors_gamma_max,
                    latex_label='$\\gamma_{}$'.format(n), unit = ' ')

            elif 'nu' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_nu_min,
                    maximum = self.priors_nu_max,
                    latex_label='$\\nu_{}'.format(n), unit = ' ')


            elif 'gauss_strt' in key:
                self.priors[key] = bilbyUniform(
                    minimum = self.priors_pulse_start,
                    maximum = self.priors_pulse_end,
                    latex_label=' ', unit = ' ')

            elif 'mu' in key:
                print('Mu priors not set')
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_tau_rise_min,
                    maximum = self.priors_tau_rise_max,
                    latex_label= ' ', unit = ' ')

            elif 'sigma' in key:
                print('Sigma priors not set')
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_tau_rise_min,
                    maximum = self.priors_tau_rise_max,
                    latex_label= ' ', unit = ' ')

            elif 't_0' in key:
                pass

            else:
                print('Key not found : {}'.format(key))


    def plot_rates(self, priors, rate_function):
        heights = [5, 1, 1, 1, 1]

        fig2 = plt.figure(constrained_layout=False)
        spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2,
                                height_ratios=heights,
                                hspace=0.0, wspace=0.0)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[1, 0])
        f2_ax3 = fig2.add_subplot(spec2[2, 0])
        f2_ax4 = fig2.add_subplot(spec2[3, 0])
        f2_ax5 = fig2.add_subplot(spec2[4, 0])
        residual_axes = [f2_ax2, f2_ax3, f2_ax4, f2_ax5]

        nbins = int( (self.GRB.bin_left[-1] - self.GRB.bin_left[0]) / 0.004 )
        bins  = np.linspace(self.GRB.bin_left[0], self.GRB.bin_left[-1], nbins)
        for i in range(4):
            result_label = self.fstring + '_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'

            try:
                result = bilby.result.read_in_result(filename=open_result)
            except:
                pass
            MAP = dict()
            try:
                del priors['constraint2']
            except:
                pass
            for parameter in priors:
                summary = result.get_one_dimensional_median_and_error_bar(parameter)
                MAP[parameter] = summary.median
            MAP['t_0'] = float(self.GRB.bin_left[0])
            widths = self.GRB.bin_right - self.GRB.bin_left
            rates_fit  = rate_function(np.diff(self.GRB.bin_left), **MAP) / widths
            # integrated, binss = np.histogram(self.GRB.channels[i], bins=bins)
            # difference = integrated - rates_fit
            difference = self.GRB.rates[:,i] - rates_fit
            # f2_ax1.plot(bins[0:-1], integrated, c = self.colours[i], linewidth=0.5, drawstyle ='steps')
            f2_ax1.plot(self.GRB.bin_left, self.GRB.rates[:,i], c = self.colours[i])
            f2_ax1.plot(self.GRB.bin_left, rates_fit, 'k:') #, label = plot_legend)

            residual_axes[i].plot(self.GRB.bin_left, difference, c = self.colours[i])

        f2_ax1.set_xticks(())
        # f2_ax1.legend(fontsize = 11)
        f2_ax2.set_xticks(())
        f2_ax3.set_xticks(())
        f2_ax4.set_xticks(())
        l = self.outdir + '/' + self.fstring + '_rates.pdf'
        fig2.savefig(l)

    @staticmethod
    def generate_constraints(parameters):
        for key in kwargs:
            for i in range(10):
                if 'start' in key:
                    n = str(i)
                    if int(n) > num_pulses:
                        num_pulses = int(n)

        constraints = ['constraint' + str(i) for i in range(num_pulses)]
        return parameters

    def one_pulse_lens_main(self):
        widths     = self.GRB.bin_right - self.GRB.bin_left
        deltat     = np.diff(self.GRB.bin_left)
        evidences  = []

        for i in range(4):
            self.priors['t_0'] = bilbyDeltaFunction(
                peak = float(self.GRB.bin_left[0]), name = None,
                latex_label = None, unit = None )

            counts       = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood   = bilbyPoissonLikelihood(  deltat, counts,
                                                    self.one_pulse_lens_rate)

            result_label = self.fstring + '_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'

            result = bilby.run_sampler( likelihood = likelihood,
                                        priors     = self.priors,
                                        sampler    = self.sampler,
                                        nlive      = self.nSamples,
                                        outdir     = self.outdir,
                                        label      = result_label,
                                        save       = True)
            try:
                del self.priors['t_0']
            except:
                pass

            plotname = self.outdir + '/' + result_label +'_corner.pdf'
            result.plot_corner(filename = plotname)
            evidences.append(result.log_bayes_factor)
        return evidences

if __name__ == '__main__':
    test = BilbyObject(973, times = (-2, 50), datatype = 'discsc', nSamples = 200, sampler = 'Nestle')

#
#
# dict = {}
# dict['arg1'] = 5
# dict['arg2'] = 13
#
#
# def do_something(**kwargs):
#     for key in kwargs:
#         print(key, ' : ', kwargs[key])
#     def function(kwargs = kwargs):
#         # print(kwargs)
#         return kwargs['arg2']
#     return function
#
# func = do_something(**dict)
# a = func()
# print(a)



    #
    # @staticmethod
    # def generate_rates(delta_t, **kwargs):
    # # def generate_rates(delta_t, t_0, background, start_1, scale_1, rise_1, decay_1):
    #     print('***************')
    #     print(kwargs)
    #     print('***************')
    #     for key in kwargs:
    #         print(key)
    #     times  = np.cumsum(delta_t)
    #     times  = np.insert(times, 0, 0.0)
    #     times += t_0
    #     widths = np.hstack((delta_t, delta_t[-1]))
    #     starts = []
    #     for key in kwargs:
    #         if 'start' in key:
    #             starts.append(int(re.sub(r"\D", "", key)))
    #     if len(starts) > 1:
    #         num_pulses = np.max(np.array(starts))
    #     else:
    #         num_pulses = 1
    #     print(num_pulses)
    #     rates = np.zeros(len(times))
    #
    #     keys  = ['start_', 'scale_', 'rise_', 'decay_', 'times_']
    #     for i in range(num_pulses):
    #         keyss = [keys[j] + str(i + 1) for j in range(len(keys))]
    #         time__      = times - kwargs[keyss[0]]
    #         time_______ = (time__) * np.heaviside(time__, 0) + 1e-12
    #         print(time_______)
    #         print('kwargs[keyss[1]] : ', kwargs[keyss[1]])
    #         print('kwargs[keyss[2]] : ', kwargs[keyss[2]])
    #         print('kwargs[keyss[3]] : ', kwargs[keyss[3]])
    #         rates += kwargs[keyss[1]] * np.exp(
    #                 - np.power( ( kwargs[keyss[2]] / time_______), 1)
    #                 - np.power( ( time_______ / kwargs[keyss[3]]), 1))
    #
    #     rates += kwargs['background']
    #     print(rates)
    #     return np.multiply(rates, widths)
    #
    #
    # @staticmethod
    # def generate_rates1(delta_t, **outer_kwargs):
    #     ''' Dynamically generate the rate function based on the
    #         input parameters.
    #
    #         HOW TO GENERATE A STATIC FUNCTION ONCE, THIS FUNCTION WILL CALL
    #         ALL THESE EXTRA KEYS ETC EVERY FUNCTION CALL
    #
    #         IT ONLY NEEDS TO BE DONE ONCE it will be the same after
    #     '''
    #     extra_params = {}
    #     num_pulses   = 0
    #     starts = []
    #     for key in outer_kwargs:
    #         if 'start' in key:
    #             starts.append(int(re.sub(r"\D", "", key)))
    #     starts = np.array(starts)
    #     if len(starts) > 1:
    #         num_pulses = np.max(starts)
    #     else:
    #         num_pulses = 1
    #     print('The number of pulses is {}'.format(num_pulses))
    #     list = ['times']
    #     # for key in extra_params:
    #     #     print(key)
    #
    #     # if 'lens' in model:
    #     #     pass
    #     # else:
    #     #     pass
    #     def rate_function(delta_t, t_0, **inner_kwargs):
    #         times  = np.cumsum(delta_t)
    #         times  = np.insert(times, 0, 0.0)
    #         times += t_0
    #         widths = np.hstack((delta_t, delta_t[-1]))
    #
    #         kwargs = inner_kwargs or outer_kwargs
    #         extra_keys   = []
    #         extra_params = {}
    #         for i in range(1, num_pulses + 1):
    #             extra_keys  += ['{}_{}'.format(list[k],i)
    #                             for k in range(len(list))]
    #             extra_params = {k: None for k in extra_keys}
    #         for i in range(1, num_pulses + 1):
    #             start_key = 'start_' + str(i)
    #             times_key = 'times_' + str(i)
    #             extra_params[times_key] =  ((times - kwargs[start_key]
    #             ) * np.heaviside(times - kwargs[start_key], 0) + 1e-12 )
    #
    #         rates = np.zeros(len(times))
    #         keys  = ['start_', 'scale_', 'rise_', 'decay_', 'times_']
    #         for i in range(1, num_pulses + 1):
    #             for j in range(len(keys)):
    #                 keys[j] += str(i)
    #             rates += ( kwargs[keys[1]] *
    #                     np.exp( - (kwargs[keys[2]] / extra_params[keys[4]])
    #                             - extra_params[keys[4]] / kwargs[keys[3]]) )
    #         return np.multiply(rates, widths)
    #     return rate_function
