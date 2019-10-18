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

class EmptyGRB(object):
    '''EmptyGRB for Bilby signal injections. '''

    def __init__(self):
        super(EmptyGRB, self).__init__()
        self.bin_left   = None
        self.bin_right  = None
        self.rates      = None

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
                        priors_bg_hi        = 1e4,  #
                        priors_td_lo        = 0,    # IMPORTANT
                        priors_td_hi        = 30,   # IMPORTANT
                        priors_mr_lo        = 0.2, #
                        priors_mr_hi        = 1,    #
                        priors_tau_lo       = 1e-3,
                        priors_tau_hi       = 1e1,
                        priors_xi_lo        = 1e-2,
                        priors_xi_hi        = 1e3,
                        priors_gamma_min    = 1e-1,
                        priors_gamma_max    = 1e1,
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_scale_min    = 1e2,
                        priors_scale_max    = 1e9):

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
        self.keys    = []
        self.priors  = bilbyPriorDict()

        self.GRB = BATSEpreprocess.BATSESignal(
            self.trigger, times = (self.start, self.end),
            datatype = self.datatype, bgs = False)

        self.num_pulses = 0
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)
        for i in range(4):
            plt.plot(self.GRB.bin_left, self.GRB.rates[:,i])
        plot_name = self.outdir + '/injected_signal'
        plt.savefig(plot_name)

    def get_trigger_label(self):
        tlabel = str(self.trigger)
        if len(tlabel) < 4:
            tlabel = ''.join('0' for i in range(4-len(tlabel))) + tlabel
        return tlabel

    def get_directory_name(self):
        directory = self.tlabel + '_model_comparison_' + str(self.nSamples)
        if 'lens' in self.model:
            directory += '/lens_model'
        else:
            directory += '/null_model'
        directory += '_' + str(self.num_pulses)
        if 'FREDx' in self.model:
            directory += '_FREDx'
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

    def make_priors(self, FRED, FREDx, gaussian, lens, constraint):
        self.keys   = []
        if constraint:
            self.priors = bilbyPriorDict(conversion_function = constraint)
        else:
            self.priors = bilbyPriorDict()
        self.add_background_prior()
        self.add_pulse_priors(FRED, FREDx)
        self.add_gaussian_priors(gaussian)
        if lens:
            self.add_lens_priors()
        self.populate_priors()
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

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
        if count_FRED is not None:
            for i in count_FRED:
                keys = ['{}_{}'.format(list[k], i) for k in range(len(list))]
                self.keys += keys
                for key in keys:
                    self.priors[key] = None

        if count_FREDx is not None:
            list.append('gamma')
            list.append('nu')
            for i in count_FREDx:
                keys = ['{}_{}'.format(list[k], i) for k in range(len(list))]
                self.keys += keys
                for key in keys:
                    self.priors[key] = None

    def add_gaussian_priors(self, count):
        if count is not None:
            list = ['start', 'scale', 'sigma']
            for i in count:
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
                if int(n) > 1:
                    c_key = 'constraint_{}'.format(n)
                    self.priors[c_key] = bilbyConstraint(
                        minimum = 0,
                        maximum = float(self.GRB.bin_right[-1]) )

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
                    latex_label='$\\nu_{}$'.format(n), unit = ' ')

            elif 'sigma' in key:
                print('Sigma priors not set')
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_xi_lo,
                    maximum = self.priors_xi_hi,
                    latex_label= '$\\sigma_{}'.format(n), unit = ' ')

            elif 't_0' in key:
                pass

            else:
                print('Key not found : {}'.format(key))

    def plot_rates(self, priors, rate_function, channels):
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

        nbins = int( (self.GRB.bin_left[-1] - self.GRB.bin_left[0]) / 0.005 )
        bins  = np.linspace(self.GRB.bin_left[0], self.GRB.bin_left[-1], nbins)
        for i in channels:
            print(channels)
            print(i)
            result_label = self.fstring + '_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'

            result = bilby.result.read_in_result(filename=open_result)
            MAP = dict()
            for i in range(1, self.num_pulses + 1):
                try:
                    key = 'constraint_' + str(i)
                    del priors[key]
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

    def inject_signal(self):
        self.model  = 'one FRED pulse lens'
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = True,
                            constraint = None)

        sample = self.priors.sample()
        sample['background'] = 2
        sample['time_delay'] = 17
        sample['magnification_ratio'] = 0.4
        sample['start_1'] = 2
        sample['scale_1'] = 3e5
        sample['tau_1']   = 2
        sample['xi_1']    = 3

        print(sample)
        self.GRB = EmptyGRB()
        times = np.arange(800) * 0.064 - 2 ## = 51.2
        dt = np.diff(times)
        t_0 = -2
        test_rates = self.one_FRED_lens_rate(dt, t_0, **sample)
        noise_rates = np.random.poisson(test_rates)
        plt.plot(times, test_rates)
        plt.plot(times, noise_rates, c='k', alpha = 0.3)
        plot_name = self.outdir + '/injected_signal'
        plt.savefig(plot_name)

        final_rates = np.zeros((len(times),1))
        final_rates[:,0] = noise_rates
        self.GRB.bin_left, self.GRB.rates = times, final_rates
        self.GRB.bin_right = self.GRB.bin_left + 0.064
        widths = self.GRB.bin_right - self.GRB.bin_left
        self.GRB.counts = self.GRB.rates * widths

    def main(self,  rate_function, channels = np.arange(4), test = False):
        widths     = self.GRB.bin_right - self.GRB.bin_left
        deltat     = np.diff(self.GRB.bin_left)
        evidences  = []

        for i in channels:
            self.priors['t_0'] = bilbyDeltaFunction(
                peak = float(self.GRB.bin_left[0]), name = None,
                latex_label = None, unit = None )
            if not test:
                counts   = np.rint(self.GRB.counts[:,i]).astype('uint')
            else:
                counts   =  np.rint(self.GRB.counts).astype('uint')
            likelihood   = bilbyPoissonLikelihood(  deltat, counts,
                                                    rate_function)

            result_label = self.fstring + '_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            try:
                result = bilby.result.read_in_result(filename=open_result)
            except:
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

        self.plot_rates(priors = self.priors.copy(),
                        rate_function = rate_function,
                        channels = channels)
        return evidences

    def two_FRED(self, **kwargs):
        self.model  = 'two FRED pulse'
        self.num_pulses = 2
        self.make_priors(   FRED = [1, 2], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = two_pulse_constraints)
        for key in self.priors:
            print(key)
        evidences = self.main(self.two_FRED_rate, **kwargs)

    def one_FRED_lens(self, **kwargs):
        self.model  = 'one FRED lens'
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = True,
                            constraint = None)
        for key in self.priors:
            print(key)
        evidences = self.main(self.one_FRED_lens_rate, **kwargs)

def two_pulse_constraints(parameters):
    parameters['constraint_2'] = parameters['start_2'] - parameters['start_1']
    return parameters


if __name__ == '__main__':
    test = BilbyObject(973, times = (-2, 50),
                datatype = 'discsc', nSamples = 70, sampler = 'Nestle')

    # test.inject_signal()
    evidences = test.two_FRED(channels = [0], test = False)
    evidences = test.one_FRED_lens(channels = [0], test = False)
