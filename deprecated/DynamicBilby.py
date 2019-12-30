import re
import sys
import pymc3
import numpy as np


from scipy.signal           import savgol_filter
from scipy.special          import gammaln

import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from matplotlib.ticker import MaxNLocator

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

from matplotlib import rc


class EmptyGRB(object):
    '''EmptyGRB for Bilby signal injections. '''

    def __init__(self):
        super(EmptyGRB, self).__init__()
        self.bin_left   = None
        self.bin_right  = None
        self.rates      = None

class BilbyObject(RateFunctionWrapper):
    ''' Wrapper object for Bayesian analysis. '''

    def __init__(self,  trigger, times, datatype,
                        priors_pulse_start, priors_pulse_end,
                        priors_td_lo = None, priors_td_hi = None,
                        test                = False,
                        satellite           = 'BATSE',
                        model               = 'lens_model',
                        ## are your bins the right size in rate function ????
                        sampler             = 'dynesty',
                        verbose             = True,
                        nSamples            = 200,
                        priors_bg_lo        = 1e-1,  ## SCALING IS COUNTS / BIN
                        priors_bg_hi        = 1e3,   ## SCALING IS COUNTS / BIN
                        priors_mr_lo        = 0.2,   ## which means that it is
                        priors_mr_hi        = 1.4,   # 1 / 0.064 times smaller
                        priors_tau_lo       = 1e-3,  # than you think it is
                        priors_tau_hi       = 1e3,   # going to be !!!!!!!!!!!!
                        priors_xi_lo        = 1e-3,
                        priors_xi_hi        = 1e3,
                        priors_gamma_min    = 1e-1,
                        priors_gamma_max    = 1e1,
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_scale_min    = 1e0,   ## SCALING IS COUNTS / BIN
                        priors_scale_max    = 1e4):  ## SCALING IS COUNTS / BIN

        super(BilbyObject, self).__init__()

        print('\n\n\n\n')
        print('DO THE PRIORS MAKE SENSE !! ??')
        print('Prior scaling is in counts / bin !!! ')
        print('THIS IS NOT COUNTS / SECOND !!!')
        print('This should only affect the A and B scale and background params')
        print('\n\n\n\n')


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

        self.MC_counter = None

        if not test:
            self.GRB = BATSEpreprocess.BATSESignal(
                self.trigger, times = (self.start, self.end),
                datatype = self.datatype, bgs = False)
        else:
            self.GRB = EmptyGRB()
            self.GRB.trigger = self.trigger
            self.GRB.start   = self.start
            self.GRB.end     = self.end
            self.GRB.datatype= self.datatype

        self.num_pulses = 0
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

        if not test:
            for i in range(4):
                plt.plot(self.GRB.bin_left, self.GRB.rates[:,i],
                            c = self.colours[i], drawstyle='steps-mid')
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
        if self.MC_counter:
            directory += '_' + str(self.MC_counter)
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
        list = ['start', 'scale', 'tau', 'xi',
                'sg_begin',  'sg_A', 'sg_tau', 'sg_omega', 'sg_phi']
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
                latex_label='B',
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

            elif 'begin' in key:
                self.priors[key] = bilbyUniform(
                    minimum = self.priors_pulse_start,
                    maximum = self.priors_pulse_end,
                    latex_label = '$\\Delta_{}$'.format(n), unit = 'sec')
                if int(n) > 1:
                    c_key = 'constraint_{}_res'.format(n)
                    self.priors[c_key] = bilbyConstraint(
                        minimum = 0,
                        maximum = float(self.GRB.bin_right[-1]) )

            elif 'sg_A' in key:
                self.priors[key] = bilbyLogUniform(1e-1,1e6,latex_label='res $A$')

            elif 'sg_tau' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e3,latex_label='res $\\tau$')

            elif 'sg_omega' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e3,latex_label='res $\\omega$')

            elif 'sg_phi' in key:
                self.priors[key] = bilbyUniform(-np.pi,np.pi,latex_label='res $\\phi$')

            elif 't_0' in key:
                pass

            else:
                print('Key not found : {}'.format(key))



    def get_fit_errors(self, open_result, keys, rate_function, t_array):
        result = bilby.result.read_in_result(filename=open_result)
        # print(result.posterior['background'].values[0:10])

        error_bars = np.zeros((2, len(t_array)))
        nDraws = 1000
        for i in range(1, len(t_array)):
            dt = t_array[i] - t_array[i-1]
            t_0= t_array[i]
            error_bars[:,i] = self.get_point_error(dt, t_0, keys,
                                        rate_function, nDraws, result) / dt
        return error_bars


    def get_point_error(self, dt, t_0, keys, rate_function, nDraws, result):
        # dict = bilbyPriorDict()
        # for key in keys:
        #     dict[key] = np.random.choice(   result.posterior[key].values,
        #                                     size = nDraws, replace = True)

        # dts, t_0s = dt * np.ones(nDraws), t_0 * np.ones(nDraws)
        # y_array = rate_function(dts, t_0s, **dict)
        ## need to fix the rate function definition so this is vectorised
        ## why can't rate function just take in times instead of dt ???

        y_array = np.zeros(nDraws)
        for i in range(nDraws):
            dict = bilbyPriorDict()
            for key in keys:
                dict[key] = np.random.choice(   result.posterior[key].values,
                                                size = 1, replace = True)
            y_array[i] = rate_function(dt, t_0, **dict)[1]
        # upper_err, lower_err = np.quantile(y_array, [0.75, 0.25])
        upper_err, lower_err = pymc3.stats.hpd(y_array, alpha = 0.01)
        return upper_err, lower_err




    def plot_rates(self, priors, rate_function, channels,
                    save_all        = False,
                    residual_fits   = None):
        heights = [5, 1, 1, 1, 1]
        # width = 6.891
        width = 3.321
        # height = (6.891 / 1.8) * 2
        height = (3.321 / 1.8) * 2
        if len(channels) > 1:
            figure, axes = plt.subplots(4, figsize = (width, height * 4))
            fig2  = plt.figure( figsize = (width, height),
                                constrained_layout=False)
            # ax    = fig2.add_subplot(111, frameon=False)
            spec2 = gridspec.GridSpec(ncols=2, nrows=5, figure=fig2,
                                    height_ratios=heights,
                                    width_ratios=[0.05, 0.95],
                                    hspace=0.0, wspace=0.0)
            ax     = fig2.add_subplot(spec2[0:5, 0], frameon=False)
            f2_ax1 = fig2.add_subplot(spec2[0, 1])
            f2_ax2 = fig2.add_subplot(spec2[1, 1])
            f2_ax3 = fig2.add_subplot(spec2[2, 1])
            f2_ax4 = fig2.add_subplot(spec2[3, 1])
            f2_ax5 = fig2.add_subplot(spec2[4, 1])
            residual_axes = [f2_ax2, f2_ax3, f2_ax4, f2_ax5]

            nbins = int( (self.GRB.bin_left[-1] - self.GRB.bin_left[0]) / 0.005 )
            bins  = np.linspace(self.GRB.bin_left[0], self.GRB.bin_left[-1], nbins)

            offsets = [0, 4000, 8000, -3000]
            # offsets = [8000, 5000, 0, 0]
            # offsets = [0, 0, 0, 0]
            for i in channels:
                result_label = self.fstring + '_result_' + self.clabels[i]
                if save_all:
                    result_label += '_' + str(self.counter)
                open_result  = self.outdir + '/' + result_label +'_result.json'

                result = bilby.result.read_in_result(filename=open_result)

                # upper_err, lower_err = self.get_fit_errors(open_result,
                #                     self.keys, rate_function, self.GRB.bin_left)

                MAP = dict()
                for j in range(1, self.num_pulses + 1):
                    try:
                        key = 'constraint_' + str(j)
                        del priors[key]
                        key = 'constraint_' + str(j) + '_res'
                        del priors[key]
                    except:
                        pass
                for parameter in priors:
                    summary = result.get_one_dimensional_median_and_error_bar(parameter)
                    MAP[parameter] = summary.median


                print('************')
                for j in range(1, self.num_pulses + 1):
                    ### for each pulse number extract the residual keys
                    pulse   = dict()
                    ressies = dict()
                    for key in priors:
                    ### now have all prior keys
                        if 'sg' in key and f'_{j}' in key:
                            try:
                                key2 = key.replace(f'_{j}', '')
                                summary = result.get_one_dimensional_median_and_error_bar(key)
                                ressies[key2] = summary.median
                            except:
                                pass
                        elif f'_{j}' in key:
                            try:
                                key2 = key.replace(f'_{j}', '')
                                summary = result.get_one_dimensional_median_and_error_bar(key)
                                pulse[key2] = summary.median
                            except:
                                pass
                    print('************')
                    for key in ressies:
                        print(key, ressies[key])
                    axes[i].plot(self.GRB.bin_left,
                        self.sine_gaussian(self.GRB.bin_left, **ressies),
                        c = self.colours[i])
                    t_0 = float(self.GRB.bin_left[0])
                    axes[i].plot(self.GRB.bin_left,
                        self.single_FRED(np.diff(self.GRB.bin_left), t_0, **pulse),
                        c = 'k')
                    print(f'drew {j}th line')


                MAP['t_0'] = float(self.GRB.bin_left[0])
                widths = self.GRB.bin_right - self.GRB.bin_left
                rates_fit  = rate_function(np.diff(self.GRB.bin_left), **MAP) / widths
                # integrated, binss = np.histogram(self.GRB.channels[i], bins=bins)
                # difference = integrated - rates_fit
                difference = self.GRB.rates[:,i] - rates_fit
                # f2_ax1.plot(bins[0:-1], integrated, c = self.colours[i],
                # linewidth=0.5, drawstyle ='steps')
                f2_ax1.plot(self.GRB.bin_left, self.GRB.rates[:,i] + offsets[i],
                c = self.colours[i], drawstyle='steps-mid', linewidth = 0.4)
                f2_ax1.plot(self.GRB.bin_left, rates_fit + offsets[i],
                'k', linewidth = 0.4) #, label = plot_legend)
                y_err = np.sqrt(self.GRB.rates[:,i] * widths) / widths
                # f2_ax1.fill_between(self.GRB.bin_left,
                #                     y1 = self.GRB.rates[:,i] + offsets[i] + y_err,
                #                     y2 = self.GRB.rates[:,i] + offsets[i] - y_err,
                #                     alpha = 0.2, color = self.colours[i],
                #                     step = 'mid')
                # f2_ax1.plot(self.GRB.bin_left, upper_err+ offsets[i], 'k:', linewidth = 0.4)
                # f2_ax1.plot(self.GRB.bin_left, lower_err+ offsets[i], 'k:', linewidth = 0.4)
                residual_axes[i].plot(self.GRB.bin_left, difference,
                c = self.colours[i], drawstyle='steps-mid', linewidth = 0.4)
                if residual_fits is not None:
                    res = residual_fits[:,i] / widths
                    residual_axes[i].plot(self.GRB.bin_left, res,
                    'k:', linewidth = 0.4)

            f2_ax1.set_xticks(())
            f2_ax1.set_xlim(left  = self.GRB.bin_left[0],
                            right = self.GRB.bin_left[-1])
            # f2_ax1.legend(fontsize = 11)
            f2_ax2.set_xticks(())
            # f2_ax2.yaxis.set_major_locator(MaxNLocator(nbins=2,prune='lower'))
            yticks = f2_ax2.get_yticks()
            print(yticks)
            # f2_ax2.set_yticks([1:])
            # f2_ax2.set_yticks(f2_ax2.get_yticks()[2:4])
            f2_ax2.set_yticks([0, 1000])
            f2_ax2.set_xlim(left  = self.GRB.bin_left[0],
                            right = self.GRB.bin_left[-1])
            f2_ax3.set_xticks(())
            # f2_ax3.set_yticks(f2_ax3.get_yticks()[1:3])
            f2_ax3.set_yticks([0, 1000])
            # f2_ax3.yaxis.set_major_locator(MaxNLocator(nbins=2,prune='lower'))
            f2_ax3.set_xlim(left  = self.GRB.bin_left[0],
                            right = self.GRB.bin_left[-1])
            f2_ax4.set_xticks(())
            # f2_ax4.set_yticks(f2_ax4.get_yticks()[1:3])
            f2_ax4.set_yticks([0, 1000])
            # f2_ax4.yaxis.set_major_locator(MaxNLocator(nbins=2,prune='lower'))
            f2_ax4.set_xlim(left  = self.GRB.bin_left[0],
                            right = self.GRB.bin_left[-1])
            # f2_ax5.set_yticks(f2_ax5.get_yticks()[1:3])
            f2_ax5.set_yticks([0, 1000])
            f2_ax5.set_xlim(left  = self.GRB.bin_left[0],
                            right = self.GRB.bin_left[-1])

            # f2_ax1.set_ylabel('counts / sec')
            f2_ax5.set_xlabel('time since trigger (s)')
            # fig2.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            ax.tick_params(labelcolor='none', top=False,
                            bottom=False, left=False, right=False)
            ax.set_ylabel('counts / sec')

            plt.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=0.16)
            plt.subplots_adjust(right=0.98)
            plt.subplots_adjust(top=0.98)
            plt.subplots_adjust(bottom=0.13)
            f2_ax1.ticklabel_format(axis = 'y', style = 'sci')
            l = self.outdir + '/' + self.fstring + '_rates.pdf'
            q = self.outdir + '/' + self.fstring + '_lines.pdf'
            if residual_fits is not None:
                l = self.outdir + '/' + self.fstring + '_residuals.pdf'
            if save_all:
                l = (self.outdir + '/' + self.fstring + '_rates_' + '_' +
                        str(self.counter) + '.pdf' )
            fig2.savefig(l)
            figure.savefig(q)

        else:
            heights = [5, 1]
            width  = 3.321
            height = 3.321 / 1.6
            fig2   = plt.figure(    figsize = (width, height),
                                    constrained_layout=False)

            spec2  = gridspec.GridSpec(ncols=1, nrows=2, figure=fig2,
                                    height_ratios=heights,
                                    hspace=0.0, wspace=0.0)

            f2_ax1 = fig2.add_subplot(spec2[0, 0])
            f2_ax2 = fig2.add_subplot(spec2[1, 0])
            for i in channels:
                result_label = self.fstring + '_result_' + self.clabels[i]
                if save_all:
                    result_label += '_' + str(self.counter)
                open_result  = self.outdir + '/' + result_label +'_result.json'

                result = bilby.result.read_in_result(filename=open_result)
                MAP = dict()
                for j in range(1, self.num_pulses + 1):
                    try:
                        key = 'constraint_' + str(j)
                        del priors[key]
                    except:
                        pass
                for parameter in priors:
                    summary = result.get_one_dimensional_median_and_error_bar(
                                    parameter)
                    MAP[parameter] = summary.median
                MAP['t_0'] = float(self.GRB.bin_left[0])
                widths = self.GRB.bin_right - self.GRB.bin_left
                rates_fit  = rate_function(np.diff(self.GRB.bin_left),
                                            **MAP) / widths
                difference = self.GRB.rates[:,i] - rates_fit
                f2_ax1.plot(self.GRB.bin_left, self.GRB.rates[:,i],
                                c = self.colours[i], drawstyle='steps-mid',
                                linewidth = 0.5)

                f2_ax1.plot(self.GRB.bin_left, rates_fit, 'k',
                            linewidth = 0.5) #, label = plot_legend)

                f2_ax2.plot(self.GRB.bin_left, difference,
                                        c = self.colours[i],
                                        drawstyle='steps-mid', linewidth = 0.5)

            f2_ax1.set_xlim(self.GRB.bin_left[0], self.GRB.bin_left[-1])
            f2_ax2.set_xlim(self.GRB.bin_left[0], self.GRB.bin_left[-1])
            f2_ax1.set_xticks(())
            yticks = f2_ax2.get_yticks()
            f2_ax2.set_yticks(f2_ax2.get_yticks()[2:4])
            f2_ax1.ticklabel_format(axis = 'y', style = 'sci')
            f2_ax1.set_ylabel('counts / sec')
            f2_ax2.set_xlabel('time since trigger (s)')
            plt.locator_params(axis='y', nbins=4)
            # plt.legend(frameon=False, loc = 1)
            plt.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=0.18)
            plt.subplots_adjust(right=0.98)
            plt.subplots_adjust(top=0.98)
            plt.subplots_adjust(bottom=0.22)
            l = self.outdir + '/' + self.fstring + '_rates.pdf'
            if residual_fits is not None:
                l = self.outdir + '/' + self.fstring + '_residuals.pdf'
            if save_all:
                l = (self.outdir + '/' + self.fstring + '_rates' + '_' +
                        str(self.counter) + '.pdf' )
            fig2.savefig(l)


    def inject_signal(self, scale_override = None, residuals = False):
        self.model      = 'one FRED pulse lens'
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = True,
                            constraint = None)

        bin_size = 0.064
        sample = self.priors.sample()
        sample['background'] = 100 * bin_size
        sample['start_1']    = 2
        sample['scale_1']    = 1e4 * bin_size
        if scale_override:
            sample['scale_1']= scale_override * bin_size
        sample['tau_1']      = 6 #0.08
        sample['xi_1']       = 3 ## (do I need to / 0.064 ???)
        sample['time_delay'] = 20#0.4
        sample['magnification_ratio'] = 0.4

        t_0     = -2
        times   = np.arange(800) * bin_size - t_0 ## = 51.2
        dt      = np.diff(times)
        test_counts  = self.one_FRED_lens_rate(dt, t_0, **sample)
        noise_counts = np.random.poisson(test_counts)

        if residuals == 'SineGaussian':
            res_inj_params = bilbyPriorDict()
            res_inj_params['sg_A']    = 2
            res_inj_params['sg_t_0']  = 2
            res_inj_params['sg_tau']  = 2
            res_inj_params['sg_omega']= 2
            res_inj_params['sg_phi']  = 2
            noise_counts += self.sine_gaussian(times, **res_inj_params)

        elif residuals == 'Bessel':
            res_inj_params = bilbyPriorDict()
            res_inj_params['bes_A']    = 1e2
            res_inj_params['bes_Omega']= 0.5
            res_inj_params['bes_s']    = 2
            res_inj_params['bes_t_0']  = 12
            res_inj_params['bes_Delta']= 2
            noise_counts[100:300] += self.residuals_bessel(times[100:300],
                                            **res_inj_params).astype('int')
        else:
            pass

        final_counts = np.zeros((len(times),1))
        final_counts[:,0] = noise_counts
        self.GRB.bin_left, self.GRB.counts = times, final_counts
        self.GRB.bin_right = self.GRB.bin_left + bin_size
        widths = self.GRB.bin_right - self.GRB.bin_left
        self.GRB.rates = self.GRB.counts / widths

        fig, ax = plt.subplots()
        ax.plot(times, test_counts / bin_size, c='r', alpha = 1)
        ax.plot(times, self.GRB.rates, c='k', alpha = 0.2, drawstyle = 'steps',
                linewidth = 0.6)
        ax.set_xlabel('Time since trigger (s)')
        ax.set_ylabel('counts / sec')
        plot_name = self.outdir + '/injected_signal.pdf'
        if scale_override:
            plot_name = (self.outdir + '/injected_signal_' +
                            str(self.counter + 1) + '.pdf' )
            print(plot_name)
        fig.savefig(plot_name)


    def main(self,  rate_function, plot = True,
                    channels = np.arange(4),
                    test = False,
                    save_all = False):
        widths     = self.GRB.bin_right - self.GRB.bin_left
        deltat     = np.diff(self.GRB.bin_left)
        evidences  = []
        errors     = []

        for i in channels:
            self.priors['t_0'] = bilbyDeltaFunction(
                peak = float(self.GRB.bin_left[0]), name = None,
                latex_label = None, unit = None )

            counts       = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood   = bilbyPoissonLikelihood(deltat, counts, rate_function)
            result_label = self.fstring + '_result_' + self.clabels[i]

            if save_all:
                self.counter += 1
                result_label += '_' + str(self.counter)
            open_result  = self.outdir + '/' + result_label +'_result.json'

            if not test:
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
                    if plot:
                        plotname = self.outdir + '/' + result_label +'_corner.pdf'
                        result.plot_corner(filename = plotname)
            else:
                result = bilby.run_sampler( likelihood = likelihood,
                                            priors     = self.priors,
                                            sampler    = self.sampler,
                                            nlive      = self.nSamples,
                                            outdir     = self.outdir,
                                            label      = result_label,
                                            save       = True)
                if plot:
                    plotname = self.outdir + '/' + result_label +'_corner.pdf'
                    result.plot_corner(filename = plotname)
            try:
                del self.priors['t_0']
            except:
                pass


            evidences.append(result.log_evidence)
            errors.append(result.log_evidence_err)

        if plot:
            self.plot_rates(priors = self.priors.copy(),
                            rate_function = rate_function,
                            channels = channels, save_all = save_all)
        return evidences, errors

    def one_FRED(self, **kwargs):
        self.model  = 'one FRED'
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = None)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.one_FRED_rate, **kwargs)
        return evidences, errors

    def one_FREDx(self, **kwargs):
        self.model  = 'one FREDx'
        self.num_pulses = 1
        self.make_priors(   FRED = None, FREDx = [1],
                            gaussian = None, lens = False,
                            constraint = None)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.one_FREDx_rate, **kwargs)
        return evidences, errors

    def two_FRED(self, **kwargs):
        self.model  = 'two FRED pulse'
        self.num_pulses = 2
        self.make_priors(   FRED = [1, 2], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = two_pulse_constraints)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.two_FRED_rate, **kwargs)
        return evidences, errors

    def one_FRED_lens(self, **kwargs):
        self.model  = 'one FRED lens'
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = True,
                            constraint = None)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.one_FRED_lens_rate, **kwargs)
        return evidences, errors

    def three_FRED(self, **kwargs):
        self.model  = 'three FRED pulse'
        self.num_pulses = 3
        self.make_priors(   FRED = [1, 2, 3], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = three_pulse_constraints)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.three_FRED_rate, **kwargs)
        return evidences, errors

    def four_FRED(self, **kwargs):
        self.model  = 'four FRED pulse'
        self.num_pulses = 4
        self.make_priors(   FRED = [1, 2, 3, 4], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = four_pulse_constraints)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.four_FRED_rate, **kwargs)
        return evidences, errors

    def two_FRED_lens(self, **kwargs):
        self.model  = 'two FRED lens'
        self.num_pulses = 2
        self.make_priors(   FRED = [1, 2], FREDx = None,
                            gaussian = None, lens = True,
                            constraint = two_pulse_constraints)
        for key in self.priors:
            print(key)
        evidences, errors = self.main(self.two_FRED_lens_rate, **kwargs)
        return evidences, errors

## end class


def two_pulse_constraints(parameters):
    parameters['constraint_2'] = parameters['start_2'] - parameters['start_1']
    return parameters

def three_pulse_constraints(parameters):
    parameters['constraint_2'] = parameters['start_2'] - parameters['start_1']
    parameters['constraint_3'] = parameters['start_3'] - parameters['start_2']
    return parameters

def four_pulse_constraints(parameters):
    parameters['constraint_2'] = parameters['start_2'] - parameters['start_1']
    parameters['constraint_3'] = parameters['start_3'] - parameters['start_2']
    parameters['constraint_4'] = parameters['start_4'] - parameters['start_3']
    return parameters


if __name__ == '__main__':
    print('Call functions from Call Centre or similar ')
