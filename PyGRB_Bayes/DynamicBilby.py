import os
import sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import scipy.special as special
from scipy.special import gammaln

# os.chdir(os.path.dirname(sys.argv[0]))
## makes the scripts location the current working directory rather than the
## directory the script was launched from
import bilby

from PyGRB_Bayes.preprocess import BATSEpreprocess
from PyGRB_Bayes.preprocess import GRB_class
from PyGRB_Bayes.backend.makepriors import MakePriors
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_singular_models
from PyGRB_Bayes.backend.rateclass import PoissonRate
from PyGRB_Bayes.backend.admin import Admin, mkdir
from PyGRB_Bayes.postprocess.make_evidence_tables import EvidenceTables


class BilbyObject(Admin, EvidenceTables):
    ''' Wrapper object for Bayesian analysis. '''

    def __init__(self,  trigger, times, datatype,
                        priors_pulse_start, priors_pulse_end,
                        priors_td_lo = None, priors_td_hi = None,
                        tte_list            = False,
                        satellite           = 'BATSE',
                        ## are your bins the right size in rate function ????
                        sampler             = 'dynesty',
                        nSamples            = 200,
                        **kwargs):

        super(BilbyObject, self).__init__()

        print('\n\n')
        print('What are you working on!!!???')
        print('\n')
        print('The priorities for this project are:')
        print('1) Unit tests for all the current .py files.')
        print('2) Getting the automated pulse fitting algorithm working.')
        print('3) Import BATSE fetch module from masters.')
        print('4) Complete SWIFT; Fermi, INTEGRAL, KONUS etc. fetch modules.')
        print('5) Automated .pdf reports for model fitting / selection.')
        print('6) Integration tests.')
        print('7) Generalise lensing fits (convolutions).')
        print('8) Documentation; sphinx.')
        print('9) Release in JOSS (ask ADACS for help?).')
        print('\n\n\n')
        print('DO THE PRIORS MAKE SENSE !! ??')
        print('Prior scaling is in counts / bin !!! ')
        print('THIS IS NOT COUNTS / SECOND !!!')
        print('This should only affect the A and B scale and background params')
        print('\n\n\n')

        self.kwargs = kwargs


        (self.start, self.end)   = times
        self.colours             = ['red', 'orange', 'green', 'blue']
        self.clabels             = ['1', '2', '3', '4']
        self.datatype            = datatype
        self.satellite           = satellite
        self.sampler             = sampler
        self.nSamples            = nSamples
        self.trigger             = trigger


        self.priors_pulse_start = priors_pulse_start
        self.priors_pulse_end   = priors_pulse_end
        self.priors_td_lo       = priors_td_lo
        self.priors_td_hi       = priors_td_hi

        self.MC_counter          = None
        self.test                = None # test

        # intialise model dict
        self.models = {}
        self.offsets = None

        if not tte_list:
            # scaf = BATSEpreprocess.BATSESignal(
            #     self.trigger, times = (self.start, self.end),
            #     datatype = self.datatype, bgs = False)

            self.GRB = BATSEpreprocess.make_GRB(
                burst = self.trigger, times = (self.start, self.end),
                datatype = self.datatype, bgs = False)

        else:
            self.GRB = GRB_class.BATSEGRB(3770, 'TTE_list', live_detectors = np.arange(5,8))
            # self.GRB = EmptyGRB()
            # self.GRB.trigger = self.trigger
            # self.GRB.start   = self.start
            # self.GRB.end     = self.end
            # self.GRB.datatype= self.datatype
        print(self.GRB)



    def _split_array_job_to_4_channels(self, models, indices):
        for idx in indices:
            n_channels = 4
            m_index    = idx // n_channels
            channel    = idx %  n_channels
            self.main_1_channel(channel, models[m_index])

    def test_pulse_type(self, indices):
        self.models = make_singular_models()
        models = [model for key, model in self.models.items()]
        self._split_array_job_to_4_channels(models, indices)

    def test_two_pulse_models(self, indices):
        keys = ['FF', 'FL', 'FsFs', 'FsL', 'XX', 'XL', 'XsXs', 'XsL']
        keys+= ['FsF', 'FFs', 'XsX', 'XXs', 'FsX', 'XsF', 'FXs', 'XFs']

        self.models = {}
        for key in keys:
            self.models[key] = create_model_from_key(key)
        models = [model for key, model in self.models.items()]
        self._split_array_job_to_4_channels(models, indices)

    def main_multi_channel(self, channels, model):
        self._setup_labels(model)

        if not self.test:
            fig, ax = plt.subplots()
            for i in channels:
                rates = self.GRB.counts[:,i] / (self.GRB.bin_right - self.GRB.bin_left)
                ax.plot(self.GRB.bin_left, rates,
                            c = self.colours[i], drawstyle='steps-mid')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Counts / second')
            plot_name = f'{self.base_folder}/injected_signal'
            fig.savefig(plot_name)

        for i in channels:
            self.main_1_channel(i, model)
        self.get_residuals(channels = channels, model = model)

    def main_1_channel(self, channel, model):
        self._setup_labels(model)

        i           = channel
        prior_shell = MakePriors(
                            priors_pulse_start = self.priors_pulse_start,
                            priors_pulse_end = self.priors_pulse_end,
                            priors_td_lo = self.priors_td_lo,
                            priors_td_hi = self.priors_td_hi,
                            **self.model,
                            **self.kwargs)
        priors = prior_shell.return_prior_dict()

        x = self.GRB.bin_left
        y = np.rint(self.GRB.counts[:,i]).astype('uint')
        likelihood = PoissonRate(x, y, **self.model)

        result_label = f'{self.fstring}_result_{self.clabels[i]}'
        result = bilby.run_sampler( likelihood = likelihood,
                                    priors     = priors,
                                    sampler    = self.sampler,
                                    nlive      = self.nSamples,
                                    outdir     = self.outdir,
                                    label      = result_label,
                                    save       = True)
        plotname = f'{self.outdir}/{result_label}_corner.pdf'
        result.plot_corner(filename = plotname)

        MAP = dict()
        for j in range(1, self.num_pulses + 1):
            try:
                key = f'constraint_{j}'
                del priors[key]
                key = f'constraint_{j}_res'
                del priors[key]
            except:
                pass
        for parameter in priors:
            summary = result.get_one_dimensional_median_and_error_bar(parameter)
            MAP[parameter] = summary.median

        self.get_residuals(channels = [channel], model = model)

    def get_residuals(self, channels, model):
        self._setup_labels(model)

        count_fits      = np.zeros((len(self.GRB.bin_left),4))
        residuals       = np.zeros((len(self.GRB.bin_left),4))
        for i in channels:
            prior_shell = MakePriors(
                                priors_pulse_start = self.priors_pulse_start,
                                priors_pulse_end = self.priors_pulse_end,
                                priors_td_lo = self.priors_td_lo,
                                priors_td_hi = self.priors_td_hi,
                                **self.model)
            priors = prior_shell.return_prior_dict()

            x = self.GRB.bin_left
            y = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood = PoissonRate(x, y, **self.model)

            result_label = f'{self.fstring}_result_{self.clabels[i]}'
            open_result  = f'{self.outdir}/{result_label}_result.json'
            result = bilby.result.read_in_result(filename=open_result)
            MAP = dict()
            for j in range(1, self.num_pulses + 1):
                try:
                    key = f'constraint_{j}'
                    del priors[key]
                except:
                    pass
            for parameter in priors:
                summary = result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP[parameter] = summary.median

            if model['lens']:
                counts_fit = likelihood._sum_rates(x, MAP,
                                                likelihood.calculate_rate_lens)
            else:
                counts_fit = likelihood._sum_rates(x, MAP,
                                                likelihood.calculate_rate)

            count_fits[:,i] = counts_fit
            residuals[:,i] = self.GRB.counts[:,i] - counts_fit

        widths = self.GRB.bin_right - self.GRB.bin_left
        rates  = self.GRB.counts        / widths[:,None]
        rates_fit       = count_fits    / widths[:,None]
        residual_rates  = residuals     / widths[:,None]

        self.plot_routine(  x = self.GRB.bin_left, y = rates,
                            y_fit = rates_fit,
                            channels = channels, y_res_fit = None)

    def plot_routine(self, x, y, y_fit, channels, y_res_fit = None, residuals = False):
        offsets = self.offsets
        n_axes  = len(channels) + 1
        width   = 3.321
        height  = (width / 1.8) * 2
        heights = [5] + ([1 for i in range(n_axes - 1)])
        fig     = plt.figure(figsize = (width, height), constrained_layout=False)
        spec    = gridspec.GridSpec(ncols=2, nrows=n_axes, figure=fig,
                                    height_ratios=heights,
                                    width_ratios=[0.05, 0.95],
                                    hspace=0.0, wspace=0.0)
        ax      = fig.add_subplot(spec[:, 0], frameon = False)
        fig_ax1 = fig.add_subplot(spec[0, 1]) ## axes label on the LHS of plot
        axes_list = []
        for i, k in enumerate(channels):
            if offsets and len(channels) > 1:
                line_label = f'offset {offsets[k]:+,}'
                fig_ax1.plot(   x, y[:,k] + offsets[k], c = self.colours[k],
                                drawstyle='steps-mid', linewidth = 0.4,
                                label = line_label)
                fig_ax1.plot(x, y_fit[:,k] + offsets[k], 'k', linewidth = 0.4)
            else:
                fig_ax1.plot(   x, y[:,k], c = self.colours[k],
                                drawstyle='steps-mid', linewidth = 0.4)
                fig_ax1.plot(x, y_fit[:,k], 'k', linewidth = 0.4)
                #, label = plot_legend)

            axes_list.append(fig.add_subplot(spec[i+1, 1]))
            difference = y[:,k] - y_fit[:,k]
            axes_list[i].plot(  x, difference, c = self.colours[k],
                                drawstyle='steps-mid',  linewidth = 0.4)
            if y_res_fit is not None:
                axes_list[i].plot(  x, y_res_fit[:,k], 'k:', linewidth = 0.4)
            axes_list[i].set_xlim(x[0], x[-1])
            axes_list[i].set_xticks(())
            tick = int(np.max(difference) * 0.67 / 100) * 100
            axes_list[i].set_yticks(([int(0), tick]))

        axes_list[-1].set_xlabel('time since trigger (s)')
        ax.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        ax.set_ylabel('counts / sec')
        plt.subplots_adjust(left=0.16)
        plt.subplots_adjust(right=0.98)
        plt.subplots_adjust(top=0.98)
        plt.subplots_adjust(bottom=0.05)

        fig_ax1.ticklabel_format(axis = 'y', style = 'sci')
        if offsets:
            fig_ax1.legend()

        fig_ax1.set_xlim(x[0], x[-1])
        if len(channels) == 1:
            result_label = f'{self.fstring}_result_{self.clabels[channels[0]]}'
            plot_name    = f'{self.outdir}/{result_label}_rates.pdf'
        else:
            plot_name = f'{self.outdir}/{self.fstring}_rates.pdf'
        # if residuals is True:
        #     plot_name = self.outdir + '/' + self.fstring + '_residuals.pdf'
        fig.savefig(plot_name)


    def get_residuals_tte(self, channels, model):
        self._setup_labels(model)

        count_fits      = np.zeros((len(self.GRB.bin_left),4))
        residuals       = np.zeros((len(self.GRB.bin_left),4))
        for i in channels:
            prior_shell = MakePriors(
                                priors_pulse_start = self.priors_pulse_start,
                                priors_pulse_end = self.priors_pulse_end,
                                priors_td_lo = self.priors_td_lo,
                                priors_td_hi = self.priors_td_hi,
                                **self.model)
            priors = prior_shell.return_prior_dict()

            x = self.GRB.bin_left
            y = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood = PoissonRate(x, y, **self.model)

            result_label = f'{self.fstring}_result_{self.clabels[i]}'
            open_result  = f'{self.outdir}/{result_label}_result.json'
            result = bilby.result.read_in_result(filename=open_result)
            MAP = dict()
            for j in range(1, self.num_pulses + 1):
                try:
                    key = f'constraint_{j}'
                    del priors[key]
                except:
                    pass
            for parameter in priors:
                summary = result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP[parameter] = summary.median

            if model['lens']:
                counts_fit = likelihood._sum_rates(x, MAP,
                                                likelihood.calculate_rate_lens)
            else:
                counts_fit = likelihood._sum_rates(x, MAP,
                                                likelihood.calculate_rate)

            count_fits[:,i] = counts_fit
            residuals[:,i] = self.GRB.counts[:,i] - counts_fit

        widths = self.GRB.bin_right - self.GRB.bin_left
        rates  = self.GRB.counts        / widths[:,None]
        rates_fit       = count_fits    / widths[:,None]
        residual_rates  = residuals     / widths[:,None]

        self.plot_routine_tte(  x = self.GRB.bin_left, y = rates,
                            y_fit = rates_fit,
                            channels = channels, y_res_fit = None)

    def plot_routine_tte(self, x, y, y_fit, channels, y_res_fit = None, residuals = False):
        offsets = self.offsets
        n_axes  = len(channels) + 1
        width   = 3.321
        height  = (width / 1.8) * 2
        heights = [5] + ([1 for i in range(n_axes - 1)])
        fig     = plt.figure(figsize = (width, height), constrained_layout=False)
        spec    = gridspec.GridSpec(ncols=2, nrows=n_axes, figure=fig,
                                    height_ratios=heights,
                                    width_ratios=[0.05, 0.95],
                                    hspace=0.0, wspace=0.0)
        ax      = fig.add_subplot(spec[:, 0], frameon = False)
        fig_ax1 = fig.add_subplot(spec[0, 1]) ## axes label on the LHS of plot
        axes_list = []
        for i, k in enumerate(channels):
            if offsets and len(channels) > 1:
                line_label = f'offset {offsets[k]:+,}'
                # fig_ax1.plot(   x, y[:,k] + offsets[k], c = self.colours[k],
                #                 drawstyle='steps-mid', linewidth = 0.4,
                #                 label = line_label)
                fig_ax1.plot(x, y_fit[:,k] + offsets[k], 'k', linewidth = 0.4)
            else:
                # fig_ax1.plot(   x, y[:,k], c = self.colours[k],
                                # drawstyle='steps-mid', linewidth = 0.4)
                fig_ax1.plot(x, y_fit[:,k], 'k', linewidth = 0.4)
                #, label = plot_legend)

            axes_list.append(fig.add_subplot(spec[i+1, 1]))
            difference = y[:,k] - y_fit[:,k]
            axes_list[i].plot(  x, difference, c = self.colours[k],
                                drawstyle='steps-mid',  linewidth = 0.4)
            if y_res_fit is not None:
                axes_list[i].plot(  x, y_res_fit[:,k], 'k:', linewidth = 0.4)
            axes_list[i].set_xlim(x[0], x[-1])
            axes_list[i].set_xticks(())
            tick = int(np.max(difference) * 0.67 / 100) * 100
            axes_list[i].set_yticks(([int(0), tick]))

        axes_list[-1].set_xlabel('time since trigger (s)')
        ax.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        ax.set_ylabel('counts / sec')
        plt.subplots_adjust(left=0.16)
        plt.subplots_adjust(right=0.98)
        plt.subplots_adjust(top=0.98)
        plt.subplots_adjust(bottom=0.05)

        fig_ax1.ticklabel_format(axis = 'y', style = 'sci')
        if offsets:
            fig_ax1.legend()

        fig_ax1.set_xlim(x[0], x[-1])
        if len(channels) == 1:
            result_label = f'{self.fstring}_result_{self.clabels[channels[0]]}'
            plot_name    = f'{self.outdir}/{result_label}_rates.pdf'
        else:
            plot_name = f'{self.outdir}/{self.fstring}_rates.pdf'
        # if residuals is True:
        #     plot_name = self.outdir + '/' + self.fstring + '_residuals.pdf'
        fig.savefig(plot_name)


if __name__ == '__main__':
    pass
