from pathlib import Path
import os, sys

os.chdir(os.path.dirname(sys.argv[0]))
## makes the scripts location the current working directory rather than the
## directory the script was launched from

import numpy as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from matplotlib.lines import Line2D

from scipy.special import gammaln

import bilby
from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction
from bilby.core.likelihood  import Analytical1DLikelihood
from bilby.core.likelihood  import PoissonLikelihood as bilbyPoissonLikelihood

try:
    from PyGRB_Bayes.core import BATSEpreprocess
    from PyGRB_Bayes.core import DynamicBackEnd ## how to import * from this?
except:
    import BATSEpreprocess
    from DynamicBackEnd import *


class BilbyObject(object):
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
                        nSamples            = 200):

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


        self.priors_pulse_start = priors_pulse_start
        self.priors_pulse_end   = priors_pulse_end
        self.priors_td_lo       = priors_td_lo
        self.priors_td_hi       = priors_td_hi

        self.MC_counter          = None
        self.test                = test

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

        ## move to make priors


    def get_trigger_label(self):
        tlabel = str(self.trigger)
        if len(tlabel) < 4:
            tlabel = ''.join('0' for i in range(4-len(tlabel))) + tlabel
        return tlabel

    def get_max_pulse(self):
        mylist = self.model['count_FRED'] + self.model['count_FREDx']
        ## set gets the unique values of the list
        myset  = set(mylist)
        try:
            self.num_pulses = max(myset) ## WILL NEED EXPANDING
        except:
            self.num_pulses = 0

    def get_base_directory(self):
        directory  = '../products/'
        directory += self.tlabel + '_model_comparison_' + str(self.nSamples)
        self.base_folder = directory

    def get_pulse_list(self):
        string = ''
        for i in range(1, self.num_pulses + 1):
            if i in self.model['count_FRED']:
                string += 'F'
            elif i in self.model['count_FREDx']:
                string += 'X'
            if i in self.model['count_sg']:
                string += 's'
            elif i in self.model['count_bes']:
                string += 'b'
        return string

    def get_directory_name(self):
        ''' Code changes the root directory to the directory above this file.
            Then product files (light-curves, posterior chains) are created in:
                " directory  = '../products/' "

            self.tlabel : 4 character burst trigger number

            adds '_model_comparison_' (could be removed really)

            add number of live points (~ accuracy proxy)

            add lens model or null model (if self.lens)

            add number of pulses

            add pulse keys (eg FFbXsF : Fred F <- bessel_res FREDx <- sg_res F)
            residual is attached to the proceeding pulse.

            MC counter is for testing the code over many trials --> save data
        '''
        self.get_base_directory()
        directory = self.base_folder
        if self.model['lens']:
            directory += '/lens_model'
        else:
            directory += '/null_model'
        directory += '_' + str(self.num_pulses)

        directory += '_' + self.get_pulse_list()
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
        if self.model['lens']:
            file_string += '_YL'
        else:
            file_string +='_NL'
        file_string += str(self.nSamples) + '_'
        return file_string


    def array_job(self, indices):
        FRED_lists  = [[k for k in range(1, i)] for i in range(2, 6)]
        FRED_lists += [[k for k in range(1, i)] for i in range(2, 4)]
        lens_lists  = ['False' for i in range(4)]
        lens_lists += ['True'  for i in range(2)]

        dictionary = dict()
        for idx in indices:
            n_channels = 4
            p_index    = idx // n_channels
            channel    = idx %  n_channels

            dictionary['channel']    = channel
            dictionary['count_FRED'] = FRED_lists[p_index]
            dictionary['count_sg']   = []
            dictionary['lens']       = lens_lists[p_index]

            self.main_1_channel(**dictionary)

    def main_4_channel(self, model):
        self.model   = model
        self.get_max_pulse()
        self.tlabel  = self.get_trigger_label()
        self.fstring = self.get_file_string()
        self.outdir  = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

        channels = [0, 1, 2, 3]

        if not self.test:
            for i in channels:
                plt.plot(self.GRB.bin_left, self.GRB.rates[:,i],
                            c = self.colours[i], drawstyle='steps-mid')
            plot_name = self.base_folder + '/injected_signal'
            plt.savefig(plot_name)

        figure, axes = plt.subplots()
        for i in channels:
            line, fit = self.main_1_channel(i, model)
            axes.add_line(line)
            axes.add_line(fit)
        axes.autoscale()
        figname = self.outdir + '/' + self.fstring +'_rates.pdf'
        figure.savefig(figname)

    def main_1_channel(self, channel, model):
        self.model   = model
        self.get_max_pulse()
        self.tlabel  = self.get_trigger_label()
        self.fstring = self.get_file_string()
        self.outdir  = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

        i           = channel
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

        result_label = self.fstring + '_result_' + self.clabels[i]
        result = bilby.run_sampler( likelihood = likelihood,
                                    priors     = priors,
                                    sampler    = self.sampler,
                                    nlive      = self.nSamples,
                                    outdir     = self.outdir,
                                    label      = result_label,
                                    save       = True)
        plotname = self.outdir + '/' + result_label +'_corner.pdf'
        result.plot_corner(filename = plotname)

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

        fig, ax     = plt.subplots()
        ax.plot(x, y, c = self.colours[i])
        line = Line2D(x, y, c = self.colours[i])
        if self.model['lens']:
            ax.plot(x,  likelihood.calculate_rate(x, MAP,
                        likelihood.insert_name_lens), 'k:')
            fit = Line2D(x,  likelihood.calculate_rate(x, MAP,
                        likelihood.insert_name_lens), ls = ':', c = 'k')

        else:
            ax.plot(x,  likelihood.calculate_rate(x, MAP,
                        likelihood.insert_name), 'k:')
            fit = Line2D(x,  likelihood.calculate_rate(x, MAP,
                        likelihood.insert_name), ls = ':', c = 'k')

        figname = self.outdir + '/' + result_label +'_rates.pdf'
        fig.savefig(figname)
        return line, fit

    def main_4_channell(self, model):
        self.model   = model
        self.get_max_pulse()
        self.tlabel  = self.get_trigger_label()
        self.fstring = self.get_file_string()
        self.outdir  = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

        if not self.test:
            for i in range(4):
                plt.plot(self.GRB.bin_left, self.GRB.rates[:,i],
                            c = self.colours[i], drawstyle='steps-mid')
            plot_name = self.base_folder + '/injected_signal'
            plt.savefig(plot_name)

        fig, ax = plt.subplots()
        channels = [0, 1, 2, 3]
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

            result_label = self.fstring + '_result_' + self.clabels[i]
            result = bilby.run_sampler( likelihood = likelihood,
                                        priors     = priors,
                                        sampler    = self.sampler,
                                        nlive      = self.nSamples,
                                        outdir     = self.outdir,
                                        label      = result_label,
                                        save       = True)
            plotname = self.outdir + '/' + result_label +'_corner.pdf'
            result.plot_corner(filename = plotname)

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

            ax.plot(x, y, c = self.colours[i])
            if lens:
                ax.plot(x,  likelihood.calculate_rate(x, MAP,
                            likelihood.insert_name_lens), 'k:')
            else:
                ax.plot(x,  likelihood.calculate_rate(x, MAP,
                            likelihood.insert_name), 'k:')

        figname = self.outdir + '/' + self.fstring +'_rates.pdf'
        fig.savefig(figname)

    def get_residuals(self, **kwargs):
        count_FRED  = kwargs['count_FRED']
        count_sg    = kwargs['count_sg']
        lens        = kwargs['lens']

        self.num_pulses = count_FRED[-1]
        if lens:
            self.model  = 'lens'
        else:
            self.model  = 'pulse'
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

        channels        = [0, 1, 2, 3]
        count_fits      = np.zeros((len(self.GRB.bin_left),4))
        residuals       = np.zeros((len(self.GRB.bin_left),4))
        for i in channels:
            prior_shell = MakePriors(
                                FRED_pulses = count_FRED,
                                residuals_sg = count_sg,
                                lens = lens,
                                priors_pulse_start = self.priors_pulse_start,
                                priors_pulse_end = self.priors_pulse_end,
                                priors_td_lo = self.priors_td_lo,
                                priors_td_hi = self.priors_td_hi)
            priors = prior_shell.return_prior_dict()

            x = self.GRB.bin_left
            y = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood = PoissonRate(x, y, count_FRED, count_sg, lens = lens)


            result_label = self.fstring + '_result_' + self.clabels[i]
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

            if lens:
                counts_fit = likelihood.calculate_rate(x, MAP,
                                                likelihood.insert_name_lens)
            else:
                counts_fit = likelihood.calculate_rate(x, MAP,
                                                likelihood.insert_name)

            count_fits[:,i] = counts_fit
            residuals[:,i] = self.GRB.counts[:,i] - counts_fit

        widths = self.GRB.bin_right - self.GRB.bin_left
        rates  = self.GRB.counts        / widths[:,None]
        rates_fit       = count_fits    / widths[:,None]
        residual_rates  = residuals     / widths[:,None]

        self.plot_4_channel(    x = self.GRB.bin_left, y = rates,
                                y_fit = rates_fit,
                                channels = channels, y_res_fit = None)

    def plot_4_channel(self, x, y, y_fit, channels, y_res_fit = None, residuals = False, offsets = None):

        n_axes  = len(channels) + 1
        # n_axes  = min(np.shape(y)) + 1
        width   = 3.321
        height  = (width / 1.8) * 2
        heights = [5] + ([1 for i in range(n_axes - 1)])
        fig     = plt.figure(figsize = (width, height), constrained_layout=False)
        spec    = gridspec.GridSpec(ncols=2, nrows=n_axes, figure=fig,
                                height_ratios=heights,
                                width_ratios=[0.05, 0.95],
                                hspace=0.0, wspace=0.0)
        ax      = fig.add_subplot(spec[:, 0], frameon = False)
        fig_ax1 = fig.add_subplot(spec[0, 1])
        axes_list = []
        for i in channels:
            if offsets:
                line_label = f'offset {offsets[i]:+,}'
                fig_ax1.plot(   x, y[:,i] + offsets[i], c = self.colours[i],
                                drawstyle='steps-mid', linewidth = 0.4,
                                label = line_label)
                fig_ax1.plot(x, y_fit[:,i] + offsets[i], 'k', linewidth = 0.4)
            else:
                fig_ax1.plot(   x, y[:,i], c = self.colours[i],
                                drawstyle='steps-mid', linewidth = 0.4)
                fig_ax1.plot(x, y_fit[:,i], 'k', linewidth = 0.4)
                #, label = plot_legend)

            axes_list.append(fig.add_subplot(spec[i+1, 1]))
            difference = y[:,i] - y_fit[:,i]
            axes_list[i].plot(  x, difference, c = self.colours[i],
                                drawstyle='steps-mid',  linewidth = 0.4)
            if y_res_fit is not None:
                axes_list[i].plot(  x, y_res_fit[:,i], 'k:', linewidth = 0.4)
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
        plt.subplots_adjust(bottom=0.13)

        fig_ax1.ticklabel_format(axis = 'y', style = 'sci')
        if offsets:
            fig_ax1.legend()

        plot_name = self.outdir + '/' + self.fstring + '_rates.pdf'
        if residuals is True:
            plot_name = self.outdir + '/' + self.fstring + '_residuals.pdf'
        fig.savefig(plot_name)






def load_3770(sampler = 'dynesty', nSamples = 100):
    bilby_inst = BilbyObject(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return bilby_inst



def load_999(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(999, times = (3, 8),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return object


def load_2571(sampler = 'dynesty', nSamples = 250):
    test = BilbyObject(2571, times = (-2, 40),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 30,
                priors_td_lo = 0,  priors_td_hi = 15)
    return test

def load_973(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(973, times = (-2, 50),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    return test



def create_model_dict(lens, count_FRED, count_FREDx, count_sg, count_bes):
    model = {}
    model['lens']        = lens
    model['count_FRED']  = count_FRED
    model['count_FREDx'] = count_FREDx
    model['count_sg']    = count_sg
    model['count_bes']   = count_bes
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
    parser.add_argument('--HPC', action = 'store_true',
                        help = 'Are you running this on SPARTAN ?')
    parser.add_argument('-i', '--indices', type=int, nargs='+',
                        help='an integer for indexing geomspace array')
    args = parser.parse_args()
    HPC = args.HPC

    print(args)

    if not HPC:
        from matplotlib import rc
        rc('font', **{'family': 'DejaVu Sans',
                    'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        rc('text.latex',
        preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}')
        SAMPLER = 'Nestle'
    else:
        SAMPLER = 'dynesty'

    GRB = load_999(sampler = SAMPLER, nSamples = 100)
    # GRB = load_3770(sampler = SAMPLER, nSamples = 1000)
    model = create_model_dict(lens = False, count_FRED  = [1],
                                            count_FREDx = [],
                                            count_sg    = [],
                                            count_bes   = [])
    GRB.main_4_channel(model)
    # GRB.main_1_channel(2, model)
    # GRB.array_job(args.indices)
