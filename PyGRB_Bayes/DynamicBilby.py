import os
import sys
import numpy as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import scipy.special as special
from scipy.special import gammaln

# os.chdir(os.path.dirname(sys.argv[0]))
## makes the scripts location the current working directory rather than the
## directory the script was launched from
from prettytable import PrettyTable

from PyGRB_Bayes import BATSEpreprocess
from PyGRB_Bayes import GRB_class
from PyGRB_Bayes.DynamicBackEnd import * ## how to import * from this?



class BilbyObject(object):
    ''' Wrapper object for Bayesian analysis. '''

    def __init__(self,  trigger, times, datatype,
                        priors_pulse_start, priors_pulse_end,
                        priors_td_lo = None, priors_td_hi = None,
                        tte_list            = False,
                        satellite           = 'BATSE',
                        ## are your bins the right size in rate function ????
                        sampler             = 'dynesty',
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
            self.GRB = BATSEpreprocess.BATSESignal(
                self.trigger, times = (self.start, self.end),
                datatype = self.datatype, bgs = False)
        else:
            self.GRB = GRB_class.BATSEGRB(3770, 'TTE_list', live_detectors = np.arange(5,8))
            # self.GRB = EmptyGRB()
            # self.GRB.trigger = self.trigger
            # self.GRB.start   = self.start
            # self.GRB.end     = self.end
            # self.GRB.datatype= self.datatype

    def make_singular_models(self):
        ''' Create the full array of 1-pulse models. '''
        # TODO create gaussian
        self.models['F']  = create_model_dict(  lens = False, count_FRED  = [1],
                                                count_FREDx = [],
                                                count_sg    = [],
                                                count_bes   = [],
                                                name = 'FRED')
        self.models['Fs'] = create_model_dict(  lens = False, count_FRED  = [1],
                                                count_FREDx = [],
                                                count_sg    = [1],
                                                count_bes   = [],
                                                name = 'FRED sg residual')
        self.models['Fb'] = create_model_dict(  lens = False, count_FRED  = [1],
                                                count_FREDx = [],
                                                count_sg    = [],
                                                count_bes   = [1],
                                                name = 'FRED bes residual')
        self.models['X']  = create_model_dict(  lens = False, count_FRED  = [],
                                                count_FREDx = [1],
                                                count_sg    = [],
                                                count_bes   = [],
                                                name = 'FREDx')
        self.models['Xs'] = create_model_dict(  lens = False, count_FRED  = [],
                                                count_FREDx = [1],
                                                count_sg    = [1],
                                                count_bes   = [],
                                                name = 'FREDx sg residual')
        self.models['Xb'] = create_model_dict(  lens = False, count_FRED  = [],
                                                count_FREDx = [1],
                                                count_sg    = [],
                                                count_bes   = [1],
                                                name = 'FREDx bes residual')

    @staticmethod
    def get_pos_from_key(key, char):
        """ Returns a list of the indices where char appears in the string.
            Pass in a list of only the pulses no residuals (ie capital letters)
            +1 is because the pulses are index from 1.
            """
        return [i+1 for i, c in enumerate(key) if c == char]

    @staticmethod
    def get_pos_from_idx(key, idx_array, char):
        return [idx_array[i] for i, c in enumerate(key) if c == char]

    def create_model_from_key(self, key):
        assert(isinstance(key, str))
        kwargs = {}
        kwargs['lens'] = True if 'L' in key else False
        key = key.strip('L')
        # Gaussian, FRED, FREDx, Convolution
        # TODO allow SG or BES to be standalone pulses
        # as needed, may add bugs down the lineif implemented naively
        pulse_types = ['G', 'F', 'X', 'C']#, 'S', 'B']
        pulse_kwargs= ['count_Gauss', 'count_FRED', 'count_FREDx', 'count_conv']
                        # 'count_sg', 'count_bes']
        res_types   = ['s', 'b']
        res_kwargs  = ['count_sg', 'count_bes']
        # list of capital letters only (ie pulses)
        pulse_keys  = ''.join([c for c in key if c.isupper()])
        pulse_list  = []
        res_list    = []
        for i, char in enumerate(pulse_types):
            # list of indices where current char ('G', 'F' etc.) appears
            idx_list = self.get_pos_from_key(pulse_keys, char)
            # appends this to list of pulses
            pulse_list += idx_list
            # also adds this list to the kwargs dict to be passed to the model
            kwargs[pulse_kwargs[i]] = idx_list
        # sort the list of pulses
        pulse_list.sort()
        # indices of where pulses appear in original string
        pulse_indices = [i for i, c in enumerate(key) if c.isupper()]
        idx_array = [0 for i in range(len(key))]
        # idx_array[i] = pulse_indices[]
        for i in range(len(key)):
            idx_array[i] = 0

        for i, (j, k) in enumerate(zip(pulse_list, pulse_indices)):
            idx_array[k] = j
        for i in range(1, len(key)):
            if idx_array[i] == 0:
                idx_array[i] = idx_array[i-1]

        for i, char in enumerate(res_types):
            kwargs[res_kwargs[i]] = self.get_pos_from_idx(key, idx_array, char)

        model = create_model_dict(**kwargs)
        return model

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
        """ Code changes the root directory to the directory above this file.
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
        """
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

    def setup_labels(self, model):
        self.model = model
        self.get_max_pulse()
        self.tlabel = self.get_trigger_label()
        self.fstring = self.get_file_string()
        self.outdir = self.get_directory_name()
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

    def array_job(self, indices):
        ''' takes one of 24 indices from HPC to parallelise this job. '''
        FRED_lists  = [[k for k in range(1, i)] for i in range(2, 6)]
        FRED_lists += [[k for k in range(1, i)] for i in range(2, 4)]
        lens_lists  = ['False' for i in range(4)]
        lens_lists += ['True'  for i in range(2)]

        model = dict()
        for idx in indices:
            n_channels = 4
            p_index    = idx // n_channels
            channel    = idx %  n_channels

            model['count_FRED'] = FRED_lists[p_index]
            model['count_FREDx']= []
            model['count_sg']   = []
            model['count_bes']  = []
            model['lens']       = lens_lists[p_index]
            self.main_1_channel(channel, model)

    def _split_array_job_to_4_channels(self, models, indices):
        for idx in indices:
            n_channels = 4
            m_index    = idx // n_channels
            channel    = idx %  n_channels
            self.main_1_channel(channel, models[m_index])

    def test_pulse_type(self, indices):
        # clear model dict if not clear already
        self.models = {}
        self.make_singular_models()
        models = [model for key, model in self.models.items()]
        self._split_array_job_to_4_channels(models, indices)

    def test_two_pulse_models(self, indices):
        keys = ['FF', 'FL', 'FsFs', 'FsL', 'XX', 'XL', 'XsXs', 'XsL']
        self.models = {}
        for key in keys:
            self.models[key] = self.create_model_from_key(key)
        models = [model for key, model in self.models.items()]
        self._split_array_job_to_4_channels(models, indices)

    def get_evidence_from_pulse(self, models, keys = None):
        self.tlabel = self.get_trigger_label()
        self.get_base_directory()
        directory = self.base_folder
        Z_file = directory + '/evidence_table.txt'
        open(Z_file, 'w').close()
        for i in range(4):
            x = PrettyTable(['Model', 'log Z', 'error'])
            x.align['Model'] = "l" # Left align models
            # One space between column edges and contents (default)
            x.padding_width = 1
            for k in range(len(models)):
                if 'name' not in [*models[k]]:
                    models[k]['name'] = keys[k]
                self.setup_labels(models[k])
                result_label = f'{self.fstring}_result_{self.clabels[i]}'
                open_result  = f'{self.outdir}/{result_label}_result.json'
                try:
                    result = bilby.result.read_in_result(filename=open_result)
                    x.add_row([ models[k]['name'],
                                f'{result.log_evidence:.2f}',
                                f'{result.log_evidence_err:.2f}'])
                except:
                    print(f'Could not find {open_result}')
            # indentation should be same as k loop
            with open(Z_file, 'a') as w:
                w.write(f'Channel {i+1}')
                w.write(str(x))
                w.write('')

    def get_evidence_singular(self):
        # clear model dict if not clear already
        self.models = {}
        self.make_singular_models()
        models = [model for key, model in self.models.items()]
        self.get_evidence_from_pulse(models)

    def get_evidence_singular_lens(self):
        keys = ['FF', 'FL', 'FsFs', 'FsL', 'XX', 'XL', 'XsXs', 'XsL']
        # keys = ['FF', 'FL', 'FbFb', 'FbL', 'XX', 'XL', 'XbXb', 'XbL']
        self.models = {}
        for key in keys:
            self.models[key] = self.create_model_from_key(key)
        models = [model for key, model in self.models.items()]
        self.get_evidence_from_pulse(models, keys)


    def main_multi_channel(self, channels, model):
        self.setup_labels(model)

        if not self.test:
            for i in channels:
                plt.plot(self.GRB.bin_left, self.GRB.rates[:,i],
                            c = self.colours[i], drawstyle='steps-mid')
            plot_name = self.base_folder + '/injected_signal'
            plt.savefig(plot_name)

        for i in channels:
            self.main_1_channel(i, model)
        self.get_residuals(channels = channels, model = model)

    def main_1_channel(self, channel, model):
        self.setup_labels(model)

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

        self.get_residuals(channels = [channel], model = model)

    def get_residuals(self, channels, model):
        self.setup_labels(model)

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

            if model['lens']:
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
            result_label = self.fstring + '_result_' + self.clabels[channels[0]]
            plot_name    = self.outdir + '/' + result_label +'_rates.pdf'
        else:
            plot_name = self.outdir + '/' + self.fstring + '_rates.pdf'
        # if residuals is True:
        #     plot_name = self.outdir + '/' + self.fstring + '_residuals.pdf'
        fig.savefig(plot_name)


def create_model_dict(  lens, count_FRED, count_FREDx, count_sg, count_bes,
                        **kwargs):
    model = {}
    model['lens']        = lens
    model['count_FRED']  = count_FRED
    model['count_FREDx'] = count_FREDx
    model['count_sg']    = count_sg
    model['count_bes']   = count_bes
    if kwargs:
        for kwarg in kwargs:
            model[kwarg] = kwargs[kwarg]
    return model

if __name__ == '__main__':
    pass
