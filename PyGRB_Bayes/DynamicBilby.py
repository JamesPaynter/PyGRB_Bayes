import os
import sys
import numpy  as np

# os.chdir(os.path.dirname(sys.argv[0]))
## makes the scripts location the current working directory rather than the
## directory the script was launched from
import bilby

from PyGRB_Bayes.preprocess import BATSEpreprocess
from PyGRB_Bayes.preprocess import GRB_class
from PyGRB_Bayes.backend.makepriors import MakePriors
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_singular_models
from PyGRB_Bayes.backend.makemodels import make_two_pulse_models
from PyGRB_Bayes.backend.rateclass import PoissonRate
from PyGRB_Bayes.backend.admin import Admin, mkdir
from PyGRB_Bayes.postprocess.make_evidence_tables import EvidenceTables
from PyGRB_Bayes.postprocess.plot_grb import GRBPlotter
from PyGRB_Bayes.postprocess.plot_analysis import PlotPulseFit


class BilbyObject(Admin, EvidenceTables):
    ''' Wrapper object for Bayesian analysis. '''

    def __init__(self,  trigger, times, datatype,
                        priors_pulse_start, priors_pulse_end,
                        priors_td_lo = None, priors_td_hi = None,
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

        self.variable = kwargs.get('variable')
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

        if datatype == 'tte_list':
            self.GRB = GRB_class.make_GRB(
                3770, 'TTE_list', live_detectors = np.arange(5,8))
        else:
            self.GRB = BATSEpreprocess.make_GRB(
                burst = self.trigger, times = (self.start, self.end),
                datatype = self.datatype, bgs = False)

        # if test:
            # self.GRB = EmptyGRB()
            # self.GRB.trigger = self.trigger
            # self.GRB.start   = self.start
            # self.GRB.end     = self.end
            # self.GRB.datatype= self.datatype
        print(self.GRB)




    def _split_array_job_to_4_channels(self, models, indices, channels = None):
        for idx in indices:
            n_channels = 4
            m_index    = idx // n_channels
            channel    = idx %  n_channels
            self.main_1_channel(channel, models[m_index])
        #
        # for idx in indices:
        #     n_channels = len(channels)
        #     m_index    = idx // n_channels
        #     channel    = channels[idx %  n_channels]
        #     self.main_1_channel(channel, models[m_index])




    def test_pulse_type(self, indices):
        self.models = make_singular_models()
        models = [model for key, model in self.models.items()]
        self._split_array_job_to_4_channels(models, indices)

    def test_two_pulse_models(self, indices):
        self.models = make_two_pulse_models()
        models = [model for key, model in self.models.items()]
        self._split_array_job_to_4_channels(models, indices)

    def main_multi_channel(self, channels, model):
        self._setup_labels(model)
        if not self.test:
            GRBPlotter( GRB = self.GRB, channels = channels,
                        outdir = self.base_folder)
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
        strings = { 'fstring' : self.fstring,
                    'clabels' : self.clabels,
                    'outdir'  : self.outdir}

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
            count_fits[:,i]= counts_fit
            residuals[:,i] = self.GRB.counts[:,i] - counts_fit

            # for k in range(len(self.GRB.bin_left)):
            #     print(k, self.GRB.bin_left[k], y[k], counts_fit[k])
            widths = self.GRB.bin_right - self.GRB.bin_left
            rates_i= self.GRB.counts[:,i] / widths
            rates_fit_i = counts_fit / widths
            rates_err_i = np.sqrt(self.GRB.counts[:,i]) / widths
            strings['widths'] = widths
            PlotPulseFit(   x = self.GRB.bin_left, y = rates_i,
                            y_err = rates_err_i,
                            y_cols = self.GRB.colours[i],
                            y_fit = rates_fit_i,
                            channels = [i], **strings)

        widths = self.GRB.bin_right - self.GRB.bin_left
        rates  = self.GRB.counts        / widths[:,None]
        rates_fit       = count_fits    / widths[:,None]
        rates_err       = np.sqrt(self.GRB.counts) / widths[:,None]
        residual_rates  = residuals     / widths[:,None]
        PlotPulseFit(   x = self.GRB.bin_left, y = rates, y_err = rates_err,
                        y_cols = self.GRB.colours, y_offsets = self.offsets,
                        y_fit = rates_fit,
                        channels = channels, **strings)

if __name__ == '__main__':
    pass
