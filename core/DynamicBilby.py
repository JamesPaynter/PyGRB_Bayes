import numpy as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from scipy.special import gammaln

import bilby
from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction
from bilby.core.likelihood  import Analytical1DLikelihood
from bilby.core.likelihood  import PoissonLikelihood as bilbyPoissonLikelihood

import BATSEpreprocess



class MakeKeys(object):
    '''
        Doc string goes here.
    '''

    def __init__(self, FRED_pulses, residuals_sg, lens = False):
        super(MakeKeys, self).__init__()
        self.FRED_pulses  = FRED_pulses
        self.residuals_sg = residuals_sg
        mylist = FRED_pulses + residuals_sg
        myset = set(mylist)
        self.max_pulse    = max(myset) ## WILL NEED EXPANDING

        self.FRED_list    = ['start', 'scale', 'tau', 'xi']
        self.FREDx_list   = self.FRED_list.copy() + ['gamma', 'nu']
        self.res_sg_list  = ['sg_A', 'sg_begin', 'sg_tau', 'sg_omega', 'sg_phi']
        self.res_bes_list = ['bes_A', 'bes_Omega', 'bes_s', 'bes_t_0', 'bes_Delta']

        self.lens         = lens
        self.lens_list    = ['time_delay', 'magnification_ratio']

        self.keys = []
        self.fill_keys_list()

    def fill_list(self, list, array):
        return ['{}_{}'.format(list[k], i) for k in range(len(list))
                                           for i in array]

    def fill_keys_list(self):
        if self.lens:
            self.keys += self.lens_list
        self.keys += ['background']
        self.keys += self.fill_list(self.FRED_list, self.FRED_pulses)
        self.keys += self.fill_list(self.res_sg_list, self.residuals_sg)


# @dataclass
# class PriorRanges:
#     priors_pulse_start: float
#     priors_pulse_end:   float
#     priors_td_lo:       float = None
#     priors_td_hi:       float = None
#     priors_bg_lo:       float = 1e-1  ## SCALING IS COUNTS / BIN
#     priors_bg_hi:       float = 1e3   ## SCALING IS COUNTS / BIN
#     priors_mr_lo:       float = 0.2   ## which means that it is
#     priors_mr_hi:       float = 1.4     # 1 / 0.064 times smaller
#     priors_tau_lo:      float = 1e-3  # than you think it is
#     priors_tau_hi:      float = 1e3   # going to be !!!!!!!!!!!!
#     priors_xi_lo:       float = 1e-3
#     priors_xi_hi:       float = 1e3
#     priors_gamma_min:   float = 1e-1
#     priors_gamma_max:   float = 1e1
#     priors_nu_min:      float = 1e-1
#     priors_nu_max:      float = 1e1
#     priors_scale_min:   float = 1e0  ## SCALING IS COUNTS / BIN
#     priors_scale_max:   float = 1e4


class MakePriors(MakeKeys):
    '''
        Doc string goes here.
    '''

    def __init__(self,  FRED_pulses, residuals_sg, lens,
                        ## just a separating line
                        priors_pulse_start, priors_pulse_end,
                        priors_td_lo = None,
                        priors_td_hi = None,
                        priors_bg_lo        = 1e-1,  ## SCALING IS COUNTS / BIN
                        priors_bg_hi        = 1e3,   ## SCALING IS COUNTS / BIN
                        priors_mr_lo        = 0.2,   ## which means that it is
                        priors_mr_hi        = 1.4,     # 1 / 0.064 times smaller
                        priors_tau_lo       = 1e-3,  # than you think it is
                        priors_tau_hi       = 1e3,   # going to be !!!!!!!!!!!!
                        priors_xi_lo        = 1e-3,
                        priors_xi_hi        = 1e3,
                        priors_gamma_min    = 1e-1,
                        priors_gamma_max    = 1e1,
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_scale_min    = 1e0,  ## SCALING IS COUNTS / BIN
                        priors_scale_max    = 1e5):  ## SCALING IS COUNTS / BIN):
        super(MakePriors, self).__init__(FRED_pulses, residuals_sg, lens)

        self.priors = bilbyPriorDict(conversion_function = self.make_constraints())

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
        self.populate_priors()

    def make_constraints(self):
        n = self.max_pulse + 1
        def constraint_function(parameters):
            for i in range(2, n):
                con_key = f'constraint_{i}'
                st_key1 = f'start_{i-1}'
                st_key2 = f'start_{i}'
                parameters[con_key] = parameters[st_key2] - parameters[st_key1]
            return parameters
        return constraint_function

    def populate_priors(self):
        ''' initialise priors

            Pass in **kwargs, then overwrite pulse parameters as
            applicable. Otherwise take generic parameters defined in init.

            add kwargs to list ??
        '''
        for key in self.keys:
            for i in range(1, self.max_pulse + 1):
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
                        maximum = float(self.priors_pulse_end -
                                        self.priors_pulse_start))

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
                        maximum = float(self.priors_pulse_end -
                                        self.priors_pulse_start) )

            elif 'sg_A' in key:
                self.priors[key] = bilbyLogUniform(1e-1,1e6,latex_label='res $A$')

            elif 'sg_tau' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e3,latex_label='res $\\tau$')

            elif 'sg_omega' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e3,latex_label='res $\\omega$')

            elif 'sg_phi' in key:
                self.priors[key] = bilbyUniform(-np.pi,np.pi,latex_label='res $\\phi$')

            elif 't_0' in key: ## deprecated I think
                pass

            else:
                print('Key not found : {}'.format(key))

    def return_prior_dict(self):
        return self.priors


class PoissonRate(MakeKeys, bilby.Likelihood):
    def __init__(self, x, y, FRED_pulses, residuals_sg, lens):
        '''
            Doc string goes here.
        '''
        super(PoissonRate, self).__init__(FRED_pulses, residuals_sg, lens)
        self.x = x
        self.y = y
        self.parameters = {k: None for k in self.keys} ## creates a dict

    @staticmethod
    def FRED_pulse(times, start, scale, tau, xi):
        # val = np.where(times - start <= 1e-12, 1e-12, scale * np.exp(
        # - xi * ( (tau / (times - start)) + ((times - start) / tau) - 2)))
        # # print(val)
        # return val
        return np.where(times - start <= 0, 1e-12, scale * np.exp(
        - xi * ( (tau / (times - start)) + ((times - start) / tau) - 2)))

    @staticmethod
    def sine_gaussian(times, sg_A, sg_begin, sg_tau, sg_omega, sg_phi):
        return (sg_A * np.exp(- np.square((times - sg_begin) / sg_tau)) *
                np.cos(sg_omega * times + sg_phi) )

    @staticmethod
    def one_FREDx_rate(times, start, scale, tau, xi, gamma, nu):
        return np.where(times - start <= 0, 1e-12, scale * np.exp(
        - np.power(xi * (tau / (times - start)), gamma)
        - np.power(xi * ((times - start) / tau), nu) - 2) )

    @staticmethod
    def insert_name(x, parameters, pulse_arr, key_list, rate_function):
        ''' finished by putting in lens func below '''
        rates = np.zeros(len(x))
        for j in pulse_arr:
            kwargs = { 'times' : x}
            for key in key_list:
                p_key      = key + f'_{j}'
                kwargs[key] = parameters[p_key]
            rates += rate_function(**kwargs)
        return rates

    @staticmethod
    def insert_name_lens(x, parameters, pulse_arr, key_list, rate_function):
        rates = np.zeros(len(x))
        for j in pulse_arr:
            kwargs  = { 'times' : x}
            l_kwargs  = { 'times' : x}
            for key in key_list:
                p_key           = key + f'_{j}'
                kwargs[key]     = parameters[p_key]
                l_kwargs[key]   = parameters[p_key]
            rates += rate_function(**kwargs)
            try:
                l_kwargs['start'] = l_kwargs['start'] + parameters['time_delay']
            except:
                pass
            try:
                l_kwargs['sg_begin'] = l_kwargs['sg_begin'] + parameters['time_delay']
            except:
                pass
            rates += rate_function(**l_kwargs) * parameters['magnification_ratio']
        return rates

    def calculate_rate(self, x, parameters, insert_name_func):
        rates = np.zeros(len(x))
        rates+= insert_name_func(   x, parameters,  self.FRED_pulses,
                                    self.FRED_list, self.FRED_pulse)
        rates+= insert_name_func(   x, parameters,    self.residuals_sg,
                                    self.res_sg_list, self.sine_gaussian)
        try:
            rates += parameters['background']
        except:
            pass
        return np.where(np.any(rates < 0.), 0, rates)


    def log_likelihood(self):
        if self.lens:
            rate = self.calculate_rate(self.x, self.parameters, self.insert_name_lens)
        else:
            rate = self.calculate_rate(self.x, self.parameters, self.insert_name)

        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Poisson rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))
        elif np.any(rate < 0.):
            raise ValueError(("Poisson rate function returns a negative",
                              " value!"))
        elif np.any(rate == 0.):
            return -np.inf
        else:
            return np.sum(-rate + self.y * np.log(rate) - gammaln(self.y + 1))


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

    def get_directory_name(self):
        directory  = '../products/'
        directory += self.tlabel + '_model_comparison_' + str(self.nSamples)
        self.base_folder = directory
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

    def main_1_channel(self, channel, **kwargs):
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

        i           = channel
        fig, ax     = plt.subplots()
        prior_shell = MakePriors(FRED_pulses = count_FRED, residuals_sg = count_sg,
                                    lens = lens,
                                    priors_pulse_start = 0,
                                    priors_pulse_end = 100,
                                    priors_td_lo = 0,
                                    priors_td_hi = 60)
        priors = prior_shell.return_prior_dict()

        x = self.GRB.bin_left
        y = np.rint(self.GRB.counts[:,i]).astype('uint')
        likelihood = PoissonRate(x, y, count_FRED, count_sg, lens = lens)

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
        ax.plot(x, likelihood.calculate_rate(x, MAP, likelihood.insert_name),
                'k:')

        figname = self.outdir + '/' + result_label +'_rates.pdf'
        fig.savefig(figname)

    def main_4_channel(self, count_FRED, count_sg, lens):

        self.num_pulses = count_FRED[-1]
        if lens:
            self.model  = 'lens'
        else:
            self.model  = 'pulse'
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()
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
                                FRED_pulses = count_FRED,
                                residuals_sg = count_sg,
                                lens        = lens,
                                priors_pulse_start = self.priors_pulse_start,
                                priors_pulse_end = self.priors_pulse_end,
                                priors_td_lo = self.priors_td_lo,
                                priors_td_hi = self.priors_td_hi)
            priors = prior_shell.return_prior_dict()

            x = self.GRB.bin_left
            y = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood = PoissonRate(x, y, count_FRED, count_sg, lens = lens)

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
                counts_fit = likelihood.calculate_rate(x, MAP, likelihood.insert_name_lens)
            else:
                counts_fit = likelihood.calculate_rate(x, MAP, likelihood.insert_name)

            count_fits[:,i] = counts_fit
            residuals[:,i] = self.GRB.counts[:,i] - counts_fit

        widths = self.GRB.bin_right - self.GRB.bin_left
        rates  = self.GRB.counts        / widths[:,None]
        rates_fit       = count_fits    / widths[:,None]
        residual_rates  = residuals     / widths[:,None]

        self.plot_4_channel(    x = self.GRB.bin_left, y = rates,
                                y_fit = rates_fit,
                                channels = channels, y_res_fit = None)


    def plot_4_channel( self, x, y, y_fit, channels,
                        y_res_fit = None, residuals = False, offsets = None):

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
        rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}')
        SAMPLER = 'Nestle'
    else:
        SAMPLER = 'dynesty'

    GRB = load_3770(sampler = SAMPLER, nSamples = 1000)
    # GRB.main_4_channel(count_FRED  = [1, 2], count_sg = [], lens = False)
    # GRB.main_4_channel(count_FRED  = [1], count_sg = [], lens = True)
    GRB.array_job(args.indices)
    # kwargs = dict()
    # kwargs['count_FRED'] = [1]
    # kwargs['count_sg']   = []
    # GRB.get_residuals(**kwargs)
