import sys, os
from abc import ABCMeta
import numpy as np

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

class Admin(metaclass=ABCMeta):
    """
    Defines the :class:`~Admin` class of the *PyGRB_Bayes* package.
    This is an abstract class that contains the private methods of the
    :class:`~BilbyObject` class. These methods predominantly translate fitting
    parameters into labels for file or folder names and vice versa.
    """
    def __init__(self):
        super(Admin, self).__init__()

    def _get_trigger_label(self):
        """ Fills a trigger number (int) to 4 digits with leading zeroes. """
        tlabel = str(self.trigger)
        if len(tlabel) < 4:
            tlabel = ''.join('0' for i in range(4-len(tlabel))) + tlabel
        return tlabel

    def _get_max_pulse(self):
        """
        Finds the number of pulses in a model by counting the highest
        pulse number in the set. Assumes that the pulse counts are a
        continuous set of integers from 1 to num_pulses.
        """
        mylist = []
        if self.model['count_FRED']:
            mylist += self.model['count_FRED']
        if self.model['count_FREDx']:
            mylist += self.model['count_FREDx']
        if self.model['count_gauss']:
            mylist += self.model['count_gauss']
        if self.model['count_conv']:
            mylist += self.model['count_conv']
        ## set gets the unique values of the list
        myset  = set(mylist)
        try:
            self.num_pulses = max(myset) ## WILL NEED EXPANDING
        except:
            self.num_pulses = 0
        # self.num_pulses = self.model['max_pulse']
        # self.num_pulses = self.max_pulse

    def _get_base_directory(self):
        """
        Sets the directory that code products are made to be /products/ in
        the folder the script was ran from.
        """
        dir = f'products/{self.tlabel}_model_comparison_{str(self.nSamples)}'
        self.base_folder = dir

    def _get_pulse_list(self):
        """ Generates the pulse list from a model to name the out directory. """
        string = ''
        for i in range(1, self.num_pulses + 1):
            if i in self.model['count_gauss']:
                string += 'G'
            elif i in self.model['count_FRED']:
                string += 'F'
            elif i in self.model['count_FREDx']:
                string += 'X'
            elif i in self.model['count_conv']:
                string += 'C'
            if i in self.model['count_sg']:
                string += 's'
            elif i in self.model['count_bes']:
                string += 'b'
        return string

    def _get_directory_name(self):
        """
        Code changes the root directory to the directory above this file.
        Then product files (light-curves, posterior chains) are created in:
            " directory  = 'products/' "

        self.tlabel : 4 character burst trigger number

        adds '_model_comparison_' (could be removed really)

        add number of live points (~ accuracy proxy)

        add lens model or null model (if self.lens)

        add number of pulses

        add pulse keys (eg FFbXsF : Fred F <- bessel_res FREDx <- sg_res F)
        residual is attached to the proceeding pulse.

        MC counter is for testing the code over many trials --> save data
        """
        self._get_base_directory()
        directory = self.base_folder
        if self.model['lens']:
            directory += '/lens_model'
        else:
            directory += '/null_model'
        directory += '_' + str(self.num_pulses)

        directory += '_' + self._get_pulse_list()
        if self.MC_counter:
            directory += '_' + str(self.MC_counter)
        return directory

    def _get_file_string(self):
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

    def _setup_labels(self, model):
        self.model = model
        self._get_max_pulse()
        self.tlabel = self._get_trigger_label()
        self.fstring = self._get_file_string()
        self.outdir = self._get_directory_name()
        mkdir(self.outdir)

    def _sample_priors(self, channel, model):
        from PyGRB_Bayes.backend.makepriors import MakePriors
        from PyGRB_Bayes.backend.rateclass import PoissonRate
        ps = MakePriors(
                            priors_pulse_start = self.priors_pulse_start,
                            priors_pulse_end = self.priors_pulse_end,
                            priors_td_lo = self.priors_td_lo,
                            priors_td_hi = self.priors_td_hi,
                            channel = channel,
                            **model,
                            **self.kwargs)
        prior_ranges = {
                'start' : (ps.priors_pulse_start, ps.priors_pulse_end),
                'scale' : (ps.priors_scale_min,   ps.priors_scale_max),
           'background' : (ps.priors_bg_lo,       ps.priors_bg_hi),
                  'tau' : (1e-1,      1e1),
                  # 'tau' : (ps.priors_tau_lo,      ps.priors_tau_hi),
                   'xi' : (1e-1,      1e1),
                   # 'xi' : (ps.priors_xi_lo,       ps.priors_xi_hi),
                'gamma' : (ps.priors_gamma_min,   ps.priors_gamma_max),
                   'nu' : (ps.priors_nu_min,      ps.priors_nu_max),
                'sigma' : (ps.priors_sigma_lo,    ps.priors_sigma_hi),
           'time_delay' : (ps.priors_td_lo,       ps.priors_td_hi),
  'magnification_ratio' : (ps.priors_mr_lo,       ps.priors_mr_hi),
  #
            'res_begin' : (ps.priors_pulse_start, ps.priors_pulse_end),
                 'sg_A' : (1e0, 1e3),
            'sg_lambda' : (1e-3, 1e3),
             'sg_omega' : (1e-3, 1e4),
               'sg_phi' : (-np.pi, np.pi),
                'bes_A' : (1e-1, 1e6),
            'bes_Omega' : (1e-3, 1e3),
                'bes_s' : (1e-3, 1e3),
            'bes_Delta' : (-np.pi, np.pi),
                           }
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        priors = ps.return_prior_dict()
        model_keys = [*priors]
        n_axes  = 2 + len(model_keys) * 2
        width   = 7
        # arbitrary scaled height
        height  = 11
        heights = [5, 0.8]+[f for x in range(len(model_keys)) for f in (3, 0.8)]
        # constrained_layout = False -> don't mess with my placing
        fig     = plt.figure(figsize = (width, height), constrained_layout=False)
        # add an extra column on the left for the main axes labels
        spec    = gridspec.GridSpec(ncols=2, nrows=n_axes, figure=fig,
                                    height_ratios=heights,
                                    width_ratios=[0.05, 0.95],
                                    hspace=0.0, wspace=0.0)
        fig_ax1 = fig.add_subplot(spec[0, 1])


        sample_dict = self._get_middle_priors(model_keys, ps, prior_ranges)
        x = self.GRB.bin_left[1:]
        P = PoissonRate(x = x, y = np.zeros(len(x)), channel = channel, **model)
        y = P.return_line_from_sample(sample_dict)
        fig_ax1.plot(x, y)
        # list of residual channel axes to append to
        axes_list = []
        for i in range(1, n_axes - 1):
            if i % 2 == 0:
                axes_list.append(fig.add_subplot(spec[i+1, 1], frameon = False))
            else:
                axes_list.append(fig.add_subplot(spec[i+1, 1]))
        for i, k in enumerate(model_keys):
            min_dict = self._get_middle_priors(model_keys, ps, prior_ranges)
            max_dict = self._get_middle_priors(model_keys, ps, prior_ranges)
            min_dict[k], max_dict[k] = self._get_start_end(k, ps, prior_ranges)
            y_min = P.return_line_from_sample(min_dict)
            y_max = P.return_line_from_sample(max_dict)
            sk = self.strip_key(k)
            axes_list[i*2].plot(x, y_min, 'k:', label = f'{sk} = {min_dict[k]}')
            axes_list[i*2].plot(x, y_max, 'k--',label = f'{sk} = {max_dict[k]}')
            axes_list[i*2].set_ylabel(f'Vary {sk}')
            axes_list[i*2].legend()

            axes_list[2*i+1].set_xticks(())
            axes_list[2*i+1].set_yticks(())
        plot_name=f'test.pdf'
        fig.savefig(plot_name)
        plt.close(fig)

    @staticmethod
    def strip_key(key):
        import re
        index = re.search('_*[0-9]*_[a-d]$', key).group(0)
        return re.sub(index, '', key)

    @staticmethod
    def _get_start_end(key, ps, prior_ranges):
        priors = ps.return_prior_dict()
        sample_dict = {}
        p_key = Admin.strip_key(key)
        (start, stop) = prior_ranges[p_key]
        return start, stop

    @staticmethod
    def _get_middle_priors(model_keys, ps, prior_ranges):
        import re
        priors = ps.return_prior_dict()
        sample_dict = {}
        for key in model_keys:
            idx = re.search('_*[0-9]*_[a-d]$', key).group(0)
            p_key = re.sub(idx, '', key)
            (start, stop) = prior_ranges[p_key]
            middle = Admin._get_middle(start, stop, priors[f'{key}'])
            if middle:
                sample_dict[key] = middle
        return sample_dict

    @staticmethod
    def _get_middle(start, stop, prior):
        from bilby.core.prior import Uniform          as bilbyUniform
        from bilby.core.prior import LogUniform       as bilbyLogUniform
        from bilby.core.prior import Constraint       as bilbyConstraint
        if isinstance(prior, bilbyUniform):
            middle = (stop - start) / 2
            return middle
        elif isinstance(prior, bilbyLogUniform):
            middle = 10 ** ((np.log10(stop) + np.log10(start)) / 2)
            return middle
        elif isinstance(prior, bilbyConstraint):
            return None
        else:
            print(f'What is this prior instance ?? {key} : {prior}')
            return None
