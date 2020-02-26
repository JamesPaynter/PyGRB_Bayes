import sys, os
from abc import ABCMeta


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
        mylist = self.model['count_FRED'] + self.model['count_FREDx']
        ## set gets the unique values of the list
        myset  = set(mylist)
        try:
            self.num_pulses = max(myset) ## WILL NEED EXPANDING
        except:
            self.num_pulses = 0

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
            if i in self.model['count_FRED']:
                string += 'F'
            elif i in self.model['count_FREDx']:
                string += 'X'
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
