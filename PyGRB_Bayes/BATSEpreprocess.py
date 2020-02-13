"""
A preprocessing module to unpack the BATSE tte and discsc bfits FITS files.
Written by James Paynter, 2020.
"""

from abc import ABCMeta
from pathlib import Path

import numpy as np
from astropy.io import fits


class EmptyGRB(object):
    """ EmptyGRB for Bilby signal injections. """

    def __init__(self, bin_left, bin_right, rates):
        super(EmptyGRB, self).__init__()
        assert(len(bin_left) == len(bin_right))
        assert(bin_right[1:] - bin_left[:-1] >= 0).all()

    @property
    def bin_left(self):
        """ Property assures that bin_left is np array. """
        return self._bin_left

    @bin_left.setter
    def bin_left(self, bin_left):
        if not isinstance(bin_left, np.ndarray):
            bin_left = np.array([bin_left])
        self._bin_left = bin_left

    @property
    def bin_right(self):
        """ Property assures that bin_right is np array. """
        return self._bin_right

    @bin_right.setter
    def bin_right(self, bin_right):
        if not isinstance(bin_right, np.ndarray):
            bin_right = np.array([bin_right])
        self._bin_right = bin_right

    @property
    def rates(self):
        """ Property assures that rates is np array. """
        return self._rates

    @rates.setter
    def rates(self, rates):
        if not isinstance(rates, np.ndarray):
            rates = np.array([rates])
        self._rates = rates




class SignalFramework(metaclass=ABCMeta):
    """
    Defines the :class:`~SignalFramework` class of the *PyGRB_Bayes* package.
    This is an abstract method that contains the common code to each satellite
    to take the processed fits files and prepare them for analysis.
    """

    def __init__(self, times, bgs):
        """
        Initialize the :class:`~SignalFramework` abstract class. This class
        should be inherited by each Satellite's child class and the init ran
        after the init of the child classes.

        Parameters
        ----------
        times : tuple, or str.
            Input the times for which the GRB object is to be created. A tuple
            should be given in the form (start, finish), with start and finish
            both defined as floats (or ints). The array will be truncated based
            on the start and end times given. 'Full' will result in the full
            light-curve being generated. Often this light-curve extends several
            hundred seconds before and after the trigger time. 'T90' will search
            the BATSE 4B catalogue 'T90' burst table for the times to truncate
            the light-curve. Some BATSE bursts do not have a 'T90' listed.
        bgs : bool.
            If *True* removes the background from each channel of the data by
            calling the :meth:`~get_background` method. The method is a first
            order approximation. This parameter should be set to *False* for
            light-curve fitting with the main :mod:`~DynamicBilby` methods.
        """

        print('The analysis time interval is', times)

        ## this will be common to all code and hence go in the super
        self.bin_centres = (self.bin_left + self.bin_right) / 2
        self.bin_widths = np.round(self.bin_right - self.bin_left, 3)
        self.max_bin = np.max(self.bin_widths)
        self.sum_rates = np.sum(self.rates, axis=1)
        self.sum_counts = np.sum(self.counts, axis=1)

        self.background = self.get_background()
        self.bg_counts = np.array([self.background[i] * (
                self.bin_right - self.bin_left) for i in
                                   range(4)]).T
        print('The background rates are', self.background)
        self.rates_bs = self.rates - self.background
        self.count_bs = self.counts - self.bg_counts

        self.sum_bs = np.sum(self.rates_bs, axis=1)
        self.sum_cnt_bs = np.sum(self.count_bs, axis=1)

        if bgs:
            self.rates = self.rates_bs
            self.counts = self.count_bs
            self.sum_rates = self.sum_bs
            self.sum_counts = self.sum_cnt_bs

        ### generic / common
        if type(self.times) is tuple:
            (self.t90_st, self.end) = self.times
            self.cut_times = True

        elif self.times == 'full':
            print('full steam ahead')
            self.start = self.bin_left[0]
            self.end = self.bin_right[-1]
            self.cut_times = False

        elif self.times == 'T90':
            print('Using the T90')
            print('Starting at T5 = %.3f seconds.' % self.t90_st)
            print('Ending at T95  = %.3f seconds.' % self.end)
            self.cut_times = True


        else:
            raise ValueError("%s is not valid.\nChoose either 'T90', "
                     "or enter the start and end times as a tuple" % self.times)

        print('''I'm up to cutting times ''')
        if self.cut_times:
            ### finds index of array that best matches the time given in table
            self.start  = (np.abs(self.bin_left - self.t90_st)).argmin()
            self.stop   = (np.abs(self.bin_left - self.end)).argmin()
            self.bin_left       = self.bin_left     [self.start:self.stop]
            self.bin_right      = self.bin_right    [self.start:self.stop]
            self.rates          = self.rates        [self.start:self.stop]
            self.errors         = self.errors       [self.start:self.stop]
            self.count_bg       = self.counts       [self.start:self.stop]
            self.counts         = self.counts       [self.start:self.stop]
            self.count_bs       = self.count_bs     [self.start:self.stop]
            self.count_err      = self.count_err    [self.start:self.stop]
            self.sum_cnt_bs     = self.sum_cnt_bs   [self.start:self.stop]
            self.bin_centres    = self.bin_centres  [self.start:self.stop]
            self.bin_widths     = self.bin_widths   [self.start:self.stop]
            self.sum_rates      = self.sum_rates    [self.start:self.stop]
            self.max_bin        = np.max(self.bin_widths)

        if self.light_GRB:
            self.return_GRB()

    def get_background(self):
        """
            Creates background from bins of width greater than nominated
            resolution. ie for 64ms uses the larger 1024ms+ bins
        """
        # return np.mean(self.rates[self.bin_widths > self.res], axis=0)
        return np.mean(self.rates[self.bin_widths > 0.065], axis=0)

    def return_GRB(self):
        """ Creates a new cleaner GRB object, dropping the scaffolding. """
        return EmptyGRB(self.bin_left, self.bin_right, self.rates)


class BATSESignal(SignalFramework):
    """ Inherits from the SignalFramework abstract class. """

    def __init__(self,  burst = None, datatype = None, times = None,
                        bgs = False, light_GRB = True):
                 # burst,  ## does not matter to super
                 # datatype='discsc',  ## does not matter to super
                 # times='T90',  ## matters to super
                 # bgs=True):  ## matters to super
        """
        Initialize the :class:`~BATSESignal` class. This class inherits from the
        SignalFramework abstract class. The parent init should run at the end
        of the child init.

        Parameters
        ----------
        burst : int.
            The BATSE burst trigger ID for locating the relevant datafile.
        datatype : str.
            The datatype should be given as either 'discsc', or 'tte'
        times : tuple, or str.
            Input the times for which the GRB object is to be created. A tuple
            should be given in the form (start, finish), with start and finish
            both defined as floats (or ints). The array will be truncated based
            on the start and end times given. 'Full' will result in the full
            light-curve being generated. Often this light-curve extends several
            hundred seconds before and after the trigger time. 'T90' will search
            the BATSE 4B catalogue 'T90' burst table for the times to truncate
            the light-curve. Some BATSE bursts do not have a 'T90' listed.
        bgs : bool.
            If *True* removes the background from each channel of the data by
            calling the :meth:`~get_background` method. The method is a first
            order approximation. This parameter should be set to *False* for
            light-curve fitting with the main :mod:`~DynamicBilby` methods.
        light_GRB: bool.
            If *True* generates a new GRB object without all the scaffolding and
            extra methods and properties of this class. Used for Bilby analsis.
        """

        self.times   = times
        self.colours = ['red', 'orange', 'green', 'blue']
        self.labels  = ['  20 - 50   keV', '  50 - 100 keV',
                         '100 - 300  keV', '300 +      keV']
        # test types
        if not isinstance(burst, int):
            raise ValueError(
                'Input variable `burst` should be an integer. '
                'Is {} when it should be int.'.format(type(burst)))
        else:
            self.burst = burst
        if not isinstance(datatype, str):
            raise ValueError(
                'Input variable `datatype` should be a string. '
                'Is {} when it should be str.'.format(type(datatype)))
        elif datatype not in ['discsc','tte']:
            print(datatype)
            print(type(datatype))
            raise AssertionError(
                'Input variable `datatype` is {} when it '
                'should be `discsc` or `tte`.'.format(datatype))
        else:
            self.datatype = datatype

        self.light_GRB = light_GRB
        # uses self.xx so that it has passed the AssertionErrors
        relative_path = f'./data/{self.datatype}_bfits_{self.burst}.fits'
        # put in a comment here about what Path does ?
        self.path = Path(__file__).parent / relative_path
        # with closes the file automatically after finished.
        with fits.open(self.path) as hdu_list:
            self.data = hdu_list[-1].data
            ### initialise data arrays from fits file, over entire data set
            self.bin_left  = np.array(self.data['TIMES'][:, 0])
            self.bin_right = np.array(self.data['TIMES'][:, 1])
            self.rates = np.array(self.data['RATES'][:, :])
            # errors in BATSE rates are just calculated by scaling Poissonian
            # from counts
            self.errors = np.array(self.data['ERRORS'][:, :])
            self.counts = np.array([np.multiply(self.rates[:, i],
                    self.bin_right - self.bin_left) for i in range(4)]).T

            self.count_err = np.sqrt(self.counts)
            self.rates_max = self.data['RATES'][:, :].max()
            if self.times == 'T90':
                # try:
                ### read T90's from BATSE 4B catalogue
                print('reading a T90')
                self.directory = ' '
                # this file is ??
                __str = self.directory + 't90_table.txt'
                length_table = np.genfromtxt(__str)
                self.burst_no = length_table[:, 0]
                self.t90_list = length_table[:, 4]
                self.t90_err_list = length_table[:, 5]
                self.t90_st_list = length_table[:, 6]
                ### BATSE specific ^^
                ### generic / common
                self.t90 = float(self.t90_list[self.burst_no == self.burst])
                self.t90_err = float(self.t90_err_list[self.burst_no == self.burst])
                self.t90_st = float(self.t90_st_list[self.burst_no == self.burst])
                self.end = self.t90_st + self.t90
                print(self.t90_st, self.end)
                self.cut_times = True
                # except:
                #     print('The 4B catalogue does not contain the T90 for this burst...')
                #     print('Please specify the start and end times as a tuple.')

            super().__init__(times, bgs)



### create a string for the path of the relevant data files
# if datatype == 'discsc':
#     relative_path = './data/discsc_bfits_' + str(self.burst) + '.fits'
#     self.path = Path(__file__).parent / relative_path
#
#     self.resolution = '64 ms'
#     self.res = 0.064
#     ### 64 ms bins
#
# elif datatype == 'tte':
#     relative_path = './data/tte_bfits_' + str(self.burst) + '.fits'
#     self.path = Path(__file__).parent / relative_path
#
#     self.resolution = '5 ms'
#     self.res = 0.005
#     ### combination data
#     ### tte has some resolution down to 2 ms
#     ### don't think the above statement is true 18/2/19
# else:
#     raise ValueError(
#     "Please select from the 'tte' or 'discsc' datatypes and try again")
# ### uses the open method to read data from FITS files
            ## print comments from fits header
            # print(
            # '''The following notes and comments are from the FITS header.\n''')
            # h_list = []
            # for i in hdu_list[-1].header:
            #     h_list.append(i)
            # for i in range(len(hdu_list[-1].header)):
            #     if h_list[i] == 'COMMENT' or h_list[i] == 'NOTE':
            #         print(h_list[i], hdu_list[-1].header[i])
            # print('\n\n')
