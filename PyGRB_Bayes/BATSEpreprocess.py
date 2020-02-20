"""
A preprocessing module to unpack the BATSE tte and discsc bfits FITS files.
Written by James Paynter, 2020.
"""

from abc import ABCMeta
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


class EmptyGRB(object):
    """ EmptyGRB for Bilby signal injections. """

    def __init__(self, bin_left, bin_right, rates):
        """
        Initialize the :class:`~SignalFramework` abstract class. This class
        should be inherited by each Satellite's child class and the init ran
        after the init of the child classes.

        Parameters
        ----------
        bin_left : np.array.
            The parameter specifies the left bins of the GRB.

        bin_left : np.array.
            The parameter specifies the right bins of the GRB.

        rates : np.array.
            The parameter specifies rates at each bin of the GRB.
            If multi-channel, it should be in the form
        """
        super(EmptyGRB, self).__init__()

        if not isinstance(bin_left, np.ndarray):
            raise ValueError(
                'Input variable `bin_left` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(bin_left)))
        self.bin_left = bin_left

        if not isinstance(bin_right, np.ndarray):
            raise ValueError(
                'Input variable `bin_right` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(bin_right)))
        self.bin_right = bin_right

        if not isinstance(rates, np.ndarray):
            raise ValueError(
                'Input variable `rates` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(rates)))
        self.rates = rates

        # assert right and left bin arrays are equal length
        assert(len(bin_left) == len(bin_right))
        # assert rates array is also the same length
        assert(len(bin_left)) == max(np.shape(rates))
        # assert that each left bin begins after the last right bin finishes
        assert(((bin_left[1:] - bin_right[:-1]) >= -1e-3).all())
        # assert rates has the right shape
        try:
            (a,b) = np.shape(rates)
        except:
            a, b = 1, 0
        assert(a > b)


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
        """ Creates background from bins of width greater than nominated
            resolution. ie for 64ms uses the larger 1024ms+ bins.
        """
        return np.mean(self.rates[self.bin_widths > 0.065], axis=0)

    def return_GRB(self):
        """ Creates a new GRB object with only bins and rates. """
        return EmptyGRB(self.bin_left, self.bin_right, self.rates)


class BATSESignal(SignalFramework):
    """ Inherits from the SignalFramework abstract class. """

    def __init__(self,  burst: int = None, datatype: str = None, times = None,
                        bgs: bool = False, light_GRB: bool = True):
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
            extra methods and properties of this class. Used for Bilby analysis.
        """

        self.colours   = ['red', 'orange', 'green', 'blue']
        self.labels    = ['  20 - 50   keV', '  50 - 100 keV',
                          ' 100 - 300  keV', '300 +      keV']
        self.datatypes = {'discsc':'discsc', 'tte':'tte'}
        try:
            self.burst = int(burst)
        except:
            raise ValueError(
                'Input variable `burst` should be an integer. '
                'Is {} when it should be int.'.format(type(burst)))
        try:
            self.datatype = self.datatypes[datatype]
        except:
            raise AssertionError(
                'Input variable `datatype` is {} when it '
                'should be `discsc` or `tte`.'.format(datatype))

        self.times     = times
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
            self.rates     = np.array(self.data['RATES'][:, :])
            # errors in BATSE rates are calculated by scaling the counts
            # count errors are calculated with as Poisson errors = sqrt(counts)
            self.errors = np.array(self.data['ERRORS'][:, :])
            self.counts = np.array([np.multiply(self.rates[:, i],
                    self.bin_right - self.bin_left) for i in range(4)]).T

            self.count_err = np.sqrt(self.counts)
            # could delete this line
            # self.rates_max = self.data['RATES'][:, :].max()
            self.t90_st, self.end = self.bin_left[0], self.bin_right[-1]
            try:
                (self.t90_st, self.end) = times
            except:
                if times == 'T90':
                    self.read_T90_table()
            super().__init__(times, bgs)

    def open_T90_excel(self):
        xls_file = f'./data/BATSE_4B_catalogue.xls'
        path = Path(__file__).parent / xls_file
        cols = ['trigger_num', 't90', 't90_error', 't90_start']
        dtypes = {  'trigger_num': np.int32, 't90' : np.float64,
                    't90_error' : np.float64, 't90_start' : np.float64}
        try:
            table = pd.read_excel(path, sheet_name = 'batsegrb', header = 0,
                                    usecols = cols, dtype = dtypes)
            return table
        except FileNotFoundError as fnf_error:
                    print(fnf_error)

    def read_T90_table(self):

        table = self.open_T90_excel()
        self.burst_list     = table['trigger_num']
        self.t90_list       = table['t90']
        self.t90_err_list   = table['t90_error']
        self.t90_st_list    = table['t90_start']
        try:
            self.t90 = float(self.t90_list[self.burst_list == self.burst])
            self.t90_err = float(self.t90_err_list[self.burst_list == self.burst])
            self.t90_st = float(self.t90_st_list[self.burst_list == self.burst])
            self.end = self.t90_st + self.t90
        except:
             raise Exception('There is no T90 for this trigger in the BATSE 4B'
                             'catalogue. Try `full` or enter custom times as a'
                             'tuple, i.e. (start, end).')
