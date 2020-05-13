import numpy as np
from abc import ABCMeta

from PyGRB.preprocess.grb import EmptyGRB

class SignalFramework(metaclass=ABCMeta):
    """
    Defines the :class:`~SignalFramework` class of the *PyGRB* package.
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

    def get_background(self):
        """ Creates background from bins of width greater than nominated
            resolution. ie for 64ms uses the larger 1024ms+ bins.
        """
        return np.mean(self.rates[self.bin_widths > 0.065], axis=0)

    def return_GRB(self):
        """ Creates a new GRB object with only bins and rates. """
        return EmptyGRB(self.bin_left, self.bin_right, self.counts, **self.kwargs)
