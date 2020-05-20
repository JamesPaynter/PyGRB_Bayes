"""
A preprocessing module to unpack the BATSE tte and discsc bfits FITS files.
Written by James Paynter, 2020.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

from PyGRB.fetch.get_BATSE        import GetBATSEBurst
from PyGRB.preprocess.abstract    import SignalFramework

def make_GRB(**kwargs):
    GRB = BATSESignal(**kwargs)
    return GRB.return_GRB()

class BATSESignal(SignalFramework):
    """ Inherits from the SignalFramework abstract class. """

    def __init__(self,  burst: int = None, datatype: str = None,
                        times = None,
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
        """

        self.colours   = ['red', 'orange', 'green', 'blue']
        self.clabels   = ['1', '2', '3', '4']
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

        self.kwargs    = {  'colours'   : self.colours,
                            'clabels'   : self.clabels,
                            'labels'    : self.labels,
                            'datatype'  : self.datatype,
                            'burst'     : self.burst,
                            'satellite' : 'BATSE'}

        self.times     = times
        self.light_GRB = light_GRB
        # GetBATSEBurst downdloads file if it does not exist
        fetch = GetBATSEBurst(trigger = self.burst, datatype = self.datatype)
        # with closes the file automatically after finished.
        with fits.open(fetch.path) as hdu_list:
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
            self.t90_st, self.end = self.bin_left[0], self.bin_right[-1]
            try:
                (self.t90_st, self.end) = times
            except:
                if times == 'T90':
                    self._read_T90_table()
            super().__init__(times, bgs)

    def _open_T90_excel(self):
        """ Open the BATSE 4B csv information file. """
        xls_file = f'../data/BATSE_4B_catalogue.xls'
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

    def _read_T90_table(self):
        """ Opens the BATSE T90 bursts as a pandas object. Searches for the
            current burst's trigger, T90, T90 error, and T90 start time.
            Throws an exception if the burst is not found in the T90 table.
            (i.e. no T90 exists for this burst).
        """
        table = self._open_T90_excel()
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
