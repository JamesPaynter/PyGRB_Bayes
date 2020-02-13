'''
A preprocessing module to unpack the BATSE tte and discsc bfits FITS files.
Written by James Paynter, 2019.
'''

# import os, sys
# path = os.path.dirname(__file__)
# path = os.path.join(path, 'bin')
# if path not in sys.path:
#     sys.path.append(path)


import math
import numpy as np
import matplotlib.pyplot as plt

from abc        import ABCMeta
from scipy      import signal
from astropy.io import fits
from math       import isclose

from pathlib import Path


#### ** UPDATE THIS TO BE A BASE CLASS FOR SWIFT BAT AND BATSE TO INHERIT FROM
###  ** DELETED THE INFO, BACKGROUND AND TESTVARIABLE METHODS **

class SignalFramework(metaclass = ABCMeta):
    '''
    Defines the method for creating the time series to be plotted or analysed.
    This is an abstract method that contains the common code to each satellite
    class.
    '''
    def __init__(self,
        times,
        bgs,
        PRINT = False):
        print('The analysis time interval is', times)

        ## this will be common to all code and hence go in the super
        self.bin_centres = (self.bin_left + self.bin_right) / 2
        self.bin_widths  = np.round(self.bin_right - self.bin_left, 3)
        self.max_bin     = np.max(self.bin_widths)
        self.sum_rates   = np.sum(self.rates,  axis=1)
        self.sum_counts  = np.sum(self.counts, axis=1)

        self.background  = self.get_background()
        self.bg_counts   = np.array([self.background[i] * (
                            self.bin_right - self.bin_left) for i in
                            range(4)]).T
        print('The background rates are', self.background)
        self.rates_bs    = self.rates  - self.background
        self.count_bs    = self.counts - self.bg_counts

        self.sum_bs      = np.sum(self.rates_bs, axis=1)
        self.sum_cnt_bs  = np.sum(self.count_bs, axis=1)

        if bgs:
            self.rates      = self.rates_bs
            self.counts     = self.count_bs
            self.sum_rates  = self.sum_bs
            self.sum_counts = self.sum_cnt_bs




        ### generic / common
        if type(self.times) is tuple:
            (self.t90_st, self.end) = self.times
            self.cut_times = True

        elif self.times == 'full':
            print('full steam ahead')
            self.start       = self.bin_left[0]
            self.end         = self.bin_right[-1]
            self.cut_times = False

        elif self.times == 'T90':
            print('Using the T90')
            print('Starting at T5  = %.3f seconds.' % self.t90_st)
            print('Ending at T95  = %.3f seconds.' % self.end)
            self.cut_times = True


        else:
            raise ValueError("%s is not valid.\nChoose either 'T90', "
            "or enter the start and end times as a tuple" % self.times)

        print('''I'm up to cutting times ''')
        if self.cut_times:
            ### finds index of array that best matches the time given in table
            self.start       = (np.abs(self.bin_left - self.t90_st)).argmin()
            self.stop        = (np.abs(self.bin_left - self.end   )).argmin()
            self.bin_left    = self.bin_left   [self.start:self.stop]
            self.bin_right   = self.bin_right  [self.start:self.stop]
            self.rates       = self.rates      [self.start:self.stop]
            self.errors      = self.errors     [self.start:self.stop]
            self.count_bg    = self.counts     [self.start:self.stop]
            self.counts      = self.counts     [self.start:self.stop]
            self.count_bs    = self.count_bs   [self.start:self.stop]
            self.count_err   = self.count_err  [self.start:self.stop]
            self.sum_cnt_bs  = self.sum_cnt_bs  [self.start:self.stop]
            self.bin_centres = self.bin_centres[self.start:self.stop]
            self.bin_widths  = self.bin_widths [self.start:self.stop]
            self.sum_rates   = self.sum_rates  [self.start:self.stop]
            self.max_bin     = np.max(self.bin_widths)



    def get_background(self):
        '''
        Creates background from bins of width greater than nominated
        resolution.
        ie for 64ms uses the larger 1024ms+ bins
        Might need to be updated for tte bfits data.
        old code commented, `update' should work for tte too
        Need to update again for 1024ms data if used
        ** Need to make robust for publishing ** '''
        # return np.mean(self.rates[self.bin_widths > self.res], axis=0)
        return np.mean(self.rates[self.bin_widths > 0.065], axis=0)

class BATSESignal(SignalFramework):
    '''
    Inherits from the SignalFramework superclass.
    '''
    def __init__(self,
    burst,                  ## does not matter to super
    datatype    = 'discsc', ## does not matter to super
    times       = 'T90',    ## matters to super
    bgs         = True):    ## matters to super

        self.colours = ['red', 'orange', 'green', 'blue']
        self.labels  = ['  20 - 50   keV', '  50 - 100 keV',
                          '100 - 300 keV', '300 +      keV']
        self.times   = times

        ### test input variables and assign
        # self.test_int(burst,'Burst Number')
        # self.test_str(datatype,'datatype')
        # self.test_str(datatype,'times')
        self.burst      = burst
        self.datatype   = datatype
        ### create a string for the path of the relevant data files
        self.directory = ' ' #'''C:/Users/James/Documents/University/Physics/GAMMA RAY BURSTS/My Work/'''
        if  datatype == 'discsc':
            relative_path = './data/discsc_bfits_' + str(self.burst) + '.fits'
            self.path = Path(__file__).parent / relative_path

            self.resolution    = '64 ms'
            self.res           = 0.064
            ### 64 ms bins

        elif datatype == 'tte':
            relative_path = './data/tte_bfits_' + str(self.burst) + '.fits'
            self.path = Path(__file__).parent / relative_path

            self.resolution    = '5 ms'
            self.res           = 0.005
            ### combination data
            ### tte has some resolution down to 2 ms
            ### don't think the above statement is true 18/2/19
        else:
            raise ValueError(
            "Please select from the 'tte' or 'discsc' datatypes and try again")
        ### uses the open method to read data from FITS files

        with fits.open(self.path) as hdu_list:
            self.data = hdu_list[-1].data

        ## closes the file automatically
        ### print notes from data file

        ## commented out for now
        ## commented out for now
        ## commented out for now
        ## commented out for now
        # print(
        # '''The following notes and comments are from the FITS header.\n''')
        # h_list = []
        # for i in hdu_list[-1].header:
        #     h_list.append(i)
        # for i in range(len(hdu_list[-1].header)):
        #     if h_list[i] == 'COMMENT' or h_list[i] == 'NOTE':
        #         print(h_list[i], hdu_list[-1].header[i])
        # print('\n\n')
        ## commented out for now
        ## commented out for now
        ## commented out for now


            ### initialise data arrays from fits file, over entire data set
            self.bin_left    = np.array(self.data['TIMES'][:, 0])
            self.bin_right   = np.array(self.data['TIMES'][:, 1])
            self.rates       = np.array(self.data['RATES'][:, :])
            self.errors      = np.array(self.data['ERRORS'][:, :])
            self.counts      = np.array([np.multiply(self.rates[:,i],
                                self.bin_right - self.bin_left) for i in
                                range(4)]).T
            self.count_err   = np.sqrt(self.counts)
            self.rates_max   = self.data['RATES'][:, :].max()
            if self.times == 'T90':
                # try:
                    ### read T90's from BATSE 4B catalogue
                print('reading a T90')
                __str             = self.directory + 't90_table.txt'
                length_table      = np.genfromtxt(__str)
                self.burst_no     = length_table[:,0]
                self.t90_list     = length_table[:,4]
                self.t90_err_list = length_table[:,5]
                self.t90_st_list  = length_table[:,6]
                ### BATSE specific ^^
                ### generic / common
                self.t90     = float(self.t90_list    [self.burst_no == self.burst])
                self.t90_err = float(self.t90_err_list[self.burst_no == self.burst])
                self.t90_st  = float(self.t90_st_list [self.burst_no == self.burst])
                self.end     = self.t90_st + self.t90
                print(self.t90_st, self.end)
                self.cut_times = True
                # except:
                #     print('The 4B catalogue does not contain the T90 for this burst...')
                #     print('Please specify the start and end times as a tuple.')

            super().__init__(times, bgs)
