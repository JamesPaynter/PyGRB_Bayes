import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import bayesian_blocks


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass


class BATSE_BFITS(object):
    """docstring for BATSE_BFITS."""

    def __init__(self):
        super(BATSE_BFITS, self).__init__()
        if self.datatype != 'TTE' or self.datatype != 'DISCSC':
            pass
        else:
            print('Analysing BATSE bfits  data')


class BATSETTEList(object):
    """docstring for BATSETTEList."""

    def __init__(self, **kwargs):
        super(BATSETTEList, self).__init__()
        if self.datatype != 'TTE_list':
            pass
        else:
            if self.verbose:
                print('Analysing BATSE TTE list data')
            print(live_detectors)
            # open a FITS file
            data_path = f'./data/tte_list_{self.trigger}.fits'
            self.path = Path(__file__).parent / data_path
            hdul = fits.open(self.path)
            self.get_energy_bin_edges(hdul[1].data)

            count_data  = hdul[2].data
            self.detectors    = np.arange(8)
            self.det_count    = np.zeros( 8)
            self.photon_list  = []
            self.channel_list = []
            for i in range(8):
                det, num_phots, a1, a2, photons, channels = count_data[i]
                self.det_count[i] = num_phots
                self.photon_list.append(photons)
                self.channel_list.append(channels)
            if live_detectors in kwargs:
                self.live_detectors = kwargs['live_detectors']
            else:
                self.live_detectors = np.arange(8)
            # if self.live_detectors is None:
                # self.
            self.channel_1_times = self.sum_detectors(1)
            self.channel_2_times = self.sum_detectors(2)
            self.channel_3_times = self.sum_detectors(3)
            self.channel_4_times = self.sum_detectors(4)
            self.channels = [   self.channel_1_times, self.channel_2_times,
                                self.channel_3_times, self.channel_4_times]
            self.channel_x_times = self.sum_channels([1,2,3,4])
            self.sampling_rate = self.get_sampling_rate()
            # self.plot_arrival_histogram()
            times, ch1_rts = self.interpolate_bins(1)
            times, ch2_rts = self.interpolate_bins(2)
            times, ch3_rts = self.interpolate_bins(3)
            times, ch4_rts = self.interpolate_bins(4)
            if self.verbose:
                print(times.shape)
                print(ch1_rts.shape)
                print(ch2_rts.shape)
                print(ch3_rts.shape)
                print(ch4_rts.shape)

            self.bin_left  = times
            self.bin_right = times + self.sampling_rate
            self.counts    = np.zeros((4, len(self.bin_left)))
            self.counts[0,:] = ch1_rts
            self.counts[1,:] = ch2_rts
            self.counts[2,:] = ch3_rts
            self.counts[3,:] = ch4_rts
            self.counts = self.counts.T
            # self.counts    = np.vstack((ch1_rts.T, ch2_rts.T, ch3_rts.T, ch4_rts.T))

    def get_energy_bin_edges(self, energy_data):
        ### 8 detectors x 4 bins => 5 edges
        self.energy_bin_edges = np.zeros((8,5))
        for i in range(len(energy_data)):
            self.energy_bin_edges[i,:] = energy_data[i][3]
        if self.verbose:
            print('\n\nThe energy bin edges are (keV):')
            print(self.energy_bin_edges)
            print('\n\n')

    def sum_detectors(self, j):
        ''' j indexes channel '''
        ch_j = np.array([])
        ## sum photons in the same channel and sort by arrival time
        for k in self.live_detectors:
            ch_j_det_k = self.photon_list[k][self.channel_list[k] == j]
            ch_j       = np.sort(np.hstack((ch_j, ch_j_det_k)))
        return ch_j

    def sum_channels(self, channels):
        try:
            channels = np.array(channels, dtype = 'i')
        except:
            pass
        idx = channels - 1
        return np.sort(np.hstack(([self.channels[i] for i in idx])))

    def get_sampling_rate(self, numbins = 100):
        unique_times  = np.unique(self.channel_x_times)
        sort_diff     = np.sort(np.diff(unique_times))
        logbins       = np.geomspace(sort_diff[0], sort_diff[-1], numbins)
        sampling_rate = np.mean(sort_diff[sort_diff < logbins[1]])
        std           = np.std( sort_diff[sort_diff < logbins[1]])
        if self.verbose:
            print(  'The sampling rate is %.6f +/- %.6f us'
                     % (sampling_rate * 1e6, std * 1e6))
        return sampling_rate

    def interpolate_bins(self, channel = 'sum'):
        if channel == 'sum':
            arrival_times = self.channel_x_times
            string        = 'channel_sum'
        else:
            arrival_times = self.channels[int(channel - 1)]
            string        = 'channel_' + str(int(channel))

        direc     = 'TTE_list_data/'
        mkdir(direc)
        path      = direc + string
        count_str = path + '_counts.npy'
        bin_str   = path + '_bins.npy'
        diff_str  = path + '_diff.npy'
        unique, counts = np.unique(arrival_times, return_counts = True)
        sttt,endd = self.channel_x_times[0], self.channel_x_times[-1]
        num_bins  = int( ( endd - sttt )
                        / self.sampling_rate )
        try:
            ## not channel dependent name saves??
            self.interpolated_counts = np.load(count_str, allow_pickle = False)
            self.interpolated_bins   = np.load(bin_str, allow_pickle = False)
            difference               = np.load(diff_str, allow_pickle = False)
            new_bins = self.interpolated_bins
            new_counts = self.interpolated_counts
            print('Loaded previously interpolated data.')
        except:
            print('Interpolating data.')
            new_bins   = np.linspace(arrival_times[0], (arrival_times[0] +
                        (self.sampling_rate * num_bins)), num_bins  )
            og_bin_no  = len(unique)
            ## create a counts bin of req. len
            new_counts = np.zeros(len(new_bins))
            ## create a zero array of length of original arr
            difference = np.zeros(og_bin_no)
            for i in range(og_bin_no):
                ## find posn of photon in new bins cf old time
                difference[i] = np.abs(new_bins - unique[i]).min()
                new_counts[np.abs(new_bins - unique[i]).argmin()] = counts[i]
            self.interpolated_counts = new_counts
            self.interpolated_bins   = new_bins
            print('Finished data interpolation.')
            np.save(diff_str,  difference, allow_pickle = False)
            np.save(count_str, new_counts, allow_pickle = False)
            np.save(bin_str,   new_bins,   allow_pickle = False)
        return new_bins, new_counts

        # if self.verbose:
        #     sort_diff = np.sort(difference)
        #     bins      = np.linspace(sort_diff[0], sort_diff[-1], num_bins)
        #     plt.hist(sort_diff, bins = bins)
        #     plt.xlabel('Difference in arrival time (us)', fontsize = 12)
        #     plt.ylabel('Histogram counts', fontsize = 12)
        #     plt.show()

    def tte_bayesian_blocks(self, channel = 'sum'):
        ''' from astropy import scargle.BB '''
        string = 'tte_bayesian_blocks'
        edges_str = 'tte_bayesian_blocks'
        if self.verbose:
            print('Initiating Bayesian Blocks')
        if channel == 'sum':
            counts = np.sum(self.counts, axis = 1)
            string += '_sum' + '.pdf'
            edges_str += '_sum'
        else:
            counts = self.counts[:,int(channel - 1)]
            string += '_' + str(channel) + '.pdf'
            edges_str += '_' + str(channel)
        edges = bayesian_blocks(t = self.bin_left,
                                ## heaviside flattens array to binary
                                ## (some bins have 2 photons)
                                ## shouldn't matter too much
                                x = np.heaviside( counts , 0),
                                sigma = self.sampling_rate,
                                fitness = 'regular_events',
                                dt = self.sampling_rate)
        if self.verbose:
            print('Plotting Bayesian Blocks.')
        np.save(edges_str, edges, allow_pickle = False)
        plt.hist(self.interpolated_counts, bins = edges)
        plt.save(string)


    def plot_arrival_histogram(self, channel = 'sum', numbins = 100):
        if channel == 'sum':
            arrival_times = self.channel_x_times
        else:
            arrival_times = self.channels[int(channel - 1)]

        unique_times, counts = np.unique(arrival_times, return_counts = True)
        sort_diff = np.sort(np.diff(unique_times))
        logbins   = np.geomspace(sort_diff[0], sort_diff[-1], numbins)
        plt.hist(sort_diff, bins = logbins)
        plt.xscale('log')
        plt.xlabel('Time to next photon (s)', fontsize = 12)
        plt.ylabel('Histogram counts', fontsize = 12)
        plt.show()

    def bin_and_plot(self, binsize = 0.0050000027):
        start = self.channel_x_times[ 0]
        finis = self.channel_x_times[-1]
        Nbins = int( (finis - start) / binsize )
        bins  = np.linspace(start, start + binsize*Nbins, Nbins)
        colours = ['r', 'orange', 'g', 'b']
        for i in range(4):
            ch, binss = np.histogram(self.channels[i], bins=bins)
            plt.plot(bins[0:-1], ch, color = colours[i], drawstyle = 'steps')
        plt.show()


class BATSEGRB(BATSE_BFITS, BATSETTEList):
    """docstring for BATSEGRB."""

    def __init__(self, trigger, datatype, verbose = True, **kwargs):
        self.trigger   = trigger
        self.datatype  = datatype
        self.verbose   = verbose
        self.directory = './data/'
        self.path = Path(__file__).parent / self.directory
        mkdir(self.path)
        print('The trigger number is %i' % self.trigger)
        print(kwargs)
        super(BATSEGRB, self).__init__()#**kwargs)


# GRB = BATSEGRB(3770, 'TTE_list', live_detectors = np.arange(5,8))
# GRB.bin_and_plot()
# GRB.tte_bayesian_blocks()
