import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from DynamicBilby import BilbyObject
import pymc3

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

def load_test(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(trigger = 1, times = (-2, 50), test = True,
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    test.inject_signal()
    return test


def MC_test(index_array, **kwargs):
    channels = [0]
    GRB = load_test(**kwargs)
    GRB.counter = 0
    GRB.MC_counter = 0
    bin_size = 0.064
    nTrials = 100
    nScales = 20
    scale_array = np.geomspace(1e2, 1e5, nScales)
    for q in index_array:
        GRB.MC_counter = q
        GRB.outdir += '_' + str(q)
        mkdir(GRB.outdir)
        print(scale_array[q])
        mkdir(str(int(scale_array[q])))
        file_name_Ylens_ev = (str(int(scale_array[q])) + '/MC_Ylens_ev_'
                            + str(int(scale_array[q])) + '.txt')
        file_name_Ylens_er = (str(int(scale_array[q])) + '/MC_Ylens_er_'
                            + str(int(scale_array[q])) + '.txt')
        file_name_Nlens_ev = (str(int(scale_array[q])) + '/MC_Nlens_ev_'
                            + str(int(scale_array[q])) + '.txt')
        file_name_Nlens_er = (str(int(scale_array[q])) + '/MC_Nlens_er_'
                            + str(int(scale_array[q])) + '.txt')
        for p in range(nTrials):
            GRB.inject_signal(scale_override = scale_array[q])
            evidences_1_lens, errors_1_lens = GRB.one_FRED_lens(
                                                channels = channels,
                                                test = False, plot = True,
                                                save_all = True)
            evidences_2_FRED, errors_2_FRED = GRB.two_FRED(
                                                channels = channels,
                                                test = False, plot = True,
                                                save_all = True)
            with open(file_name_Ylens_ev, 'a+') as f:
                f.write(str(evidences_1_lens[0]))
                f.write('\n')
            with open(file_name_Ylens_er, 'a+') as f:
                f.write(str(errors_1_lens[0]))
                f.write('\n')
            with open(file_name_Nlens_ev, 'a+') as f:
                f.write(str(evidences_2_FRED[0]))
                f.write('\n')
            with open(file_name_Nlens_er, 'a+') as f:
                f.write(str(errors_2_FRED[0]))
                f.write('\n')

def run_MC():
    parser = argparse.ArgumentParser(description='Access geomspace array')

    parser.add_argument('indices', metavar='N', type=int, nargs='+',
                        help='an integer for indexing geomspace array')

    args = parser.parse_args()
    print(args.indices)
    MC_test(index_array = args.indices, nSamples = 100, sampler = 'Nestle')

def run_analysis():
    nScales = 20
    scale_array = np.geomspace(1e4, 1e7, nScales)
    array_Ylens_ev = np.zeros((nScales,100))
    array_Ylens_er = np.zeros((nScales,100))
    array_Nlens_ev = np.zeros((nScales,100))
    array_Nlens_er = np.zeros((nScales,100))
    for q in range(nScales):
        file_name_Ylens_ev = ('MC/' +(str(int(scale_array[q])) + '/MC_Ylens_ev_'
                            + str(int(scale_array[q])) + '.txt'))
        file_name_Ylens_er = ('MC/' +(str(int(scale_array[q])) + '/MC_Ylens_er_'
                            + str(int(scale_array[q])) + '.txt'))
        file_name_Nlens_ev = ('MC/' +(str(int(scale_array[q])) + '/MC_Nlens_ev_'
                            + str(int(scale_array[q])) + '.txt'))
        file_name_Nlens_er = ('MC/' +(str(int(scale_array[q])) + '/MC_Nlens_er_'
                            + str(int(scale_array[q])) + '.txt'))

        with open(file_name_Ylens_ev, 'r') as f:
            count = 0
            for line in f:
                array_Ylens_ev[q,count] = float(line)
                count += 1
        with open(file_name_Ylens_er, 'r') as f:
            count = 0
            for line in f:
                array_Ylens_er[q,count] = float(line)
                count += 1
        with open(file_name_Nlens_ev, 'r') as f:
            count = 0
            for line in f:
                array_Nlens_ev[q,count] = float(line)
                count += 1
        with open(file_name_Nlens_er, 'r') as f:
            count = 0
            for line in f:
                array_Nlens_er[q,count] = float(line)
                count += 1

    Z = array_Ylens_ev - array_Nlens_ev
    # for i in range(nScales):
        # print(np.sort(Z[i,:].astype('float')))
        # break
        # np.quantile(Z[i,:], 0.5)
        # plt.scatter(scale_array[i]*np.ones(100), Z[i,:])
        # break
        # CI50BayesFactor = pymc3.stats.hpd(Z[i,:], alpha = 0.499)
        # CI90BayesFactor = pymc3.stats.hpd(Z[i,:], alpha = 0.30)
        # CI99BayesFactor = pymc3.stats.hpd(Z[i,:], alpha = 0.20)
        # plt.scatter(scale_array[i], CI50BayesFactor[0], c = 'k')
        # plt.scatter(scale_array[i], CI50BayesFactor[1], c = 'k', alpha = 0.5)
        # plt.scatter(scale_array[i], CI90BayesFactor[0], c = 'b')
        # plt.scatter(scale_array[i], CI90BayesFactor[1], c = 'b')
        # plt.scatter(scale_array[i], CI99BayesFactor[0], c = 'r')
        # plt.scatter(scale_array[i], CI99BayesFactor[1], c = 'r')

    # bins, edges = np.histogram(Z[0,:], bins = [-300, -200, -100, -50, 0, 50, 100, 200, 300])
    # plt.hist(Z[0,:], bins =  np.arange(40) - 10 )
    # plt.plot(edges[0:-1], bins, drawstyle = 'steps-mid')
    # print(bins, edges)
    # # plt.hist(np.mean(Z[5,:]), bins = 10, log = True)
    plt.xscale('log')
    for i in range(14):
        plt.scatter(scale_array[i]*np.ones(100),Z[i,:])
    plt.savefig('plotplot.pdf')

if __name__ == '__main__':
    run_analysis()
