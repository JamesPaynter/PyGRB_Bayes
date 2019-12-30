import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from DynamicBilby import BilbyObject
import pymc3
from matplotlib import rc

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


def run_analysis():
    nScales = 20
    scale_array = np.geomspace(1e2, 1e5, nScales)
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

    fig, ax = plt.subplots(figsize = (3.321, 2))
    Z = array_Ylens_ev - array_Nlens_ev
    CI00BayesFactor = np.zeros(20)
    CI50BayesFactor = np.zeros((2,20))
    CI90BayesFactor = np.zeros((2,20))
    CI99BayesFactor = np.zeros((2,20))
    for i in range(nScales):
        CI00BayesFactor[i]   = np.mean(Z[i,:])
        CI50BayesFactor[:,i] = pymc3.stats.hpd(Z[i,:], alpha = 0.50)
        CI90BayesFactor[:,i] = pymc3.stats.hpd(Z[i,:], alpha = 0.10)
        CI99BayesFactor[:,i] = pymc3.stats.hpd(Z[i,:], alpha = 0.01)
        # print(np.sort(Z[i,:]))
        # ax.scatter(scale_array[i]*np.ones(100), Z[i,:], s = 0.1)
    ax.plot(scale_array, CI00BayesFactor, 'k:', linewidth = 0.5)
    ax.fill_between(scale_array, y1 = CI50BayesFactor[0],
                    y2 = CI50BayesFactor[1], alpha = 0.4,
                    color = 'cornflowerblue')
    ax.fill_between(scale_array, y1 = CI90BayesFactor[0],
                    y2 = CI90BayesFactor[1], alpha = 0.2,
                    color = 'cornflowerblue')
    ax.fill_between(scale_array, y1 = CI99BayesFactor[0],
                    y2 = CI99BayesFactor[1], alpha = 0.1,
                    color = 'cornflowerblue')
    ax.set_xscale('log')
    ax.set_xlabel('Scale Parameter, $A$')
    ax.set_ylabel('log BF $=$ log $\\mathcal{Z}_{\\mathfrak{L}} - $log $\\mathcal{Z}_{\\O}$')

    plt.subplots_adjust(left=0.16)
    plt.subplots_adjust(right=0.98)
    plt.subplots_adjust(top=0.98)
    plt.subplots_adjust(bottom=0.20)
    fig.savefig('MC_Figure.pdf')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
    parser.add_argument('--HPC', action = 'store_true',
                        help = 'Are you running this on SPARTAN ?')
    parser.add_argument('-indices', metavar='--N', type=int, nargs='+',
                        help='an integer for indexing geomspace array')
    args = parser.parse_args()
    HPC = args.HPC

    if not HPC:
        rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}')
        SAMPLER = 'Nestle'
    else:
        SAMPLER = 'dynesty'
    # MC_test(args.indices, sampler = SAMPLER, nSamples = 250)
    run_analysis()
