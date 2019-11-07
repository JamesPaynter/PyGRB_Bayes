import os
import argparse
import numpy as np
from DynamicBilby import BilbyObject

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
    bin_size = 0.064
    nTrials = 100
    nScales = 20
    scale_array = np.geomspace(1e4, 1e7, nScales)
    for q in index_array:
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
            with open(file_name_Ylens_er, 'a+') as f:
                f.write(str(errors_1_lens[0]))
            with open(file_name_Nlens_ev, 'a+') as f:
                f.write(str(evidences_2_FRED[0]))
            with open(file_name_Nlens_er, 'a+') as f:
                f.write(str(errors_2_FRED[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Access geomspace array')

    parser.add_argument('indices', metavar='N', type=int, nargs='+',
                        help='an integer for indexing geomspace array')

    args = parser.parse_args()
    print(args.indices)
    MC_test(index_array = args.indices, nSamples = 100, sampler = 'Nestle')
