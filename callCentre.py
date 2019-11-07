import os
import numpy as np
import matplotlib.pyplot as plt
from DynamicBilby import BilbyObject

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

def two_pulse_constraints(parameters):
    parameters['constraint_2'] = parameters['start_2'] - parameters['start_1']
    return parameters


def load_test(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(trigger = 1, times = (-2, 50), test = True,
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    test.inject_signal()
    return test

def load_3770(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 1.0,
                priors_td_lo = 0,  priors_td_hi = 1.0)
    return test

def load_973(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(973, times = (-2, 50),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    return test

def load_2571(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(2571, times = (-2, 40),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 30,
                priors_td_lo = 0,  priors_td_hi = 15)
    return test

def load_469(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(469, times = (-2, 12),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 12)
    return object

def load_999(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(999, times = (3, 7),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 1, priors_pulse_end = 7)
    return object

def load_6621(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(6621, times = (30, 45),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 25, priors_pulse_end = 45)
    return object

def load_6630(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(6630, times = (-2, 25),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 25)
    return object

def load_6814(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(6814, times = (-2, 10),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 10)
    return object

def load_7475(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(7475, times = (-2, 60),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 60)
    return object

def load_7965(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(7965, times = (-2, 10),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 10)
    return object

def load_8099(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(8099, times = (2, 15),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return object


def compare_FRED_FREDx(function, channels = [0,1,2,3], sampler = 'Nestle', **kwargs):
    GRB = function(sampler, **kwargs)
    evidences_1_FRED,  errors_1_FRED  = GRB.one_FRED( channels = channels, test = False)
    evidences_1_FREDx, errors_1_FREDx = GRB.one_FREDx(channels = channels, test = False)
    for i in range(4):
        print('---------------')
        print('For channel {}'.format(i+1))
        print('The FRED evidence is : {0:.3f} +/- {1:.3f}'.format(
                evidences_1_FRED[i], errors_1_FRED[i]))
        print('The FRED-X evidence is : {0:.3f} +/- {1:.3f}'.format(
                evidences_1_FREDx[i], errors_1_FREDx[i]))
        BF = evidences_1_FRED[i] - evidences_1_FREDx[i]
        if evidences_1_FRED[i] > evidences_1_FREDx[i]:
            print('The winner is FRED')
        else:
            print('The winner is FRED-X')
        print('Bayes Factor: ', BF)

def compare_lens_no_lens_one_pulse(function, channels = [0,1,2,3],
                                    sampler = 'Nestle', **kwargs):

        GRB = function(sampler, **kwargs)
        evidences_1_lens, errors_1_lens = GRB.one_FRED_lens(
                                            channels = channels,
                                            test = False, plot = True)
        evidences_2_FRED, errors_2_FRED = GRB.two_FRED(
                                            channels = channels,
                                            test = False, plot = True)
        for i in channels:
            print('---------------')
            print('For channel {}'.format(i+1))
            print('The FRED evidence is : {0:.3f} +/- {1:.3f}'.format(
                    evidences_2_FRED[i], errors_2_FRED[i]))
            print('The lensing evidence is : {0:.3f} +/- {1:.3f}'.format(
                    evidences_1_lens[i], errors_1_lens[i]))
            BF = evidences_1_lens[i] - evidences_2_FRED[i]
            if evidences_2_FRED[i] > evidences_1_lens[i]:
                print('The winner is FRED')
            else:
                print('The winner is lensing')
            print('Bayes Factor: ', BF)

def MC_test(**kwargs):
    channels = [0]
    GRB = load_test(**kwargs)
    GRB.counter = 0
    plt.close('all')
    bin_size = 0.064
    nTrials = 100
    nScales = 10
    scale_array = np.geomspace(1e4, 1e8, nScales)
    for q in range(len(scale_array)):
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
    MC_test(nSamples = 100, sampler = 'Nestle')
    # compare_lens_no_lens_one_pulse(load_test, channels = [0], nSamples = 200)

    # compare_FRED_FREDx(load_8099, nSamples = 500)
    # compare_lens_no_lens_one_pulse(load_3770, nSamples = 500)
