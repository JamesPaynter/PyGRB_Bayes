from DynamicBilby import BilbyObject

def load_3770(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return test

def load_3770_custom(times, sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(3770, times = times,
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return test

def load_973(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(973, times = (-2, 50),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    return test

def load_2571(sampler = 'dynesty', nSamples = 2500):
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

def load_test(sampler = 'dynesty', nSamples = 100):
    # object = BilbyObject(1, times = (-0.1, 1.0), test = True,
    #             datatype = 'tte', nSamples = nSamples, sampler = sampler,
    #             priors_pulse_start = -0.1, priors_pulse_end = 0.6)
    # object.inject_signal()
    # print(object.GRB.bin_left)
    # return object
    test = BilbyObject(trigger = 1, times = (-2, 50), test = True,
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    test.inject_signal()
    print(test.GRB.bin_left)
    return test



def compare_FRED_FREDx( function, channels = [0,1,2,3],
                        sampler = 'Nestle', **kwargs):
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

def compare_num_FRED_pulses(function, channels = [0,1,2,3],
                            sampler = 'Nestle', **kwargs):
    GRB = function(sampler, **kwargs)
    evidences, errors = np.zeros(4), np.zeros(4)
    evidences[0], errors[0] = GRB.one_FRED(  channels = channels, test = False)
    evidences[1], errors[1] = GRB.two_FRED(  channels = channels, test = False)
    evidences[2], errors[2] = GRB.three_FRED(channels = channels, test = False)
    evidences[3], errors[3] = GRB.four_FRED( channels = channels, test = False)
    for i in range(4):
        print('---------------')
        print('For channel {}'.format(i+1))
        print('The one FRED evidence is : {0:.3f} +/- {1:.3f}'.format(
                evidences[0], errors[0]))
        print('The one FRED evidence is : {0:.3f} +/- {1:.3f}'.format(
                evidences[1], errors[1]))
        print('The one FRED evidence is : {0:.3f} +/- {1:.3f}'.format(
                evidences[2], errors[2]))
        print('The four FRED evidence is : {0:.3f} +/- {1:.3f}'.format(
                evidences[3], errors[3]))
        num_pulse = np.argmax(evidences) + 1
        print('The preferred number of pulses is %i' % num_pulse)


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

def compare_lens_no_lens_two_pulse(function, channels = [0,1,2,3],
                                    sampler = 'Nestle', **kwargs):

        GRB = function(sampler, **kwargs)
        evidences_1_lens, errors_1_lens = GRB.two_FRED_lens(
                                            channels = channels,
                                            test = False, plot = True)
        evidences_2_FRED, errors_2_FRED = GRB.four_FRED(
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
            print('Log Bayes Factor: ', BF)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
    parser.add_argument('--HPC', action = 'store_true',
                        help = 'Are you running this on SPARTAN ?')
    args = parser.parse_args()
    HPC = args.HPC

    if not HPC:
        rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        SAMPLER = 'Nestle'
    else:
        SAMPLER = 'dynesty'

    # compare_num_FRED_pulses(load_3770_custom, times = (-0.1, 0.35))

    compare_lens_no_lens_one_pulse(load_2571, sampler = SAMPLER)

    # compare_FRED_FREDx(load_8099, nSamples = 500, sampler = SAMPLER)
    # compare_FRED_FREDx(load_8099, nSamples = 500)
    # compare_lens_no_lens_two_pulse(load_3770, nSamples = 500, sampler = 'dynesty')
