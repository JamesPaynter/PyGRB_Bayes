import sys, os

import argparse
from PyGRB_Bayes.DynamicBilby import BilbyObject
from PyGRB_Bayes.DynamicBilby import create_model_dict
from PyGRB_Bayes.backend.makemodels import create_model_from_key


def load_3770(sampler = 'dynesty', nSamples = 100):
    bilby_inst = BilbyObject(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return bilby_inst

def load_999(sampler = 'dynesty', nSamples = 100):
    object = BilbyObject(999, times = (3, 8),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return object

def load_2571(sampler = 'dynesty', nSamples = 250):
    test = BilbyObject(2571, times = (-2, 40),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 30,
                priors_td_lo = 0,  priors_td_hi = 15)
    return test

def load_973(sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(973, times = (-2, 50),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)
    return test

def load_8099(sampler = 'dynesty', nSamples = 200):
    test = BilbyObject(8099, times = (2, 15),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return test

def load_3770_a(times, sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(3770, times,
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.2,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return test

def load_3770_b(times, sampler = 'dynesty', nSamples = 100):
    test = BilbyObject(3770, times = (0.2, 0.7),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = .2, priors_pulse_end = 0.7,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return test





def analysis_for_3770():
    GRB = load_3770_a(times=(-0.2, 0.2), sampler=SAMPLER, nSamples=5000)
    GRB.offsets = [0, 4000, 8000, -3000]
    # GRB.make_singular_models()
    # model = GRB.create_model_from_key('FFFFFFFX')

    # GRB.main_1_channel(2, model)

    # for key, model in GRB.models.items():
    #     GRB.get_residuals(channels = [0, 1, 2, 3], model = model)
    model = GRB.create_model_from_key('X')
    GRB.get_residuals(channels = [0, 1, 2, 3], model = model)
    # GRB.get_evidence_singular()

def analysis_for_8099(indices):
    GRB = load_8099(sampler = SAMPLER, nSamples = 500)
    GRB.make_singular_models()
    GRB.test_pulse_type(indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
    parser.add_argument('--HPC', action = 'store_true',
                        help = 'Are you running this on SPARTAN ?')
    parser.add_argument('-i', '--indices', type=int, nargs='+',
                        help='an integer for indexing geomspace array')
    args = parser.parse_args()
    HPC = args.HPC


    if not HPC:
        from matplotlib import rc
        rc('font', **{'family': 'DejaVu Sans',
                    'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        rc('text.latex',
        preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}')
        SAMPLER = 'Nestle'

        # analysis_for_8099()
        GRB = load_8099(sampler = SAMPLER, nSamples = 500)

    else:
        SAMPLER = 'dynesty'
        analysis_for_8099(args.indices)


        GRB = load_3770_a(times=(-0.1, 0.2), sampler=SAMPLER, nSamples=5000)
        GRB.test_pulse_type(args.indices)

        GRB = load_3770_b(times=(0.2, 0.7), sampler=SAMPLER, nSamples=5001)



# GRB = load_3770(sampler = SAMPLER, nSamples = 1000)
# model = create_model_dict(lens = False, count_FRED  = [],
#                                         count_FREDx = [1, 2],
#                                         count_sg    = [],
#                                         count_bes   = [])
# GRB.main_multi_channel(channels = [0, 1, 2, 3], model = model)



# GRB.main_1_channel(2, model)
# GRB.array_job(args.indices)
