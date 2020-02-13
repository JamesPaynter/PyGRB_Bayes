import sys, os
# path = os.path.dirname(__file__)
# path = os.path.join(path, 'bin')
# if path not in sys.path:
#     sys.path.append(path)

import argparse
from .. import core

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
    object = BilbyObject(8099, times = (2, 15),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return object

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


def create_model_dict(  lens, count_FRED, count_FREDx, count_sg, count_bes,
                        **kwargs):
    model = {}
    model['lens']        = lens
    model['count_FRED']  = count_FRED
    model['count_FREDx'] = count_FREDx
    model['count_sg']    = count_sg
    model['count_bes']   = count_bes
    if kwargs:
        for kwarg in kwargs:
            model[kwarg] = kwargs[kwarg]
    return model



parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
parser.add_argument('--HPC', action = 'store_true',
                    help = 'Are you running this on SPARTAN ?')
parser.add_argument('-i', '--indices', type=int, nargs='+',
                    help='an integer for indexing geomspace array')
args = parser.parse_args()
HPC = args.HPC

print(args)

if not HPC:
    from matplotlib import rc
    rc('font', **{'family': 'DejaVu Sans',
                'serif': ['Computer Modern'],'size': 8})
    rc('text', usetex=True)
    rc('text.latex',
    preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}')
    SAMPLER = 'Nestle'
else:
    SAMPLER = 'dynesty'

# GRB = load_8099(sampler = SAMPLER, nSamples = 51)
# GRB.make_singular_models()
# GRB.main_1_channel(channel = 1, model = GRB.model_Xs)
# GRB = load_3770(sampler = SAMPLER, nSamples = 1000)
# model = create_model_dict(lens = False, count_FRED  = [],
#                                         count_FREDx = [1, 2],
#                                         count_sg    = [],
#                                         count_bes   = [])
# GRB.main_multi_channel(channels = [0, 1, 2, 3], model = model)

GRB = load_3770_a(times = (-0.1, 0.2), sampler = SAMPLER, nSamples = 2000)
GRB.test_pulse_type(args.indices)
GRB.get_evidence_singular()

# GRB.main_1_channel(2, model)
# GRB.array_job(args.indices)
