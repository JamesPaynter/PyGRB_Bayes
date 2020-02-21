import sys, os

import argparse
from PyGRB_Bayes.DynamicBilby import BilbyObject
from PyGRB_Bayes.DynamicBilby import create_model_dict

import bilby
logger = bilby.core.utils.logger
logger.disabled = True

def load_8099(sampler = 'dynesty', nSamples = 200):
    test = BilbyObject(8099, times = (2, 15),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return test

def analysis_for_8099(indices):
    GRB = load_8099(sampler = SAMPLER, nSamples = 500)
    GRB.make_singular_models()
    GRB.test_pulse_type(indices)

def evidence_for_8099():
    GRB = load_8099(sampler = SAMPLER, nSamples = 500)
    # keys = ['FF', 'FL', 'FsFs', 'FsL', 'XX', 'XL', 'XsXs', 'XsL']
    # model_dict = {}
    # for key in keys:
    #     model_dict[key] = GRB.create_model_from_key(key)
    GRB.make_singular_models()
    models = [model for key, model in GRB.models.items()]
    for model in models:
        try:
            GRB.get_residuals(channels = [0, 1, 2, 3], model = model)
        except:
            pass
    GRB.get_evidence_singular()

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
        evidence_for_8099()
    else:
        SAMPLER = 'dynesty'
        analysis_for_8099(args.indices)
