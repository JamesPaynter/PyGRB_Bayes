import sys, os

import argparse
from PyGRB_Bayes.DynamicBilby import BilbyObject
from PyGRB_Bayes.DynamicBilby import create_model_dict

# import bilby
# logger = bilby.core.utils.logger
# logger.disabled = True

def load_3770(sampler = 'dynesty', nSamples = 100):
    bilby_inst = BilbyObject(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return bilby_inst


def analysis_for_3770(indices):
    GRB = load_3770(sampler=SAMPLER, nSamples=500)
    GRB.offsets = [0, 4000, 8000, -3000]
    GRB.test_two_pulse_models(indices)

    GRB = load_3770(sampler=SAMPLER, nSamples=2000)
    GRB.offsets = [0, 4000, 8000, -3000]
    GRB.test_two_pulse_models(indices)

    GRB = load_3770(sampler=SAMPLER, nSamples=4500)
    GRB.offsets = [0, 4000, 8000, -3000]
    GRB.test_two_pulse_models(indices)

def evidence_for_3770():
    num_samples = [500, 2000, 4500]
    for samples in num_samples:
        GRB = load_3770(sampler=SAMPLER, nSamples=samples)
        GRB.offsets = [0, 4000, 8000, -3000]
        keys = ['FF', 'FL', 'FbFb', 'FbL', 'XX', 'XL', 'XbXb', 'XbL']
        # keys = ['FF', 'FL', 'FsFs', 'FsL', 'XX', 'XL', 'XsXs', 'XsL']
        model_dict = {}
        for key in keys:
            model_dict[key] = GRB.create_model_from_key(key)
        models = [model for key, model in model_dict.items()]

        # for model in models:
            # try:
            #     GRB.get_residuals(channels = [0, 1, 2, 3], model = model)
            # except:
            #     pass
        GRB.get_evidence_singular_lens()

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
        evidence_for_3770()


    else:
        SAMPLER = 'dynesty'
        analysis_for_3770(args.indices)
