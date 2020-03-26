import sys, os
import argparse
import numpy as np

from bilby.core.prior import Uniform          as bilbyUniform
from bilby.core.prior import DeltaFunction    as bilbyDeltaFunction
from bilby.core.prior import Gaussian         as bilbyGaussian

from PyGRB_Bayes.main.fitpulse import PulseFitter
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_two_pulse_models


# np.seterr(divide='ignore', invalid='ignore')



def analysis_for_3770(indices):
    nSamples = 2000
    keys = ['XsL', 'XsXs']
    channels = [0, 1, 2, 3]
    n_per_split = len(keys) * len(channels)

    prior_sets = [{ 'priors_gamma_min': 1e-1, ## generic
                    'priors_gamma_max': 1e1,
                    'priors_nu_min'   : 1e-1,
                    'priors_nu_max'   : 1e1},
                {   'priors_gamma_min': 1e-2,
                    'priors_gamma_max': 1e2,
                    'priors_nu_min'   : 1e-2,
                    'priors_nu_max'   : 1e2},
                {   'priors_gamma_min': 1e-3,
                    'priors_gamma_max': 1e3,
                    'priors_nu_min'   : 1e-3,
                    'priors_nu_max'   : 1e3}
                    ]

    directory_labels = ['small_box_log_flat', 'mid_box_log_flat', 'large_box_log_flat']
    for ii, prior_set in enumerate(prior_sets):
        GRB = PulseFitter(3770, times = (-.1, 1),
                    datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                    priors_pulse_start = -.1, priors_pulse_end = 0.6,
                    priors_td_lo = 0,  priors_td_hi = 0.5,
                    directory_label = directory_labels[ii],
                    **prior_set)

        GRB.offsets = [0, 4000, 8000, -3000]
        ## need to break up into indices (belowwww)

        model_dict = {}
        for key in keys:
            model_dict[key] = create_model_from_key(key,
                        custom_name = f'{key}_{directory_labels[ii]}')
        models = [model for key, model in model_dict.items()]
        indx = np.intersect1d(indices,
            np.arange(n_per_split * ii, n_per_split * (ii + 1))) % n_per_split
        GRB._split_array_job_to_4_channels(models = models,
            indices = indx,
            channels = channels)
#################################

    directory_labels = ['small_box_flat', 'mid_box_flat', 'large_box_flat']
    for ii, prior_set in enumerate(prior_sets):
        GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                    datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                    priors_pulse_start = -.1, priors_pulse_end = 0.6,
                    priors_td_lo = 0,  priors_td_hi = 0.5,
                    directory_label = directory_labels[ii])
        GRB_wrap.offsets = [0, 4000, 8000, -3000]
        # GRB.test_two_pulse_models(indices)
        ## need to break up into indices (belowwww)
        model_dict = {}
        for key in keys:
            model_dict[key] = create_model_from_key(key,
                        custom_name = f'{key}_{directory_labels[ii]}')
        models = [model for key, model in model_dict.items()]
        for mm, model in enumerate(models):
            GRB_wrap._setup_labels(model)
            overwrite_priors = dict()
            for n in range(1, GRB_wrap.num_pulses + 1):
                for k in ['a', 'b', 'c', 'd']:
                    overwrite_priors[f'gamma_{n}_{k}'] = bilbyUniform(
                    minimum=prior_set['priors_gamma_min'],
                    maximum=prior_set['priors_gamma_max'],
                    latex_label=f'$\\gamma_{n} {k}$', unit=' ')
                    overwrite_priors[f'nu_{n}_{k}'] = bilbyUniform(
                    minimum=prior_set['priors_nu_min'],
                    maximum=prior_set['priors_nu_max'],
                    latex_label=f'$\\nu_{n} {k}$', unit=' ')
            GRB_wrap.overwrite_priors = overwrite_priors

            indx = np.intersect1d(indices,
                np.arange(  24 + 4 * ( mm + ii),
                            24 + 4 * ((mm + ii) + 1))) % n_per_split
            GRB._split_array_job_to_4_channels(models = [model],
                indices = indx, channels = channels)

    # should be at index 48 by now
    directory_label = 'delta'
    GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5,
                directory_label = directory_label)
    GRB_wrap.offsets = [0, 4000, 8000, -3000]
    ## need to break up into indices (belowwww)
    model_dict = {}
    for key in keys:
        model_dict[key] = create_model_from_key(key,
                    custom_name = f'{key}_{directory_label}')
    models = [model for key, model in model_dict.items()]
    for model in models:
        GRB_wrap._setup_labels(model)
        overwrite_priors = dict()
        for n in range(1, GRB_wrap.num_pulses + 1):
            for k in ['a', 'b', 'c', 'd']:
                overwrite_priors[f'gamma_{n}_{k}'] = bilbyDeltaFunction(1,  latex_label = f'$\\gamma$ {n} {k}')
                overwrite_priors[f'nu_{n}_{k}'] = bilbyDeltaFunction(1,     latex_label = f'$\\nu$ {n} {k}')
        GRB_wrap.overwrite_priors = overwrite_priors

        indx = np.intersect1d(indices,
            np.arange(  48 + 4 *  mm,
                        48 + 4 * (mm + 1))) % n_per_split
        GRB._split_array_job_to_4_channels(models = [model],
            indices = indx, channels = channels)

    directory_label = 'gaussian'
    GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5,
                directory_label = directory_label)
    GRB_wrap.offsets = [0, 4000, 8000, -3000]
    ## need to break up into indices (belowwww)
    model_dict = {}
    for key in keys:
        model_dict[key] = create_model_from_key(key,
                    custom_name = f'{key}_{directory_label}')
    models = [model for key, model in model_dict.items()]
    for model in models:
        GRB_wrap._setup_labels(model)
        overwrite_priors = dict()
        for n in range(1, GRB_wrap.num_pulses + 1):
            overwrite_priors[f'gamma_{n}_a'] = bilbyGaussian(
                mu = 0.7, sigma = 2.5,  latex_label = f'$\\gamma$ {n} a')
            overwrite_priors[f'gamma_{n}_b'] = bilbyGaussian(
                mu = 0.3, sigma = 0.4,  latex_label = f'$\\gamma$ {n} b')
            overwrite_priors[f'gamma_{n}_c'] = bilbyGaussian(
                mu = 0.38, sigma = 0.3, latex_label = f'$\\gamma$ {n} c')
            overwrite_priors[f'gamma_{n}_d'] = bilbyGaussian(
                mu = 0.5, sigma = 5,    latex_label = f'$\\gamma$ {n} d')
            overwrite_priors[f'nu_{n}_a'] = bilbyGaussian(
                mu = 2, sigma = 2,      latex_label = f'$\\nu$ {n} a')
            overwrite_priors[f'nu_{n}_b'] = bilbyGaussian(
                mu = 3.3, sigma = 1.2,  latex_label = f'$\\nu$ {n} b')
            overwrite_priors[f'nu_{n}_c'] = bilbyGaussian(
                mu = 2.74, sigma = 0.8, latex_label = f'$\\nu$ {n} c')
            overwrite_priors[f'nu_{n}_d'] = bilbyGaussian(
                mu = 2.7, sigma = 5,    latex_label = f'$\\nu$ {n} d')
        GRB_wrap.overwrite_priors = overwrite_priors

        indx = np.intersect1d(indices,
            np.arange(56 + 4 * mm, 56 + 4 * (mm + 1))) % n_per_split
        GRB._split_array_job_to_4_channels(models = [model],
            indices = indx, channels = [0, 1, 2, 3])
# at index 64 by now.... DONE


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
        SAMPLER = 'nestle'
        analysis_for_3770([14])


    else:
        SAMPLER = 'dynesty'
        analysis_for_3770(args.indices)
