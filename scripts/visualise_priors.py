from PyGRB.main.fitpulse import PulseFitter
from PyGRB.backend.makemodels import create_model_dict
from PyGRB.backend.makemodels import create_model_from_key
from PyGRB.preprocess.simulated_grb import GRB_discsc, GRB_tte




def make_models():
    keys = ['X', 'Fs', 'Xs'] # ,'G', 'C']
    model_dict = {}
    for key in keys:
        model_dict[key] = create_model_from_key(key)
    return model_dict


def test_analysis():
    nSamples = 100
    start, end  = 0.0, 4.0
    discsc = PulseFitter(0, times = (start, end),
            datatype = 'discsc', nSamples = nSamples, sampler = 'nestle',
            priors_pulse_start = start, priors_pulse_end = end,
            GRB = GRB_discsc, test = True,
            priors_td_lo = start,  priors_td_hi = end)
    discsc.models = make_models()
    #
    start, end  = 0.0, 0.3
    tte    = PulseFitter(0, times = (start, end),
            datatype = 'tte', nSamples = nSamples, sampler = 'nestle',
            priors_pulse_start = start, priors_pulse_end = end,
            GRB = GRB_tte, test = True,
            priors_td_lo = start,  priors_td_hi = end)
    tte.models = make_models()
    #
    models = [model for key, model in discsc.models.items()]
    for model in models:
        discsc._sample_priors(channel = 0, model = model)
        break
        # tte._sample_priors(   channel = 0, model = model)

if __name__ == '__main__':
    test_analysis()
