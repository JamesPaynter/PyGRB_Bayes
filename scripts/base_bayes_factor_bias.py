from PyGRB.main.fitpulse import PulseFitter
from PyGRB.backend.makemodels import create_model_dict
from PyGRB.backend.makemodels import create_model_from_key
from PyGRB.preprocess.simulated_grb import GRB_discsc, GRB_tte


def make_models():
    keys = ['FF', 'XX', 'FL', 'XL']
    model_dict = {}
    for key in keys:
        model_dict[key] = create_model_from_key(key)
    return model_dict

def test_analysis():
    samples = [200, 500, 1000]
    for nSamples in samples:
        discsc = PulseFitter(0, times = (0, 6.4),
                datatype = 'discsc', nSamples = nSamples, sampler = 'nestle',
                priors_pulse_start = 0, priors_pulse_end = 6.4,
                GRB = GRB_discsc, test = True,
                priors_td_lo = 0,  priors_td_hi = 1)
        discsc.models = make_models()
        tte    = PulseFitter(0, times = (0, 0.5),
                datatype = 'tte', nSamples = nSamples, sampler = 'nestle',
                priors_pulse_start = 0, priors_pulse_end = 0.5,
                GRB = GRB_tte, test = True,
                priors_td_lo = 0,  priors_td_hi = 1)
        tte.models = make_models()
        models = [model for key, model in discsc.models.items()]
        for model in models:
            discsc.main_1_channel(channel = 0, model = model)
            tte.main_1_channel(channel = 0, model = model)
        discsc.get_evidence_from_models(discsc.models)
        tte.get_evidence_from_models(tte.models)

if __name__ == '__main__':
    test_analysis()
