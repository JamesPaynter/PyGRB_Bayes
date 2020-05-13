from PyGRB.main.fitpulse import PulseFitter



class AnimateConvergence(PulseFitter):
    """docstring for AnimateConvergence."""

    def __init__(self):
        super(AnimateConvergence, self).__init__()



    def checkpoint_visual(self, channel, model):
        self._setup_labels(model)
        ii = 0
        i = channel
        # while True:
        for ii in range(100):
            print(ii)
            open_result = f'{self.outdir}/checkpoint_{ii}.json'
            result = bilby.result.read_in_result(filename=open_result)
            x = result.nested_samples
            # result.posterior = x[min(5000, int(len(x)/2)):]
            result.posterior = x[int(3*len(x)/4):]
            plotname = f'{self.outdir}/checkpoint_{ii}_corner.png'

            result.plot_corner(filename = plotname)
            ii += 1
            # except:
            #     pass
            self._setup_labels(model)
            strings = { 'fstring' : f'{self.fstring}_{ii}',
                        'clabels' : self.clabels,
                        'outdir'  : self.outdir}
            prior_shell = MakePriors(
                            priors_pulse_start = self.priors_pulse_start,
                            priors_pulse_end = self.priors_pulse_end,
                            priors_td_lo = self.priors_td_lo,
                            priors_td_hi = self.priors_td_hi,
                            channel      = i,
                            **self.model)
            priors = prior_shell.return_prior_dict()
            x = self.GRB.bin_left
            y = np.rint(self.GRB.counts[:,i]).astype('uint')
            likelihood = PoissonRate(x, y, i, **self.model)
            MAP = dict()
            posteriors = dict()
            c_keys = ['a', 'b', 'c', 'd']
            k      = c_keys[i]
            for j in range(1, self.num_pulses + 1):
                try:
                    key = f'constraint_{j}_{k}'
                    del priors[key]
                except:
                    pass
            for parameter in priors:
                posteriors[parameter] = result.posterior[parameter].values
                summary = result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP[parameter] = summary.median
            if model['lens']:
                counts_fit = likelihood._sum_rates(x, MAP,
                                    likelihood.calculate_rate_lens)
            else:
                counts_fit = likelihood._sum_rates(x, MAP,
                                                likelihood.calculate_rate)

            widths = self.GRB.bin_right - self.GRB.bin_left
            rates_i= self.GRB.counts[:,i] / widths
            rates_fit_i = counts_fit / widths
            rates_err_i = np.sqrt(self.GRB.counts[:,i]) / widths
            strings['widths'] = widths
            # posterior_draws   = posterior_draws / widths[:,None]
            # posterior_lines[:,:,i] = posterior_draws
            PlotPulseFit(   x = self.GRB.bin_left, y = rates_i,
                            y_err = rates_err_i,
                            y_cols = self.GRB.colours[i],
                            y_fit = rates_fit_i,
                            channels = [i],
                            datatype = self.datatype,
                            # posterior_draws = posterior_draws,
                            # nDraws = nDraws,
                            **strings)

if __name__ == '__main__':
    pass
