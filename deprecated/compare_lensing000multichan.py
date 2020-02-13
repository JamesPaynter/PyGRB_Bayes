import bilby
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.special          import gammaln
from bilby.core.prior       import PriorDict
from bilby.core.prior       import Uniform
from bilby.core.prior       import Constraint
from bilby.core.prior       import LogUniform
from bilby.core.likelihood  import PoissonLikelihood
from bilby.core.likelihood  import JointLikelihood

from scipy.signal import savgol_filter

import BATSEpreprocess

import sys
sys.path.append("C:\\Users\\James\\Documents\\University\\Projects\\PyGRB_lens\\PyGRB_lens\\PyGRB_Bayes")

SAMPLES = 100
outdir = 'model_comparison_' + str(SAMPLES) + '/lens_model'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

tau_rise_min, tau_rise_max  = 1e-2, 1e2
tau_fall_min, tau_fall_max  = 1e-2, 1e2
scale_min, scale_max        = 1e3, 1e7
start, end                  = -50, 150

one_pulse_lens_priors    = PriorDict()

one_pulse_lens_priors['background_a']  = LogUniform(   1e-1, 1e4, latex_label='$c$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['background_b']  = LogUniform(   1e-1, 1e4, latex_label='$c$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['background_c']  = LogUniform(   1e-1, 1e4, latex_label='$c$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['background_d']  = LogUniform(   1e-1, 1e4, latex_label='$c$',
                                                unit = 'counts / sec')


one_pulse_lens_priors['time_delay']  = Uniform(  0, 30,
                                            latex_label='$\\Delta t$',
                                            unit = 'sec')
one_pulse_lens_priors['magnification_ratio']  = Uniform(  0.2, 1,
                                            latex_label='$\\mu$',
                                            unit = ' ')
one_pulse_lens_priors['start1_a']      = Uniform(  start, end,
                                            latex_label='$\\Delta_1$',
                                            unit = 'sec')
one_pulse_lens_priors['scale1_a']      = LogUniform(   scale_min, scale_max,
                                                latex_label='$A_1$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['rise1_a'] = LogUniform( minimum = tau_rise_min,
                                        maximum = tau_rise_max,
                                        latex_label='$ \\tau_{r,1} $',
                                        unit = ' ')
one_pulse_lens_priors['decay1_a'] = LogUniform(minimum = tau_fall_min,
                                        maximum = tau_fall_max,
                                        latex_label='$\\tau_{d,1}$',
                                        unit = ' ')
one_pulse_lens_priors['start1_b']      = Uniform(  start, end,
                                            latex_label='$\\Delta_1$',
                                            unit = 'sec')
one_pulse_lens_priors['scale1_b']      = LogUniform(   scale_min, scale_max,
                                                latex_label='$A_1$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['rise1_b'] = LogUniform( minimum = tau_rise_min,
                                        maximum = tau_rise_max,
                                        latex_label='$ \\tau_{r,1} $',
                                        unit = ' ')
one_pulse_lens_priors['decay1_b'] = LogUniform(minimum = tau_fall_min,
                                        maximum = tau_fall_max,
                                        latex_label='$\\tau_{d,1}$',
                                        unit = ' ')
one_pulse_lens_priors['start1_c']      = Uniform(  start, end,
                                            latex_label='$\\Delta_1$',
                                            unit = 'sec')
one_pulse_lens_priors['scale1_c']      = LogUniform(   scale_min, scale_max,
                                                latex_label='$A_1$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['rise1_c'] = LogUniform( minimum = tau_rise_min,
                                        maximum = tau_rise_max,
                                        latex_label='$ \\tau_{r,1} $',
                                        unit = ' ')
one_pulse_lens_priors['decay1_c'] = LogUniform(minimum = tau_fall_min,
                                        maximum = tau_fall_max,
                                        latex_label='$\\tau_{d,1}$',
                                        unit = ' ')
one_pulse_lens_priors['start1_d']      = Uniform(  start, end,
                                            latex_label='$\\Delta_1$',
                                            unit = 'sec')
one_pulse_lens_priors['scale1_d']      = LogUniform(   scale_min, scale_max,
                                                latex_label='$A_1$',
                                                unit = 'counts / sec')
one_pulse_lens_priors['rise1_d'] = LogUniform( minimum = tau_rise_min,
                                        maximum = tau_rise_max,
                                        latex_label='$ \\tau_{r,1} $',
                                        unit = ' ')
one_pulse_lens_priors['decay1_d'] = LogUniform(minimum = tau_fall_min,
                                        maximum = tau_fall_max,
                                        latex_label='$\\tau_{d,1}$',
                                        unit = ' ')


def one_pulse_lens_rate_a(delta_t, background_a, time_delay, magnification_ratio,
                        start1_a, scale1_a, rise1_a, decay1_a):

    times1 = (GRB.bin_left - start1_a) * np.heaviside(GRB.bin_left - start1_a, 0) + 1e-12
    times0 = ((GRB.bin_left - start1_a - time_delay) * np.heaviside(
                                GRB.bin_left - start1_a - time_delay, 0) + 1e-12 )

    rates = (background_a + scale1_a * np.exp(- np.power(( rise1_a / times1), 1)
                                         - np.power((times1 / decay1_a), 1) )
    + magnification_ratio * scale1_a * np.exp(- np.power(( rise1_a / times0), 1)
                                         - np.power((times0 / decay1_a), 1) ))
    return np.multiply(rates, widths)

def one_pulse_lens_rate_b(delta_t, background_b, time_delay, magnification_ratio,
                        start1_b, scale1_b, rise1_b, decay1_b):

    times1 = ( GRB.bin_left - start1_b) * np.heaviside(GRB.bin_left - start1_b, 0) + 1e-12
    times0 = ((GRB.bin_left - start1_b - time_delay) * np.heaviside(
                                GRB.bin_left - start1_b - time_delay, 0) + 1e-12 )

    rates = (background_b + scale1_b * np.exp(- np.power(( rise1_b / times1), 1)
                                         - np.power((times1 / decay1_b), 1) )
    + magnification_ratio * scale1_b * np.exp(- np.power(( rise1_b / times0), 1)
                                         - np.power((times0 / decay1_b), 1) ))
    return np.multiply(rates, widths)

def one_pulse_lens_rate_c(delta_t, background_c, time_delay, magnification_ratio,
                        start1_c, scale1_c, rise1_c, decay1_c):

    times1 = (GRB.bin_left - start1_c) * np.heaviside(GRB.bin_left - start1_c, 0) + 1e-12
    times0 = ((GRB.bin_left - start1_c - time_delay) * np.heaviside(
                                GRB.bin_left - start1_c - time_delay, 0) + 1e-12 )

    rates = (background_c + scale1_c * np.exp(- np.power(( rise1_c / times1), 1)
                                         - np.power((times1 / decay1_c), 1) )
    + magnification_ratio * scale1_c * np.exp(- np.power(( rise1_c / times0), 1)
                                         - np.power((times0 / decay1_c), 1) ))
    return np.multiply(rates, widths)

def one_pulse_lens_rate_d(delta_t, background_d, time_delay, magnification_ratio,
                        start1_d, scale1_d, rise1_d, decay1_d):

    times1 = ( GRB.bin_left - start1_d) * np.heaviside(GRB.bin_left - start1_d, 0) + 1e-12
    times0 = ((GRB.bin_left - start1_d - time_delay) * np.heaviside(
                                GRB.bin_left - start1_d - time_delay, 0) + 1e-12 )

    rates = (background_d + scale1_d * np.exp(- np.power(( rise1_d / times1), 1)
                                         - np.power((times1 / decay1_d), 1) )
    + magnification_ratio * scale1_d * np.exp(- np.power(( rise1_d / times0), 1)
                                         - np.power((times0 / decay1_d), 1) ))
    return np.multiply(rates, widths)

colours = ['red', 'orange', 'green', 'blue']
bursts  = [973]
timeq   = [(-2,50)]


for j in range(len(bursts)):
    print('Analysis for GRB %i' % int(bursts[j]))
    label = 'BATSE_' + str(bursts[j]) + '_'
    (start, end) = timeq[j]

    GRB = BATSEpreprocess.BATSESignal(  bursts[j], times = (start, end),
                                        datatype = 'discsc', bgs = False)

    widths  = GRB.bin_right - GRB.bin_left
    counts  = np.rint(GRB.counts[:,2]).astype('uint')
    delta_t = np.diff(GRB.bin_left)

    heights = [5, 1, 1, 1, 1]

    fig2 = plt.figure(constrained_layout=False)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2, height_ratios=heights,
                            hspace=0.0, wspace=0.0)
    f2_ax1 = fig2.add_subplot(spec2[0, 0])
    f2_ax2 = fig2.add_subplot(spec2[1, 0])
    f2_ax3 = fig2.add_subplot(spec2[2, 0])
    f2_ax4 = fig2.add_subplot(spec2[3, 0])
    f2_ax5 = fig2.add_subplot(spec2[4, 0])

    residual_axes = [f2_ax2, f2_ax3, f2_ax4, f2_ax5]

    priors = one_pulse_lens_priors.copy()
    counts_a = np.rint(GRB.counts[:,0]).astype('uint')
    counts_b = np.rint(GRB.counts[:,1]).astype('uint')
    counts_c = np.rint(GRB.counts[:,2]).astype('uint')
    counts_d = np.rint(GRB.counts[:,3]).astype('uint')
    likelihood = JointLikelihood(
                 PoissonLikelihood(delta_t, counts_a, one_pulse_lens_rate_a),
                 PoissonLikelihood(delta_t, counts_b, one_pulse_lens_rate_b),
                 PoissonLikelihood(delta_t, counts_c, one_pulse_lens_rate_c),
                 PoissonLikelihood(delta_t, counts_d, one_pulse_lens_rate_d))
    labell = label + '___'

    # labell = label + '_' + str(n)
    fileee = outdir + '/' + labell +'_result.json'
    # result = bilby.result.read_in_result(filename=fileee)

    result = bilby.run_sampler( likelihood = likelihood,
                                priors = priors,
                                sampler = 'Nestle', nlive = SAMPLES,
                                outdir = outdir, label = labell)

    plotname = outdir + '/' + labell +'_corner.pdf'
    result.plot_corner(filename = plotname)
    # for i in range(4):
    #
    #     MAP = dict()
    #     for parameter in priors:
    #         summary = result.get_one_dimensional_median_and_error_bar(parameter)
    #         MAP[parameter] = summary.median
    #     # plot_legend = ('$I_{'+str(colours[i])+'}(t)=$'
    #     #                 + str(int(MAP['background'])) + ' + ' +
    #     #                 str(int(MAP['scale1'])) + ' $\\exp($'
    #     #                 '$-|\\frac{'+str(np.round(MAP['rise1'], 2))+'}{t-'
    #     #                 +str(np.round(MAP['start1'], 2))+'}|^{'
    #     #                 +str(np.round(MAP['index1'], 2))+'}$ - '
    #     #                 +'$|\\frac{t-'+str(np.round(MAP['start1'], 2))
    #     #                 +'}{'+str(np.round(MAP['decay1'], 2))+'}|^{'
    #     #                 +str(np.round(MAP['index2'], 2))+'})$'
    #                   # )
    #     rates_fit  = one_pulse_lens_rate(delta_t, **MAP) / widths
    #     difference = GRB.rates[:,i] - rates_fit
    #     f2_ax1.plot(GRB.bin_left, GRB.rates[:,i], c = colours[i])
    #     f2_ax1.plot(GRB.bin_left, rates_fit, 'k:')#, label = plot_legend)
    #
    #     residual_axes[i].plot(GRB.bin_left, difference, c = colours[i])
    #     residual_axes[i].plot(GRB.bin_left, savgol_filter(difference, 15, 3), 'k:')

    # f2_ax1.set_xticks(())
    # # f2_ax1.legend(fontsize = 11)
    # f2_ax2.set_xticks(())
    # f2_ax3.set_xticks(())
    # f2_ax4.set_xticks(())
    #
    # l = outdir + '/' + label + 'rates.pdf'
    # fig2.savefig(l)










def lens_evidence():
    print('nice')
    return 9
