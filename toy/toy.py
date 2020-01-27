"""
An example of how to use bilby to perform paramater estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise

"""
from __future__ import division
import bilby
import numpy as np
import matplotlib.pyplot as plt

# A few simple setup steps
label = 'linear_regression'
outdir = 'outdir'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
    return (sg_A * np.exp(- np.square((time - sg_t_0) / sg_tau)) *
            np.cos(sg_omega * time + sg_phi) )


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(sg_A    = 1e4,
                            sg_t_0  = 3,
                            sg_tau  = 1,
                            sg_omega= 6,
                            sg_phi  = 1)

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = np.random.normal(1, 0.01, N)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, 'o', label='data')
ax.plot(time, model(time, **injection_parameters), '--r', label='signal')
ax.set_xlabel('time')
ax.set_ylabel('y')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors['sg_A']     = bilby.prior.LogUniform(1e2,1e4,latex_label='$A$')
priors['sg_t_0']   = bilby.prior.Uniform(3,7,latex_label='$t_0$')
priors['sg_tau']   = bilby.prior.LogUniform(1e-1,1e2,latex_label='$\\tau$')
priors['sg_omega'] = bilby.prior.LogUniform(1e-1,1e1,latex_label='$\\omega$')
priors['sg_phi']   = bilby.prior.Uniform(0,2*np.pi,latex_label='$\\phi$')

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='Nestle', nlive=500,
    sample='unif', injection_parameters=injection_parameters, outdir=outdir,
    label=label)
result.plot_corner()
