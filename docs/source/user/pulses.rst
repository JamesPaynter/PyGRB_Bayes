Pulse Types
===========

A standard gamma-ray burst looks like





A simple pulse parameterisation one might imagine is a Gaussian, like those used to model emission lines in eg. quasar spectra.
The equation for a Gaussian pulse is

.. math::

    S(t|A,\Delta,\sigma) = A \exp \left[ \frac{\left( t - \Delta \right)^2}{2\sigma^2} \right]

The standard pulse parameterisation used to model gamma-ray bursts is a fast-rise exponential-decay (FRED) curve.

.. math::

    S(t|A,\Delta,\tau,\xi) = A \exp \left[ - \xi \left(  \frac{t - \Delta}{\tau} + \frac{\tau}{t-\Delta}  \right)   \right]

.. math::

    S(t|A,\Delta,\tau,\xi,\gamma,\nu) = A \exp \left[ -\xi^\gamma \left(\frac{t - \Delta}{\tau}\right)^\gamma - \xi^\nu \left(\frac{\tau}{t-\Delta}\right)^\nu\right]


These parameterisations do always not capture all the structure in a gamma-ray burst pulse.
We use a sine-gaussian residual function to account for these residuals.

.. math::

    \text{res}(t)= A_\text{res} \exp \left[ - \left(\frac{t-\Delta_\text{res}} {\tau_\text{res}}\right)^2 \right] \cos\left(\omega t + \varphi \right)
