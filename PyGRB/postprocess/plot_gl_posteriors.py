import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

import bilby

from PyGRB.postprocess.abp import AbstractBasePlot


class GravLens(AbstractBasePlot):
    """docstring for GravLens."""

    def __init__(self, **kwargs):
        self.plot_type = kwargs.get('p_type', 'presentation')
        super(GravLens, self).__init__(plot_type = self.plot_type)

        self.lens_bounds = kwargs.get('lens_bounds')
        self.fstring = kwargs.get('fstring')
        self.outdir  = kwargs.get('outdir')
        self.colours = ['red', 'orange', 'green', 'blue']
        self.clabels = ['1', '2', '3', '4']

        self.plot_delmu_delt()
        self.plot_mass_from_delmu_delt()

    def _generate_plot(self):
        height = self.plot_dict['width'] / 1.618
        fig, axes = plt.subplots(   figsize = (self.plot_dict['width'], height),
                                    constrained_layout = True)
        return fig, axes



    def plot_delmu_delt(self):
        fig, axes = self._generate_plot()

        axes.set_xlabel('Time Delay, $\\Delta t$ (s)',
                        fontsize = self.plot_dict['font_size'])
        axes.set_ylabel('Magnification Ratio, $r$',
                        fontsize = self.plot_dict['font_size'])
        labels = ['$\\Delta t$', '$r$']
        bounds = self.lens_bounds
        defaults_kwargs = dict(
            bins=50, smooth=0.9,
            label_kwargs=dict(fontsize=self.plot_dict['font_size']),
            title_kwargs=dict(fontsize=self.plot_dict['font_size']),
            color = None,
            truth_color='tab:orange', #quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=False, fill_contours=True,
            max_n_ticks=3, labels = labels,
            range = bounds)

        #  ****************
        Zs     = []
        (xmin, xmax) = bounds[0]
        (ymin, ymax) = bounds[1]
        # j makes it complex number, then end points *are* included
        X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        # flattens X and Y into 1D arrays and then vstacks
        # eg. [[1,1,1,1,1,],[1,1,1,1,1]]
        positions = np.vstack([X.ravel(), Y.ravel()])
        #  ****************

        for ii in range(4):
            result_label = self.fstring + '_result_' + self.clabels[ii]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.result.read_in_result(filename = open_result)

            x = result.posterior['time_delay'].values
            y = 1 / result.posterior['magnification_ratio'].values
            defaults_kwargs['color'] = self.colours[ii]
            corner.hist2d(x, y, **defaults_kwargs, fig = fig)

            #  ****************
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            Zs.append( np.reshape(kernel(positions).T, X.shape) )

        H = np.ones(X.shape)
        for Z in Zs:
            H = np.multiply(H,Z)
        self.sum_posteriors = (X, Y, H)
        # Compute the density levels.
        # copied / adapted from corner.hist2d
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
        Hflat = H.flatten()
        inds  = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        if np.any(m) and not quiet:
            logging.warning("Too few points to create valid contours")
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()


        # This "color map" is the list of colors for the contour levels if the
        # contours are filled.
        from matplotlib.colors import LinearSegmentedColormap, colorConverter
        rgba_color = colorConverter.to_rgba('black')
        contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
        for i, l in enumerate(levels):
            contour_cmap[i][-1] *= float(i) / (len(levels)+1)

        axes.contourf(X, Y, H, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                        colors = contour_cmap   )
        axes.contour(X, Y, H, V, colors = 'black')
        # copied / adapted from corner.hist2d
        # axes.set_ylim(bottom = ymin)

        plot_name = f'{self.outdir}/{self.fstring}_delmu_delt.{self.plot_dict["ext"]}'
        fig.savefig(plot_name)
        plt.close(fig)


    @staticmethod
    def _return_mass(time, mag):
        """ Returns the lens mass for a given time delay and magnification
            ratio. The returned value is M(1+z_l) if the lens is cosmological.

            See Narayan & Wallington (1992), or Krauss & Small (1991).
        """
        c_cubed = 2.6944e25
        const_G = 6.67430e-11
        solar_M = 1.998e30
        prefactor = 0.5 * c_cubed / const_G / solar_M
        f_r = np.reciprocal( np.log(mag) + (mag - 1) / np.sqrt(mag) )
        return prefactor * time * f_r

    def open_result_file(self, fstring, outdir, clabel):
        result_label = f'{fstring}_result_{clabel}'
        open_result  = f'{outdir}/{result_label}_result.json'
        return bilby.result.read_in_result(filename = open_result)


    def _draw_mass_kde(self, time, mag, colour, axes):
        """ Draws """
        m = self._return_mass(time, mag)
        # cuts off negative masses, which 'exist' for mag ratio greater than 1.
        m = m[m > 0]
        print(colour, np.percentile(m, q = [5, 50, 95]))
        m = m[m < 2e5]
        density = stats.gaussian_kde(m)
        xs = np.geomspace(min(m), max(m), 100)
        dens = density(xs)
        axes.plot(        xs, dens, color = colour, linewidth = 0.2)
        axes.fill_between(xs, dens, color = colour, alpha = 0.2)
        return np.max(dens)

    def plot_mass_from_delmu_delt(self):
        fig, axes = self._generate_plot()
        ymax = - np.inf
        for ii in range(4):
            result = self.open_result_file( fstring = self.fstring,
                                            outdir  = self.outdir,
                                            clabel  = self.clabels[ii])
            x = result.posterior['time_delay'].values
            y = 1 / result.posterior['magnification_ratio'].values

            ymax = max(ymax, self._draw_mass_kde( time = x, mag = y,
                                colour = self.colours[ii], axes = axes))

        (X, Y, H) = self.sum_posteriors
        X, Y, H = X.ravel(), Y.ravel(), H.ravel()
        P = H / np.sum(H)
        idx = np.random.choice(len(X), size = 10000, replace = True, p = P)
        x, y = X[idx], Y[idx]
        ymax = max(ymax, self._draw_mass_kde( time = x, mag = y,
                            colour = 'black', axes = axes))

        axes.set_xlabel('Lens mass, $(1+z_\\textsc{l}) M_\\textsc{l}$ (M$_{\\odot}$)',
                        fontsize = self.plot_dict['font_size'])
        axes.set_ylabel('Probability Density',
                fontsize = self.plot_dict['font_size'])
        axes.set_xlim(1e4, 2e5)
        axes.set_xscale('log')
        ymin = 0.0001 * ymax
        # axes.set_ylim(bottom = ymin)
        # axes.set_yscale('log')
        axes.tick_params(axis='y', which = 'both', left = False, right = False, labelleft = False)
        plot_name = f'{self.outdir}/{self.fstring}_mass.{self.plot_dict["ext"]}'
        fig.savefig(plot_name)
        plt.close(fig)


    def plot_vel_disp_from_delmu_delt(self):
        fig, axes = self._generate_plot()
        axes[0].set_xlabel('Velocity Dispersion, $\\sigma$ (km  sec$^{-1}$)',
                        fontsize = self.plot_dict['font_size'])
        axes[1].set_ylabel('Probability Density',
                        fontsize = self.plot_dict['font_size'])

        plot_name = f'{self.outdir}/{self.fstring}_vel_disp.{self.plot_dict["ext"]}'
        fig.savefig(plot_name)



if __name__ == '__main__':
    pass
