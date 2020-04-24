import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import bilby

from PyGRB_Bayes.postprocess.abp import AbstractBasePlot


class GravLens(AbstractBasePlot):
    """docstring for GravLens."""

    def __init__(self, **kwargs):
        self.plot_type = kwargs.get('p_type', 'presentation')
        super(GravLens, self).__init__(plot_type = self.plot_type)

        const_c = 299792458
        const_G = 6.67430e-11
        solar_M = 1.998e30
        self.point_prefactor = 0.5 * np.power(const_c, 3) / const_G

        # self.delt_array  = t_posterios
        # self.delmu_array = mu_posteriors

        self.fstring = kwargs.get('fstring')
        self.outdir  = kwargs.get('outdir')
        self.colours = ['red', 'orange', 'green', 'blue']
        self.clabels = ['1', '2', '3', '4']

        self.plot_delmu_delt()

    def generate_2x2_plot(self):
        height = self.plot_dict['width'] / 1.618
        # fig = plt.figure(figsize = (self.plot_dict['width'], height),
        #                  constrained_layout = False)
        fig, axes = plt.subplots( figsize = (self.plot_dict['width'], height),
                                constrained_layout = True)
        # # plot is in upper right corner of 2x2 grid
        # # top left and bottom right are for axes labels
        # spec = gridspec.GridSpec(   ncols = 2, nrows = 2,
        #                             figure = fig,
        #                             height_ratios = [0.95, 0.05],
        #                             width_ratios  = [0.05, 0.95],
        #                             hspace = 0.0, wspace = 0.0)
        # x_ax = fig.add_subplot(spec[0, 0], frameon = False)
        # y_ax = fig.add_subplot(spec[1, 1], frameon = False)
        # p_ax = fig.add_subplot(spec[0, 1], frameon = False)
        # x_ax =
        # y_ax =
        # axes = [x_ax, y_ax, p_ax]
        return fig, axes



    def plot_delmu_delt(self):
        fig, axes = self.generate_2x2_plot()

        axes.set_xlabel('Time Delay, $\\Delta t$ (s)',
                        fontsize = self.plot_dict['font_size'])
        axes.set_ylabel('Magnification Ratio, $r$',
                        fontsize = self.plot_dict['font_size'])
        labels = ['$\\Delta t$', '$r$']
        bounds = [(0.37, 0.42), (0.50, 1.5)]
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
        import scipy.stats as stats
        Zs     = []
        (xmin, xmax) = bounds[0]
        (ymin, ymax) = bounds[1]
        # j makes it complex number, then end points *are* included
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        # flattens X and Y into 1D arrays and then vstacks
        # eg. [[1,1,1,1,1,],[1,1,1,1,1]]
        positions = np.vstack([X.ravel(), Y.ravel()])
        #  ****************

        for ii in range(4):
            result_label = self.fstring + '_result_' + self.clabels[ii]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.result.read_in_result(filename = open_result)

            x = result.posterior['time_delay'].values
            y = result.posterior['magnification_ratio'].values
            defaults_kwargs['color'] = self.colours[ii]
            corner.hist2d(x, y, **defaults_kwargs, fig = fig)

            #  ****************
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            Zs.append( np.reshape(kernel(positions).T, X.shape) )

        H = np.ones(X.shape)
        for Z in Zs:
            H = np.multiply(H,Z)

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


        plot_name = f'{self.outdir}/{self.fstring}_delmu_delt.{self.plot_dict["ext"]}'
        fig.savefig(plot_name)


    def plot_mass_from_delmu_delt(self):
        f_r = np.reciprocal(
                np.log(self.delmu_array) +
                (self.delmu_array - 1) / np.sqrt(self.delmu_array)
                )
        mass_z = self.point_prefactor * self.delt_array * f_r

        fig, axes = self.generate_2x2_plot()
        axes[0].set_xlabel('Mass, $(1+z_\\textsc{l}) M_\\textsc{l}$ (M$_{\\odot}$)',
                        fontsize = self.plot_dict['font_size'])
        axes[1].set_ylabel('Probability Density',
                fontsize = self.plot_dict['font_size'])

        plot_name = f'{self.outdir}/{self.fstring}_mass.{self.plot_dict["ext"]}'
        fig.savefig(plot_name)


    def plot_vel_disp_from_delmu_delt(self):
        fig, axes = self.generate_2x2_plot()
        axes[0].set_xlabel('Velocity Dispersion, $\\sigma$ (km  sec$^{-1}$)',
                        fontsize = self.plot_dict['font_size'])
        axes[1].set_ylabel('Probability Density',
                        fontsize = self.plot_dict['font_size'])

        plot_name = f'{self.outdir}/{self.fstring}_vel_disp.{self.plot_dict["ext"]}'
        fig.savefig(plot_name)



if __name__ == '__main__':
    pass
