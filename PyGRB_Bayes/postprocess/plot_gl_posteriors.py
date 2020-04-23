import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PyGRB_Bayes.postprocess.abp import AbstractBasePlot


class GravLens(AbstractBasePlot):
    """docstring for GravLens."""

    def __init__(self, t_posterios, mu_posteriors, **kwargs):
        self.plot_type = kwargs.get('p_type', 'presentation')
        super(GravLens, self).__init__(plot_type = self.plot_type)

        const_c = 299792458
        const_G = 6.67430e-11
        self.point_prefactor = 0.5 * np.power(const_c, 3) / const_G

        self.delt_array  = t_posterios
        self.delmu_array = mu_posteriors

        self.fstring = kwargs.get('fstring')
        self.outdir  = kwargs.get('outdir')

    def generate_2x2_plot(self):
        height = self.plot_dict['width'] / 1.618
        fig = plt.figure(figsize = (self.plot_dict['width'], height),
                         constrained_layout = False)
        # plot is in upper right corner of 2x2 grid
        # top left and bottom right are for axes labels
        spec = gridspec.GridSpec(   ncols = 2, nrows = 2,
                                    figure = fig,
                                    height_ratios = [0.05, 0.95],
                                    width_ratios  = [0.05, 0.95],
                                    hspace = 0.0, wspace = 0.0)
        x_ax = fig.add_subplot(spec[0, 0], frameon = False)
        y_ax = fig.add_subplot(spec[1, 1], frameon = False)
        p_ax = fig.add_subplot(spec[1, 0], frameon = False)
        axes = [x_ax, y_ax, p_ax]
        return fig, axes

    def plot_delmu_delt(self):
        fig, axes = self.generate_2x2_plot()
        axes.set_ylabel('Magnification Ratio, $r$',
                        fontsize = self.plot_dict['font_size'])
        axes.set_xlabel('Time Delay, $\\Delta t$ (s)',
                        fontsize = self.plot_dict['font_size'])

        plot_name = f'{self.outdir}/{self.fstring}_delmu_delt.{self.plot_dict['ext']}'
        fig.savefig(plot_name)


    def plot_mass_from_delmu_delt(self):
        f_r = np.reciprocal(
                np.log(self.delmu_array) +
                (self.delmu_array - 1) / np.sqrt(self.delmu_array)
                )
        mass_z = self.point_prefactor * self.delt_array * f_r

        fig, axes = self.generate_2x2_plot()
        axes.set_xlabel('Mass, $(1+z_\\textsc{l}) M_\\textsc{l}$ (M$_{\\odot}$)',
                        fontsize = self.plot_dict['font_size'])
        axes.set_ylabel('Relative Probability',
                fontsize = self.plot_dict['font_size'])

        plot_name = f'{self.outdir}/{self.fstring}_mass.{self.plot_dict['ext']}'
        fig.savefig(plot_name)


    def plot_vel_disp_from_delmu_delt(self):
        fig, axes = self.generate_2x2_plot()
        axes.set_xlabel('Velocity Dispersion, $\\sigma$ (km  sec$^{-1}$)',
                        fontsize = self.plot_dict['font_size'])
        axes.set_ylabel('Relative Probability',
                        fontsize = self.plot_dict['font_size'])

        plot_name = f'{self.outdir}/{self.fstring}_vel_disp.{self.plot_dict['ext']}'
        fig.savefig(plot_name)



if __name__ == '__main__':
    pass
