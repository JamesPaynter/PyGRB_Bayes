import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from astropy import cosmology

from matplotlib import rc
rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'],'size': 8})
rc('text', usetex=True)

def F_x(x):
    ''' From Saas Fee pg. 77'''
    return np.where(x == 1, 1/3,
          (np.where(x  < 1,  np.arccosh(1/x) / np.sqrt(1 - np.square(x)),
                             np.arccos( 1/x) / np.sqrt(np.square(x) - 1)))    )
def f_x(x):
    ''' From Saas Fee pg. 77'''
    return (1 - F_x(x)) / (np.square(x) - 1)
def h_x(x):
    ''' From Saas Fee pg. 77'''
    return 2 * (F_x(x) + np.log(x / 2)) / np.square(x)


def comoving_line_of_sight_distance(z):
    ''' From Hogg '''
    omega_m = 0.3
    omega_v = 0.7
    const_c = 3e8
    const_H = 67e3 / 3.086e22
    integral, err = integrate.quad(
    lambda x: const_c / const_H * (np.sqrt(omega_v + omega_m * ((1 + x) ** 3))),
    0, z)
    return integral

def d_A(z_0, z_s):
    ''' From Dodelson '''
    return np.where(z_0 == 0, comoving_line_of_sight_distance(z_s) / (1 + z_s),
                             (comoving_line_of_sight_distance(z_s) -
                              comoving_line_of_sight_distance(z_0))/ (1 + z_s) )

def H_z_sqrd(z):
    omega_m = 0.3
    omega_v = 0.7
    const_H = 67e3 / 3.086e22
    return const_H ** 2 * (omega_v + omega_m * ((1 + z) ** 3))

def delta(conc):
    ''' From Saas Fee pg. 76'''
    return (200 / 3) * (conc ** 3) / (np.log(1 + conc) - ( conc / (1 + conc))  )

def rho_crit(z_l):
    ''' From Saas Fee pg. 76'''
    const_G = 6.67e-11
    const_H = 67e3 / 3.086e22
    return 3 * H_z_sqrd(z_l) / (8 * np.pi * const_G)

def Sigma_crit(z_l, z_s):
    ''' From Saas Fee pg. 21'''
    const_c = 3e8
    const_G = 6.67e-11
    return (const_c ** 2) / (4 * np.pi * const_G
                        ) * d_A(0, z_s) / (d_A(0, z_s) * d_A(z_l, z_s))

def kappa_crit(r_s, conc, z_l, z_s):
    ''' From Saas Fee pg. 76 '''
    return 2 * r_s * delta(conc) * rho_crit(z_l) / Sigma_crit(z_l, z_s)

def c_from_mass(mass, z):
    ''' mass in solar masses

        Stu's paper 2015 -- The accretion history of dark haloes III
    '''
    alpha = 1.62774 - 0.2458  * (1 + z) + 0.01716 * (1 + z) ** 2
    beta  = 1.66079 + 0.00359 * (1 + z) - 1.69010 * (1 + z) ** 0.00417
    gamma =-0.02049 + 0.0253  * (1 + z) ** -0.1044

    xi    = 1.226 - 0.1009 * (1 + z) * 0.00378 * (1 + z) ** 2
    zeta  = 0.008634 - 0.08814 * (1 + z) ** -0.58816

    return np.where(z < 4,  10 ** (alpha + (beta * np.log10(mass)) *
                                (1 + gamma * np.square(np.log10(mass)))),
                            10 ** (xi + zeta * np.log10(mass))
                    )

def c_from_mass_planck(mass, z):
    ''' mass in solar masses

        Stu's paper 2015 -- The accretion history of dark haloes III
    '''
    alpha = 1.7543  - 0.2766  * (1 + z) + 0.02039 * (1 + z) ** 2
    beta  = 0.2573  + 0.00351 * (1 + z) - 0.3038  * (1 + z) ** 0.0269
    gamma =-0.01537 + 0.02102 * (1 + z) ** -0.1475

    xi    = 1.3081 - 0.1078 * (1 + z) * 0.00398 * (1 + z) ** 2
    zeta  = 0.0223 - 0.0944 * (1 + z) ** -0.3907

    return np.where(z < 4,  10 ** (alpha + (beta * np.log10(mass)) *
                                (1 + gamma * np.square(np.log10(mass)))),
                            10 ** (xi + zeta * np.log10(mass))
                    )

# masses = np.geomspace(1e4, 1e16, 1000)
# plt.plot(masses, c_from_mass_planck(masses, 0))
# plt.xscale('log')
# plt.show()


def return_kappa(theta, mass, conc, z_l, z_s, bar = False):
    ''' From Saas Fee pg. 77

        theta:  The angle between the lens and the observed image

        # r_s:    The charateristic radial size of the NFW profile
        mass:   The M200 mass of the NFW profile

        conc:   The concentration of the NFW profile

        z_l:    The redshift of the NFW deflector

        z_s:    The redshift of the source (GRB)

    '''
    r_s     = (3 * mass / (800 * np.pi * rho_crit(z_l))) ** (1/3)
    theta_s = r_s / d_A(0, z_l)
    if bar:
        return kappa_crit(r_s, conc, z_l, z_s) * h_x(theta / theta_s)
    return kappa_crit(r_s, conc, z_l, z_s) * f_x(theta / theta_s)


def return_kappa_stus_conc(theta, mass, z_l, z_s, bar = False):
    ''' From Saas Fee pg. 77

        theta:  The angle between the lens and the observed image

        # r_s:    The charateristic radial size of the NFW profile
        mass:   The M200 mass of the NFW profile

        # conc:   The concentration of the NFW profile
                now updated with stuart's paper's m conc relation

        z_l:    The redshift of the NFW deflector

        z_s:    The redshift of the source (GRB)

    '''
    r_s     = (3 * (mass * 2e30) / (800 * np.pi * rho_crit(z_l))) ** (1/3)
    theta_s = r_s / d_A(0, z_l)
    conc    = c_from_mass_planck(mass, z_l)
    if bar:
        result = kappa_crit(r_s, conc, z_l, z_s) * h_x(theta / theta_s)
    result = kappa_crit(r_s, conc, z_l, z_s) * f_x(theta / theta_s)
    return result

def find_theta_given_m(mass, z_l, z_s, bar, kappa, guess):
    vfunc = np.vectorize(return_kappa_stus_conc)
    try:
        output = fsolve(lambda x: vfunc(abs(x), mass, z_l, z_s, bar) - kappa,
                        guess, full_output = True)
        theta, = output[0]
        if 'converged' not in output[-1]:
            return 0
        else:
            return theta
    except:
        return 0

def point_mass_einstein_radius(mass, z_l, z_s):
    const_c = 3e8
    const_G = 6.67e-11
    M_solar = 2e30
    dist_ra = d_A(z_l, z_s) / (d_A(0, z_l) * d_A(0, z_s))
    return np.sqrt(4 * const_G * mass * M_solar * dist_ra) / const_c / 4.8481368e-9

def plot_mass_vs_convergence():

    width   = 3.321
    height  = 3.321
    fig, ax = plt.subplots(figsize = (width, height))

    halo_mass_array = np.geomspace(1e3, 1e8, 500)

    ls = ['-', '-.', '--', ':'][::-1]
    colours = [ u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    z_l_arr = [0.1, 0.5, 1]
    z_s_arr = [0.5, 1.0, 2, 5]

    for i, z_l in enumerate(z_l_arr):
        for j, z_s in enumerate(z_s_arr):
            if z_s > z_l:
                thetas = np.zeros(len(halo_mass_array))
                for k, halo in enumerate(halo_mass_array):
                    thetas[k] = find_theta_given_m(halo, z_l, z_s, bar = True, kappa = 1, guess = 1e-7)

                inds= np.argwhere(thetas > 0)
                ax.plot(    halo_mass_array[inds], thetas[inds]/ 4.8481368e-9,
                            c = colours[i], linestyle = ls[j])

    lines = ax.get_lines()
    z_l_labels = [f'$z_l =$ {z_l_arr[i]}' for i in range(3)]
    z_s_labels = [f'$z_s =$ {z_s_arr[i]}' for i in range(4)]
    legend1 = plt.legend([lines[i] for i in [2, 5, 7]], z_l_labels,
                            loc=2)
    #dummy lines with NO entries, just to create the black style legend
    dummy_lines = []
    for b_idx, b in enumerate(z_s_arr):
        dummy_lines.append(ax.plot([],[], c="black", ls = ls[b_idx])[0])
    legend2 = plt.legend([dummy_lines[i] for i in range(4)], z_s_labels,
                            loc=4)
    ax.add_artist(legend1)


    # ax.set_title('$z_l = 1$, $z_s =$ 1.1 (solid), 2 (dash dot), 5 (dash), 10 (dotted)')
    ax.set_xlabel('Halo Mass, $M$ (M$_\\odot$)')
    ax.set_ylabel('Effective Einstein Radius, $\\theta(\\kappa=1)$ (mas)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(halo_mass_array[0], halo_mass_array[-1])
    ax.set_ylim(1e-2, max(thetas)/ 4.8481368e-9)

    mass = np.geomspace(1e3, 1e8, 1000)
    ax.plot(mass, point_mass_einstein_radius(mass, 1, 2), 'k:')

    plt.tight_layout()
    fig.savefig('mass_vs_ER2.pdf')
    plt.show()


plot_mass_vs_convergence()









def plot_theta_vs_convergence():
    mas_array = np.geomspace(1e-12, 1e-2, 100000) /  4.8481368e-9 ## radians, 1e-7 is one arc sec?
    theta_array = mas_array * 4.8481368e-9
    halo_mass = 1e5 * 2e30
    y = return_kappa(theta_array, halo_mass, 11, 1, 10)


    ls = ['-', '-.', '--', ':']
    colours = [ u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    fig, ax = plt.subplots(figsize = (16,10))
    for i, c in enumerate([0.1, 1.0, 10, 100, 1e3]):
        for j, z_s in enumerate([1.1, 2., 5., 10.]):
            ax.plot(mas_array, return_kappa(theta_array, halo_mass, c, 1, z_s),
                c = colours[i], linestyle = ls[j],
                label = {0: f'c = {c}'}.get(j, '') ) ## put in for z_s too ??
    ax.fill_between([0,1e8], 1,10, color='r', alpha = 0.1)
    ax.set_title('$z_l = 1$, $z_s =$ 1.1 (solid), 2 (dash dot), 5 (dash), 10 (dotted)')
    ax.set_xlabel('Image position, $\\theta$ (mas)')
    ax.set_ylabel('Convergence, $\\kappa$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(mas_array[0],mas_array[-1])
    ax.set_ylim(1e-6,10)
    ax.legend()
    plt.show()
