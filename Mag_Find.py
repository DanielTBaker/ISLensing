import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit, brenth
from scipy.interpolate import interp1d
import astropy.units as u
from scipy.signal import convolve2d,convolve
from scipy.integrate import cumtrapz
from astropy import constants as const
from matplotlib import animation
from time import perf_counter
from scipy.integrate import dblquad,quad
from scipy.special import lambertw as W
from scipy.misc import derivative
import Lens_Mod
import os
plt.ioff()
from time import perf_counter
from scipy.optimize import minimize
from matplotlib.colors import LogNorm

ne = (.3 / (u.cm**3))
delta_ne = (.003 / (u.cm**3))

om = (np.logspace(3,
                  np.log10(6) + 4, 1000) * u.MHz).to(1 / u.s)[::-1]

Ds = (389 * u.pc).to(u.m)
Dp = (620 * u.pc).to(u.m)
Dps = Ds - Dp
s = 1 - Ds / Dp

widths = np.logspace(-5, -2, 4) * u.mas

beta_dspec = np.linspace(0, 2, 100000) * u.mas
x_interp = np.linspace(-5, 5, 2000000) * (u.mas.to(u.rad)) * Ds.value
idx = np.linspace(0, x_interp.shape[0] - 1, x_interp.shape[0]).astype(int)

beta_dspec_shift = np.roll(beta_dspec, int(beta_dspec.shape[0] / 2))
beta_dspec_shift -= beta_dspec_shift[0]

gau = np.exp(
    -np.power(beta_dspec_shift[np.newaxis, :] / widths[:, np.newaxis], 2) /
    2)
gau /= gau.sum(1)[:, np.newaxis]
Fgau = np.fft.rfft(gau, axis=1)
dspec = np.zeros((om.shape[0], beta_dspec.shape[0]))

dirlistall = os.listdir('./')
dirlist = list(filter(lambda x: x.startswith('Sims'), dirlistall))

filelist = list()
for i in range(len(dirlist)):
    filelistall = os.listdir('./%s/' % dirlist[i])
    temp_list = list(filter(lambda x: x.startswith('x-'), filelistall))
    filelist.append(temp_list)

run_start = perf_counter()
def vec_hist(a, bins):
    i = np.repeat(np.arange(0,a.shape[0]),a.shape[1])
    return np.histogram2d(i, a.flatten(), (a.shape[0], bins))[0]

for dir_num in range(len(dirlist)):
    Amps = np.zeros(len(filelist[dir_num]))
    for i in range(Amps.shape[0]):
        Amps[i] = float(filelist[dir_num][i][2:-4])
    Amps = np.sort(Amps)

    freqs_under = np.zeros((Amps.shape[0], widths.shape[0])) * u.GHz
    freqs_over = np.zeros((Amps.shape[0], widths.shape[0])) * u.GHz

    mags_under = np.zeros((Amps.shape[0], widths.shape[0]))
    mags_over = np.zeros((Amps.shape[0], widths.shape[0]))

    beta_under = np.zeros((Amps.shape[0], widths.shape[0])) * u.mas
    beta_over = np.zeros((Amps.shape[0], widths.shape[0])) * u.mas

    if os.path.isfile('%s/Evolution.npz' % dirlist[dir_num]):
        temp = np.load('%s/Evolution.npz' % dirlist[dir_num])
        freqs_under[np.isin(Amps, temp['A'])] = temp['f_u'] * u.GHz
        freqs_over[np.isin(Amps, temp['A'])] = temp['f_0'] * u.GHz

        mags_under[np.isin(Amps, temp['A'])] = temp['mu_u']
        mags_over[np.isin(Amps, temp['A'])] = temp['mu_0']

        beta_under[np.isin(Amps, temp['A'])] = temp['beta_u'] * u.mas
        beta_over[np.isin(Amps, temp['A'])] = temp['beta_0'] * u.mas

        Amps = Amps[np.isin(Amps, temp['A'], invert=True)]

    print(Amps)
    if Amps.shape[0]>0:
        for rat in Amps:
            if rat == 0.0:
                I = np.load('./%s/I-0.0.npy' % dirlist[dir_num])
                x = np.load('./%s/x-0.0.npy' % dirlist[dir_num])
                I *= 0
                dr = './%s/A0/' % dirlist[dir_num]
            else:
                I = np.load('./%s/I-%s.npy' % (dirlist[dir_num], rat))
                x = np.load('./%s/x-%s.npy' % (dirlist[dir_num], rat))
                dr = './%s/A%s/' % (dirlist[dir_num], rat)
            if not os.path.exists(dr):
                os.mkdir(dr)
            I = I[np.concatenate((np.array([True]), np.abs(np.diff(x)) > 0))]
            x = x[np.concatenate((np.array([True]), np.abs(np.diff(x)) > 0))]
            if rat==0.0:
                I[:]=I.mean()
            plt.figure(1)
            plt.plot(((x / Ds.value) * u.rad).to(u.mas), I)
            plt.xlabel('Line of Sight Coordinate (mas)')
            plt.ylabel('Effective Thickness')
            plt.title('Lens Profile for $A_{par} = %s$' % rat)
            plt.xlim((-5, 5))
            plt.savefig('%sI-%s.png' % (dr, rat))
            plt.close('all')
            interp = interp1d(x, I, kind='quadratic')
            I_interp = interp(x_interp)
            print('%s Interpolation Complete at %s' % (rat,
                                                       perf_counter() - run_start))
            for dens in np.array(['Under']):
                dspec *= 0
                if dens == 'Over':
                    delta_ne = np.abs(delta_ne)
                else:
                    delta_ne = -np.abs(delta_ne)
                beta_AR = np.empty(x_interp.shape)
                theta_AR, beta_AR = Lens_Mod.Im_find(x_interp * u.m, I_interp, ne,
                                                                 delta_ne, om, Ds, s)
                dspec[:,:-1]=vec_hist(beta_AR, beta_dspec)
                dspec *= (np.median(
                    (np.diff(x_interp) / Ds.value) * (u.rad.to(u.mas))) /
                          np.median(np.diff(beta_dspec))).value
                dspec[:, -1] = 1
                print('%s %sdense Dspec Complete at %s' %
                      (rat, dens, perf_counter() - run_start))
                om_temp = om[dspec.max(1) == dspec.max()][:1]
                theta_AR, beta_AR = Lens_Mod.Im_find(
                    x_interp * u.m, I_interp, ne, delta_ne, om_temp, Ds, s)
                mu = 1 / np.gradient(beta_AR[0,:], theta_AR)
                rough_temp = idx[1:][np.abs(np.diff(np.sign(mu))) == 2]
                rough = np.concatenate((np.zeros(1, dtype=int), rough_temp,
                                        np.array([x_interp.shape[0]], dtype=int)))
                plt.figure(1)
                for k in range(rough.shape[0] - 1):
                    plt.subplot(211)
                    plt.plot(beta_AR[0,rough[k]:rough[k + 1]],
                             theta_AR[rough[k]:rough[k + 1]])
                    plt.xlabel('Pulsar Position (mas)')
                    plt.ylabel('Image Position (mas)')
                    plt.title('Image Positions for $A_{par} = %s$ (%sdense)' %
                              (rat, dens))
                    plt.subplot(212)
                    plt.plot(beta_AR[0,rough[k]:rough[k + 1]],
                             np.abs(mu[rough[k]:rough[k + 1]]))
                    plt.xlabel('Pulsar Position (mas)')
                    plt.ylabel('Magnification')
                    plt.title('Magnification for $A_{par} = %s$ (%sdense)' %
                              (rat, dens))
                    plt.yscale('log')
                plt.subplot(211)
                plt.xlim((-5, 5))
                plt.ylim((-5, 5))
                plt.tight_layout()
                plt.savefig('%sImages-%s%s.png' % (dr, rat, dens[0]))
                plt.close('all')

                om_idx = int(np.argwhere(om == om_temp)[0])
                beta_idx1, beta_idx2, om_idx1, om_idx2 = 0, -1, 0, -1
                print('%s %sdense Dspec Plotting Complete at %s' %
                      (rat, dens, perf_counter() - run_start))
                Fdspec = np.fft.rfft(dspec, axis=1)
                for i in range(widths.shape[0]):
                    dspec2 = np.fft.irfft(Fdspec * Fgau[i, :], axis=1)
                    print('%s %sdense %s width Dspec Complete at %s' %
                          (rat, dens, widths[i], perf_counter() - run_start))
                    om_temp = om[dspec2.max(1) == dspec2.max()][0]
                    beta_max = beta_dspec[dspec2.max(0) == dspec2.max()][0]
                    if dens == 'Over':
                        freqs_over[Amps == rat, i] = om_temp
                        mags_over[Amps == rat, i] = dspec2.max()
                        beta_over[Amps == rat, i] = beta_max
                    else:
                        freqs_under[Amps == rat, i] = om_temp
                        mags_under[Amps == rat, i] = dspec2.max()
                        beta_under[Amps == rat, i] = beta_max
        np.savez(
            '%s/Evolution.npz' % dirlist[dir_num],
            mu_o=mags_over,
            mu_u=mags_under,
            beta_o=beta_over,
            beta_u=beta_under,
            f_u=freqs_under,
            f_o=freqs_over,
            wid=widths,
            A=Amps)
