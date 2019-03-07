from mpi4py import MPI
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy import constants as const
import Lens_Mod
import os
import argparse
from emcee.utils import MPIPool
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts = MPI.Wtime()

ne = (.3 / (u.cm**3))
delta_ne = (.003 / (u.cm**3))

om = (np.logspace(3 + np.log10(5),
                  np.log10(4) + 4, 500) * u.MHz).to(1 / u.s)[::-1]

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


def vec_hist(a, bins):
    i = np.repeat(np.arange(0,a.shape[0]),a.shape[1])
    return np.histogram2d(i, a.flatten(), (a.shape[0], bins))[0]

def dspec_calc(task):
	global delta_ne
	global om
	global beta_dspec
	global Fgau
	global dspec
	global x_interp
	global Ds
	global s
	global widths
	dspec*=0
	dirname = task[0]
	rat = float(task[1][2:-4])
	dens = task[2]
	if rat == 0.0:
		I = np.load('./%s/I-0.0.npy' % dirname)
		x = np.load('./%s/x-0.0.npy' % dirname])
		I *= 0
		dr = './%s/A0/' % dirlist[dir_num]
	else:
		I = np.load('./%s/I-%s.npy' % (dirname, rat))
		x = np.load('./%s/x-%s.npy' % (dirname, rat))
		dr = './%s/A%s/' % (dirname, rat)
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
	Fdspec = np.fft.rfft(dspec, axis=1)
	om_max=np.zeros(widths.shape[0])
	beta_max=np.zeros(widths.shape[0])
	mu_max=np.zeros(widths.shape[0])
	for i in range(widths.shape[0]):
		dspec2 = np.fft.irfft(Fdspec * Fgau[i, :], axis=1)
		om_max[i] = om[dspec2.max(1) == dspec2.max()][0]
		beta_max[i] = beta_dspec[dspec2.max(0) == dspec2.max()][0]
		mu_max[i] = dspec2.max()
	return(mu_max,beta_max,om_max,dirnam,rat,dens)

pool = MPIPool(loadbalance=True)
if not pool.is_master():
	pool.wait()
	sys.exit(0)

dirlistall = os.listdir('./')
dirlist = list(filter(lambda x: x.startswith('Sims'), dirlistall))

filelist = list()
for i in range(len(dirlist)):
    filelistall = os.listdir('./%s/' % dirlist[i])
    temp_list = list(filter(lambda x: x.startswith('x-'), filelistall))
    filelist.append(temp_list)

tasks=list()
for i in range(len(dirlist))
	for k in range(len(filelist[i])):
		tasks.append((dirlist[i],filelist[i][k],'Under'))

vals = pool.map(dspec_calc,tasks)
np.save('MagPeaks.npy',vals)
