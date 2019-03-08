from mpi4py import MPI
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
from emcee.utils import MPIPool

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts = MPI.Wtime()

ne = (.3 / (u.cm**3))
delta_ne = (.003 / (u.cm**3))

om = (np.logspace(3,
                  np.log10(6) + 4, 2000) * u.MHz).to(1 / u.s)[::-1]

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

def dspec_find(task):
	rat=task[1]
	direct=task[0]
	dens=task[2]
	global x_interp
	global ne
	global delta_ne
	global om
	global Ds
	global s
	global width
	global Fgau
	global dspec
	ts=MPI.Wtime()
	if rat == 0.0:
		I = np.load('./%s/I-0.0.npy' % direct)
		x = np.load('./%s/x-0.0.npy' % direct)
		I *= 0
		dr = './%s/A0/' % direct
	else:
		I = np.load('./%s/I-%s.npy' % (direct, rat))
		x = np.load('./%s/x-%s.npy' % (direct, rat))
		dr = './%s/A%s/' % (direct, rat)
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
	print('%s %s Interpolation Complete at %s' % (direct,rat,MPI.Wtime()-ts))
	dspec *= 0
	if dens == 'Over':
		delta_ne = np.abs(delta_ne)
	else:
		delta_ne = -np.abs(delta_ne)
	for i in range(om.shape[0]):
		theta_AR, beta_AR = Lens_Mod.Im_find(x_interp * u.m, I_interp, ne,delta_ne, om[i:i+1], Ds, s)
		dspec[i,:-1]=np.histogram(beta_AR[0,:], bins=beta_dspec)
	dspec *= (np.median((np.diff(x_interp) / Ds.value) * (u.rad.to(u.mas))) /np.median(np.diff(beta_dspec))).value
	dspec[:, -1] = 1
	print('%s %s %sdense Dspec Complete at %s' %(direct, rat, dens, MPI.Wtime()-ts))
	om_temp = om[dspec.max(1) == dspec.max()][:1]
	theta_AR, beta_AR = Lens_Mod.Im_find(x_interp * u.m, I_interp, ne, delta_ne, om_temp, Ds, s)
	mu = 1 / np.gradient(beta_AR[0,:], theta_AR)
	rough_temp = idx[1:][np.abs(np.diff(np.sign(mu))) == 2]
	rough = np.concatenate((np.zeros(1, dtype=int), rough_temp,
	np.array([x_interp.shape[0]], dtype=int)))
	plt.figure(1)
	for k in range(rough.shape[0] - 1):
		plt.subplot(211)
		plt.plot(beta_AR[0,rough[k]:rough[k + 1]],
		theta_AR[rough[k]:rough[k + 1]])
		plt.subplot(212)
		plt.plot(beta_AR[0,rough[k]:rough[k + 1]],
		np.abs(mu[rough[k]:rough[k + 1]]))
	plt.xlabel('Pulsar Position (mas)')
	plt.ylabel('Magnification')
	plt.title('Magnification for $A_{par} = %s$ (%sdense)' %
	(rat, dens))
	plt.yscale('log')
	plt.subplot(211)
	plt.xlabel('Pulsar Position (mas)')
	plt.ylabel('Image Position (mas)')
	plt.title('Image Positions for $A_{par} = %s$ (%sdense)' %(rat, dens))
	plt.xlim((-5, 5))
	plt.ylim((-5, 5))
	plt.tight_layout()
	plt.savefig('%sImages-%s%s.png' % (dr, rat, dens[0]))
	plt.close('all')

	Fdspec = np.fft.rfft(dspec, axis=1)
	om_max=np.zeros(widths.shape)
	beta_max=np.zeros(widths.shape)
	mu_max=np.zeros(widths.shape)
	for i in range(widths.shape[0]):
		dspec2 = np.fft.irfft(Fdspec * Fgau[i, :], axis=1)
		print('%s %s %s %sdense %s width Dspec Complete at %s' % (direct, rat, dens, widths[i], MPI.Wtime()-ts))
		om_max[i] = om[dspec2.max(1) == dspec2.max()][0]
		beta_max[i] = beta_dspec[dspec2.max(0) == dspec2.max()][0]
		mu_max[i] = dspec2.max()
	return(mu_max,om_max,beta_max,dens,rat,direct)

pool = MPIPool(loadbalance=True)
if not pool.is_master():
	pool.wait()
	sys.exit(0)

tasks =list()
for i in range(len(dirlist)):
	for k in range(len(filelist[i])):
		tasks.append((dirlist[i],float(filelist[i][k][2:-4]),'Under'))

vals=pool.map(dspec_find,tasks)
pool.close()

mu_max=np.zeros((len(vals),widths.shape[0]))
om_max=np.zeros((len(vals),widths.shape[0]))
beta_max=np.zeros((len(vals),widths.shape[0]))
rats=np.zeros(len(vals))
dens=np.empty(len(vals),dtype=str)
dirs=np.empty(len(vals),dtype=str)

for i in range(len(vals)):
	mu_max[i,:]=vals[i][0]
	om_max[i,:]=vals[i][1]
	beta_max[i,:]=vals[i][2]
	rats[i]=vals[i][4]
	dens[i]=vals[i][3]
	dirs[i]=vals[i][5]

np.savez('Evolution.npz', mu_max,om_max,beta_max,rats,dens,dirs)



