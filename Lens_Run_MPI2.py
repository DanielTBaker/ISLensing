from mpi4py import MPI
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brenth
import astropy.units as u
from scipy.integrate import quad
from scipy.special import lambertw as W
from astropy import constants as const
import Lens_Mod
from scipy.misc import derivative
from time import perf_counter
import os
import argparse
from emcee.utils import MPIPool

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts = MPI.Wtime()


parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-s', type = int, default = 2, help = 'Lens Thickness Sigma')
parser.add_argument('-l', type = float, default = 0, help = 'Lower Bound on Amplitude Parameter')
parser.add_argument('-u', type = float, default = 2.2, help = 'Upper Bound on Amplitude Parameter')
parser.add_argument('-n', type = int, default = 45, help = 'Number of Amplitude Parameters in Range')

args = parser.parse_args()

rats = np.linspace(args.l,args.u,args.n)

T = .03*(u.AU.to(u.m))
A = .3*(u.AU.to(u.m))
R = 4.8*(u.kpc.to(u.m))
sig = (1./args.s)*T/(2*np.sqrt(2*np.log(2)))
#S_par=np.array([A,np.sqrt(A*R)])
inc = 1e-5*u.rad

Ds = (389*u.pc).to(u.m)

params = np.array([sig, inc.value, Ds.value, np.sqrt(A*R)])

def sheet(z,A,sig):
    return(A*np.exp(-np.power(z/sig,2)/2))
def sheet_dl(z,A,sig):
    return(np.sqrt(1+np.power(A*z/(sig**2),2)*np.exp(-np.power(z/sig,2))))
def sheet_dir(z,inc,A,sig):
    return(-z*A*np.exp(-np.power(z/sig,2)/2)/(sig**2)-np.tan(inc.value))

pool = MPIPool(loadbalance=True)
if not pool.is_master():
	pool.wait()
	sys.exit(0)

dirlistall = os.listdir('./')
dirlist = list(filter(lambda x: x.startswith('Sims'), dirlistall))
Exists=False
for i in range(len(dirlist)):
	if np.all(params==np.load('./%s/Params.npy' %dirlist[i])):
		Exists=True
		par_number=dirlist[i][5:]
if not Exists:
	par_number=len(dirlist)+1
	while not Exists:
		try:
			os.mkdir('./Sims-%s' %par_number)
			np.save('./Sims-%s/Params.npy' %par_number,params)
			Exists=True
		except:
			if np.all(params==np.load('./Sims-%s/Params.npy' %par_number)):
				Exists=True
			else:
				par_number+=1

filelistall = os.listdir('./Sims-%s/' %par_number)
filelist = list(filter(lambda x: x.startswith('x-'), filelistall))
Amps=np.zeros(len(filelist))
for i in range(Amps.shape[0]):
    Amps[i]=float(filelist[i][2:-4])
rats=rats[np.isin(rats,Amps,invert=True)]

for rat in rats:
	S_par=np.array([np.tan(inc.value)*np.sqrt(A*R)*np.exp(1./2.)*rat,np.sqrt(A*R)])

	zmax=100*S_par[1]
	zmin=-100*S_par[1]
	x0_crit,z_list=Lens_Mod.zbnd_find(zmin,zmax,10000,sheet_dir,sheet,S_par,inc)
	print(x0_crit,z_list)

	x=((np.linspace(-10,10,10000))*(u.mas.to(u.rad))*Ds).astype('float128')

	for i in range(x0_crit.shape[0]):
		x=np.concatenate((x,np.linspace(x0_crit[i]-5*sig,x0_crit[i]+5*sig,1001)))
	x=np.unique(x).value*Ds.unit

	print('%s I Calc Start at %s' %(rat,MPI.Wtime()-ts))

	I=Lens_Mod.I_calc_mpi(x.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig,pool)

	x=x[I>I[0]/1e10]
	I=I[I>I[0]/1e10]

	print('(%s) %s Rough Grid Complete at %s' %(rank,rat,MPI.Wtime()-ts))

	x2,I2=Lens_Mod.res_improve_mpi(1e-5,x,I,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig,pool,size)

	np.save('./Sims-%s/x-%s' %(par_number,rat) , x2.value)
	np.save('./Sims-%s/I-%s' %(par_number,rat) , I2)
	plt.figure()
	plt.plot(x2,I2)
	plt.plot(x,I)
	plt.savefig('./Sims-%s/Thickness-%s.png' %(par_number,rat))
	plt.close('all')

	time_end=perf_counter()
	print('%s Done at %s' %(rat,MPI.Wtime()-ts))

pool.close()
