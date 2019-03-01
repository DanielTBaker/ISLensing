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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts=MPI.Wtime()


parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-s',type=int,default = 2,help='Lens Thickness Sigma')

args=parser.parse_args()

def job_nums(size,jobs,rank):
    n_base=jobs//size
    extras=np.mod(jobs,size)
    if rank<extras:
        nums=np.linspace(0,n_base,n_base+1,dtype=int)
        nums+=rank*(n_base+1)
    else:
        nums=np.linspace(0,n_base-1,n_base,dtype=int)
        nums+=extras*(n_base+1)+(rank-extras)*n_base
    return(nums)

rats=np.linspace(.8,2.2,29)

T=.03*(u.AU.to(u.m))
A=.3*(u.AU.to(u.m))
R=4.8*(u.kpc.to(u.m))
sig=T/(2*np.sqrt(2*np.log(2)))
#S_par=np.array([A,np.sqrt(A*R)])
inc=1e-5*u.rad

Ds=(389*u.pc).to(u.m)

print(args.s)
params=np.array([args.s,inc.value,Ds.value,np.sqrt(A*R)])
print(params)

dirlistall = os.listdir('./')
dirlist = list(filter(lambda x: x.startswith('Sims'), dirlistall))
if rank==0:
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
else:
	par_number=None
comm.Barrier()
par_number = comm.bcast(par_number, root=0)
filelistall = os.listdir('./Sims-%s/' %par_number)
filelist = list(filter(lambda x: x.startswith('x-'), filelistall))
Amps=np.zeros(len(filelist))
for i in range(Amps.shape[0]):
    Amps[i]=float(filelist[i][2:-4])
rats=rats[np.isin(rats,Amps,invert=True)]
comm.Barrier()




def sheet(z,A,sig):
    return(A*np.exp(-np.power(z/sig,2)/2))
def sheet_dl(z,A,sig):
    return(np.sqrt(1+np.power(A*z/(sig**2),2)*np.exp(-np.power(z/sig,2))))
def sheet_dir(z,inc,A,sig):
    return(-z*A*np.exp(-np.power(z/sig,2)/2)/(sig**2)-np.tan(inc.value))

for rat in rats[job_nums(size,rats.shape[0],rank)]:
	try:
		time_start=perf_counter()
		S_par=np.array([np.tan(inc.value)*np.sqrt(A*R)*np.exp(1./2.)*rat,np.sqrt(A*R)])

		zmax=100*S_par[1]
		zmin=-100*S_par[1]
		x0_crit,z_list=Lens_Mod.zbnd_find(zmin,zmax,10000,sheet_dir,sheet,S_par,inc)

		#x=(np.concatenate((np.linspace(-100,-3,100),np.linspace(-3,3,100)[1:-1],np.linspace(3,100,100)))*u.mas).to(u.rad).value*Ds.value
		x=(np.concatenate((np.linspace(-30,-3,10),np.linspace(-3,.5,10)[1:],np.linspace(.5,1.5,10)[1:],np.linspace(1.5,3,10)[1:-1],np.linspace(3,30,10)))*(u.mas.to(u.rad))*Ds).astype('float128')

		print('(%s) %s I Calc Start at %s' %(rank,rat,MPI.Wtime()-ts))

		I=Lens_Mod.I_calc(x.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig)

		print('(%s) %s Rough Grid Complete at %s' %(rank,rat,MPI.Wtime()-ts))

		x2,I2=Lens_Mod.res_improve(1e-5,x,I,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig)

		np.save('./Sims-%s/x-%s' %(par_number,rat) , x2.value)
		np.save('./Sims-%s/I-%s' %(par_number,rat) , I2)
		plt.figure()
		plt.plot(x2,I2)
		plt.plot(x,I)
		plt.savefig('./Sims-%s/Thickness-%s.png' %(par_number,rat))
		plt.close('all')
	#	np.save('E-%s' %rat , Err)

		time_end=perf_counter()
		print('(%s) %s Done at %s' %(rank,rat,MPI.Wtime()-ts))
	except:
		print('(%s) %s Cannot Run' %(rank,rat))

