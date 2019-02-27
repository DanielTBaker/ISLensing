import numpy as np
from scipy.optimize import brenth
import astropy.units as u
from scipy.integrate import quad
from scipy.special import lambertw as W
from astropy import constants as const
import Lens_Mod
from scipy.misc import derivative
from time import perf_counter

def sheet(z,A,sig):
    return(A*np.exp(-np.power(z/sig,2)/2))
def sheet_dl(z,A,sig):
    return(np.sqrt(1+np.power(A*z/(sig**2),2)*np.exp(-np.power(z/sig,2))))
def sheet_dir(z,inc,A,sig):
    return(-z*A*np.exp(-np.power(z/sig,2)/2)/(sig**2)-np.tan(inc.value))

T=.03*(u.AU.to(u.m))
A=.3*(u.AU.to(u.m))
R=4.8*(u.kpc.to(u.m))
sig=T/(2*np.sqrt(2*np.log(2)))
#S_par=np.array([A,np.sqrt(A*R)])
inc=1e-5*u.rad

Ds=(389*u.pc).to(u.m)
Dp=(620*u.pc).to(u.m)
Dps=Ds-Dp
s=1-Ds/Dp

ne=(.3/(u.cm**3))
delta_ne=(.003/(u.cm**3))

om=(314.5*u.MHz).to(1/u.s)

W=(sig/Ds.value)*(u.rad.to(u.mas))/5
E_frac=1e-2
for rat in np.array([0]):
	time_start=perf_counter()
	S_par=np.array([np.tan(inc.value)*np.sqrt(A*R)*np.exp(1./2.)*rat,np.sqrt(A*R)])

	zmax=100*S_par[1]
	zmin=-100*S_par[1]
	x0_crit,z_list=Lens_Mod.zbnd_find(zmin,zmax,10000,sheet_dir,sheet,S_par,inc)

	#x=(np.concatenate((np.linspace(-100,-3,100),np.linspace(-3,3,100)[1:-1],np.linspace(3,100,100)))*u.mas).to(u.rad).value*Ds.value
	x=(np.concatenate((np.linspace(-30,-3,10),np.linspace(-3,.5,500)[1:],np.linspace(.5,1.5,5000)[1:],np.linspace(1.5,3,100)[1:-1],np.linspace(3,30,10)))*(u.mas.to(u.rad))*Ds).astype('float128')

	print('Intial I Calculation')
	I=Lens_Mod.I_calc(x.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,T,sig)

	print('Image Boudaries')
	theta_AR,beta_AR,rough=Lens_Mod.Im_find(x,I,ne,delta_ne,om,Ds,s)

	print('Fine Grid I Calculation')
	x2,I2,Err=Lens_Mod.caustic_res(x,I,beta_AR,theta_AR,rough,sheet,sheet_dl,zmax,zmin,S_par,inc,x0_crit,z_list,T,sig,ne,delta_ne,om,Ds,s,W,E_frac)

	#params='%s-%s-%s-%s-%s-%s-%s-%s' %(inc.value,T,sig,frac,S_par[1],om.value,ne.value,delta_ne.value)
	np.save('x-%s' %rat , x2)
	np.save('I-%s' %rat , I2)
	np.save('E-%s' %rat , Err)

	time_end=perf_counter()
	print('%s Done in %s' %(rat,time_end-time_start))

