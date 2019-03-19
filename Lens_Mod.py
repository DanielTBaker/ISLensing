import numpy as np
from scipy.optimize import brenth,fmin
import astropy.units as u
from scipy.integrate import quad,cumtrapz,quadrature
from scipy.special import lambertw as W
from astropy import constants as const

##Integral of density along line of sight due to gaussian centered at the point parameterized by z on the sheet
def igrand(z,x0,inc,sig,S_par,sheet,sheet_dl):
	return(np.exp(-np.power(dist(z,x0,inc,S_par,sheet)/sig,2))*sheet_dl(z,*S_par))

##x separation between LoS and sheet for a given z
def dist(zp,x0,inc,S_par,sheet):
    return(np.cos(inc)*(x0-sheet(zp,*S_par))+np.sin(inc)*zp)

def I_calc_indiv(tasks):
	x,inc,sig,x0_crit,z_list,S_par,zmin,zmax,sheet,sheet_dl = tasks
	I=0
	x0=x/np.cos(inc.value)
	##x intercept of boundry curves
	x0_l=x0+(5*sig/np.cos(inc.value))
	x0_h=x0-(5*sig/np.cos(inc.value))
	pairs=list()
	##If x0_l is smaller than the first x0 s.t los is tangent to sheet, then it intersects only once
	if x<x0_crit[0]:
		x_max=sheet(z_list[0].max(),*S_par)
		lower=z_list[0].max()
		upper=max(((x_max-x0)/np.tan(inc.value),zmax))
		pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
	elif x0>x0_crit[-1]:
		x_max=sheet(z_list[-1].min(),*S_par)
		lower=min(((x_max-x0_l)/np.tan(inc.value),zmin))
		upper=z_list[-1].min()
		pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
	else:
		ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0].max(),:][0,:]
		full=np.concatenate((ar1,ar2))
		int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
		full=full[full.argsort()]
		diff=np.diff(int_set)
		ints=full[1:][np.abs(diff)>0]
		ints=np.concatenate((full[:1],ints))
		for i in range(ints.shape[0]-1):
			pairs.append((ints[i],ints[i+1]))
	ints_low=np.zeros(len(pairs),dtype='float128')
	for i in range(len(pairs)):
		ints_low[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0_l,inc,S_par,sheet))
	pairs=list()
	if x0_h<x0_crit[0]:
		x_max=sheet(z_list[0].max(),*S_par)
		lower=z_list[0].max()
		upper=max(((x_max-x0_h)/np.tan(inc.value),zmax))
		pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
	elif x0_h>x0_crit[-1]:
		x_max=sheet(z_list[-1].min(),*S_par)
		lower=min(((x_max-x0_h)/np.tan(inc.value),zmin))
		upper=z_list[-1].min()
		pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
	else:
		ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0_h].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0_h].max(),:][0,:]
		full=np.concatenate((ar1,ar2))
		int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
		full=full[full.argsort()]
		diff=np.diff(int_set)
		ints=full[1:][np.abs(diff)>0]
		ints=np.concatenate((full[:1],ints))
		for i in range(ints.shape[0]-1):
			pairs.append((ints[i],ints[i+1]))
	ints_high=np.zeros(len(pairs),dtype='float128')
	for i in range(len(pairs)):
		ints_high[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0_h,inc,S_par,sheet))
	pairs=list()
	if x0<x0_crit[0]:
		x_max=sheet(z_list[0].max(),*S_par)
		lower=z_list[0].max()
		upper=max(((x_max-x0)/np.tan(inc.value),zmax))
		pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
	elif x0>x0_crit[-1]:
		x_max=sheet(z_list[-1].min(),*S_par)
		lower=min(((x_max-x0)/np.tan(inc.value),zmin))
		upper=z_list[-1].min()
		pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
	else:
		ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0].max(),:][0,:]
		full=np.concatenate((ar1,ar2))
		int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
		full=full[full.argsort()]
		diff=np.diff(int_set)
		ints=full[1:][np.abs(diff)>0]
		ints=np.concatenate((full[:1],ints))
		for i in range(ints.shape[0]-1):
			pairs.append((ints[i],ints[i+1]))
	ints=np.zeros(len(pairs),dtype='float128')
	for i in range(len(pairs)):
		ints[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0,inc,S_par,sheet))
	ints_tot=np.concatenate((ints_low,ints_high,ints))
	for i in range(len(z_list)):
		ints_tot=np.concatenate((ints_tot,z_list[i]))
	ints_tot=np.concatenate((np.array([min((zmin,ints_tot.max()-2*(ints_tot.max()-ints_tot.min()))),max((zmax,ints_tot.min()+2*(ints_tot.max()-ints_tot.min())))]),ints_tot))
	ints_tot=np.unique(ints_tot)
	for i in range(ints_tot.shape[0]//2):
		I+=quadrature(igrand,ints_tot[::2][i],ints_tot[1::2][i],args=(x0,inc,sig,S_par,sheet,sheet_dl))[0]
	return(I)

def I_calc_mpi(x,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig,pool):
	tasks=list((x[i],inc,sig,x0_crit,z_list,S_par,zmin,zmax,sheet,sheet_dl) for i in range(x.shape[0]))
	return(np.array(pool.map(I_calc_indiv,tasks)))

##Determine effective thickness along LoS for point x
def I_calc(x,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig):
	I=np.zeros((x.shape[0]),dtype='float128')
	for idx in range(I.shape[0]):
		##x intercept for los
		x0=x[idx]/np.cos(inc.value)
		##x intercept of boundry curves
		x0_l=x0+(100*sig/np.cos(inc.value))
		x0_h=x0-(100*sig/np.cos(inc.value))
		pairs=list()
		##If x0_l is smaller than the first x0 s.t los is tangent to sheet, then it intersects only once
		if x0_l<x0_crit[0]:
			x_max=sheet(z_list[0].max(),*S_par)
			lower=z_list[0].max()
			upper=max(((x_max-x0_l)/np.tan(inc.value),zmax))
			pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
		elif x0_l>x0_crit[-1]:
			x_max=sheet(z_list[-1].min(),*S_par)
			lower=min(((x_max-x0_l)/np.tan(inc.value),zmin))
			upper=z_list[-1].min()
			pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
		else:
			ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0_l].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0_l].max(),:][0,:]
			full=np.concatenate((ar1,ar2))
			int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
			full=full[full.argsort()]
			diff=np.diff(int_set)
			ints=full[1:][np.abs(diff)>0]
			ints=np.concatenate((full[:1],ints))
			pairs=list()
			for i in range(ints.shape[0]-1):
				pairs.append((ints[i],ints[i+1]))
		ints_low=np.zeros(len(pairs),dtype='float128')
		for i in range(len(pairs)):
			ints_low[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0_l,inc,S_par,sheet))
		pairs=list()
		if x0_h<x0_crit[0]:
			x_max=sheet(z_list[0].max(),*S_par)
			lower=z_list[0].max()
			upper=max(((x_max-x0_h)/np.tan(inc.value),zmax))
			pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
		elif x0_h>x0_crit[-1]:
			x_max=sheet(z_list[-1].min(),*S_par)
			lower=min(((x_max-x0_h)/np.tan(inc.value),zmin))
			upper=z_list[-1].min()
			pairs.append(((2*lower-((upper+lower)/2),2*upper-((upper+lower)/2))))
		else:
			ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0_l].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0_l].max(),:][0,:]
			full=np.concatenate((ar1,ar2))
			int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
			full=full[full.argsort()]
			diff=np.diff(int_set)
			ints=full[1:][np.abs(diff)>0]
			ints=np.concatenate((full[:1],ints))
			pairs=list()
			for i in range(ints.shape[0]-1):
				pairs.append((ints[i],ints[i+1]))
		ints_high=np.zeros(len(pairs),dtype='float128')
		for i in range(len(pairs)):
			ints_high[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0_h,inc,S_par,sheet))
		ints_tot=np.sort(np.concatenate((ints_low,ints_high)))
		for i in range(ints_tot.shape[0]//2):
			I[idx]+=quad(igrand,ints_tot[::2][i],ints_tot[1::2][i],args=(x0,inc,sig,S_par,sheet,sheet_dl),epsrel=1.e-16)[0]
	return(I)
##Calulate image positions (beta,theta) and give rough estimate of caustic locations
def Im_find(x,I,ne,delta_ne,om,Ds,s):
	ne=ne.to(1/x.unit**3)
	delta_ne=delta_ne.to(1/x.unit**3)
	nx=x.shape[0]
	lam=(const.c.to(x.unit/u.s))/om
	re=(2.8179403229e-15*u.m).to(x.unit)
	alpha=-s*(lam[:,np.newaxis]**2)*re/(2*np.pi)*np.gradient(I,x.value)[np.newaxis,:]*delta_ne
	theta_AR=((x/Ds))
	beta_AR=(alpha+theta_AR)*(u.rad.to(u.mas))
	theta_AR=((x/Ds))*(u.rad.to(u.mas))
	return(theta_AR,beta_AR)


def res_improve(err,x,I,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig):
	I2=np.copy(I)
	x2=np.copy(x)*x.unit

	idx_list = list()
	for i in range(x2.shape[0]-2):
		if np.abs(I2[i+1]-(I2[i]+(I2[i+2]-I2[i])*(x2[i+1]-x2[i])/(x2[i+2]-x2[i])))/I2[i+1]>err:
			idx_list.append(i+1)
	idx=np.array(idx_list,dtype=int)
	x_new=np.zeros(2*idx.shape[0])
	x_new[::2]=(x2[idx+1].value+x2[idx].value)/2
	x_new[1::2]=(x2[idx-1]+x2[idx])/2
	x_new*=x2.unit
	x_new=np.unique(x_new)
	I_new=I_calc(x_new.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig)

	x2=np.concatenate((x2.value,x_new.value))*x2.unit
	I2=np.concatenate((I2,I_new))
	I2=I2[x2.argsort()]
	x2=x2[x2.argsort()]
	while idx.shape[0]>0:
		idx_list = list()
		for i in range(x2.shape[0]-2):
			if np.abs(I2[i+1]-(I2[i]+(I2[i+2]-I2[i])*(x2[i+1]-x2[i])/(x2[i+2]-x2[i])))/I2[i+1]>err:
				idx_list.append(i+1)
		idx=np.array(idx_list,dtype=int)
		x_new=np.zeros(2*idx.shape[0])
		x_new[::2]=(x2[idx+1].value+x2[idx].value)/2
		x_new[1::2]=(x2[idx-1]+x2[idx])/2
		x_new*=x2.unit
		x_new=np.unique(x_new)
		I_new=I_calc(x_new.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig)

		x2=np.concatenate((x2.value,x_new.value))*x2.unit
		I2=np.concatenate((I2,I_new))
		I2=I2[x2.argsort()]
		x2=x2[x2.argsort()]
	return(x2,I2)

def res_improve_mpi(err,x,I,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig,pool,size):
	I2=np.copy(I)
	x2=np.copy(x)*x.unit

	idx_list = list()
	for i in range(x2.shape[0]-2):
		if np.abs(I2[i+1]-(I2[i]+(I2[i+2]-I2[i])*(x2[i+1]-x2[i])/(x2[i+2]-x2[i])))/I2[i+1]>err:
			idx_list.append(i+1)
	idx=np.array(idx_list,dtype=int)
	print(idx.shape[0])
	x_new=np.zeros(2*idx.shape[0])
	x_new[::2]=(x2[idx+1].value+x2[idx].value)/2
	x_new[1::2]=(x2[idx-1]+x2[idx])/2
	x_new*=x2.unit
	x_new=np.unique(x_new)
	I_new=I_calc_mpi(x_new.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig,pool)

	x2=np.concatenate((x2.value,x_new.value))*x2.unit
	I2=np.concatenate((I2,I_new))
	I2=I2[x2.argsort()]
	x2=x2[x2.argsort()]
	iters=0
	while idx.shape[0]>0 and iters<20:
		idx_list = list()
		for i in range(x2.shape[0]-2):
			if np.abs(I2[i+1]-(I2[i]+(I2[i+2]-I2[i])*(x2[i+1]-x2[i])/(x2[i+2]-x2[i])))/I2[i+1]>err:
				idx_list.append(i+1)
		idx=np.array(idx_list,dtype=int)
		x_new=np.zeros(2*idx.shape[0])
		x_new[::2]=(x2[idx+1].value+x2[idx].value)/2
		x_new[1::2]=(x2[idx-1]+x2[idx])/2
		x_new*=x2.unit
		x_new=np.unique(x_new)
		I_new=I_calc_mpi(x_new.value,sheet,sheet_dl,S_par,zmax,zmin,inc,x0_crit,z_list,sig,pool)
		x2=np.concatenate((x2.value,x_new.value))*x2.unit
		I2=np.concatenate((I2,I_new))
		x2,idx_unique=np.unique(x2,return_index=True)
		I2=I2[idx_unique]
		iters+=1
		x2=x2[I2>I2[0]/1e10]
		I2=I2[I2>I2[0]/1e10]
#		I2=I2[x2.argsort()]
#		x2=x2[x2.argsort()]
	return(x2,I2)
##Improve resolution around all caustics 
def caustic_res(x,I,beta_AR,theta_AR,rough,sheet,sheet_dl,zmax,zmin,S_par,inc,x0_crit,z_list,T,sig,ne,delta_ne,om,Ds,s,W,E_frac):
	ne=ne.to(1/x.unit**3)
	delta_ne=delta_ne.to(1/x.unit**3)
	nx=x.shape[0]
	lam=(const.c.to(x.unit/u.s))/om
	re=(2.8179403229e-15*u.m).to(x.unit)
	omp=np.sqrt(4*np.pi*re*(ne+delta_ne))*(const.c.to(x.unit/u.s))
	n=np.sqrt(1-(omp/om)**2)
	n0=np.sqrt(1-(np.sqrt(4*np.pi*re*ne)*(const.c.to(x.unit/u.s))/om)**2)
	x2=x.value
	I2=np.copy(I)
	Err=np.zeros(2*(rough.shape[0]+1))
	for turn_idx in range(rough.shape[0]):
		if turn_idx==0:
			x_low=x.value[:rough[turn_idx]]
			beta_low=beta_AR[:rough[turn_idx]]
			theta_low=theta_AR[:rough[turn_idx]]
		else:
			x_low=x.value[rough[turn_idx-1]:rough[turn_idx]]
			beta_low=beta_AR[rough[turn_idx-1]:rough[turn_idx]]
			theta_low=theta_AR[rough[turn_idx-1]:rough[turn_idx]]
		mu_low=np.gradient(theta_low,beta_low)
		f0_low=np.abs(mu_low[np.abs(beta_low[-1]-beta_low)>W]).max()
		muI_low=cumtrapz(np.abs(mu_low),-np.abs(beta_low-beta_low[-1]))
		Err_low=((muI_low[-1]-muI_low[-2])**2)/(2*muI_low[-2]-muI_low[-4]-muI_low[-1])
		if turn_idx==rough.shape[0]-1:
			x_high=x.value[rough[turn_idx]:]
			beta_high=beta_AR[rough[turn_idx]:]  
			theta_high=theta_AR[rough[turn_idx]:]
		else:
			x_high=x.value[rough[turn_idx]:rough[turn_idx+1]]
			beta_high=beta_AR[rough[turn_idx]:rough[turn_idx+1]]  
			theta_high=theta_AR[rough[turn_idx]:rough[turn_idx+1]]
		mu_high=np.gradient(theta_high,beta_high)
		f0_high=np.abs(mu_high[np.abs(beta_high[-1]-beta_high)>W]).max()
		muI_high=cumtrapz(np.abs(mu_high)[::-1],-np.abs(beta_high[::-1]-beta_high[0]))
		Err_high=((muI_high[-1]-muI_high[-2])**2)/(2*muI_high[-2]-muI_high[-4]-muI_high[-1])
		print('F0 High: %s' %f0_high)
		print('F0 Low: %s' %f0_low)
		x3=np.array([])
		I3=np.array([])
		while Err_high>E_frac*(W-np.abs(beta_high[1]-beta_high[0]))*f0_high/(1-E_frac) or Err_low>E_frac*(W-np.abs(beta_low[-1]-beta_low[-2]))*f0_low/(1-E_frac) or np.abs(beta_low[-1]-beta_low[-2])>W/2 or np.abs(beta_high[1]-beta_high[0])>W/2:
			if np.abs(beta_low[-1]-beta_low[-2])>W/2:
				print('Low db: %s' %np.abs(beta_low[-1]-beta_low[-2]))
				print('Goal: %s' %(W/2))
			elif Err_low>E_frac*(W-np.abs(beta_low[-1]-beta_low[-2]))*f0_low/(1-E_frac):
				print('Err Low: %s' %Err_low)
				print('Err Low Target: %s' %(E_frac*(W-np.abs(beta_low[-1]-beta_low[-2]))*f0_low/(1-E_frac)))
			if np.abs(beta_high[1]-beta_high[0])>W/2:
				print('High db: %s' %np.abs(beta_high[1]-beta_high[0]))
				print('Goal: %s' %(W/2))
			elif Err_high>E_frac*(W-np.abs(beta_high[1]-beta_high[0]))*f0_high/(1-E_frac):
				print('Err High: %s' %Err_high)
				print('Err High Target: %s' %(E_frac*(W-np.abs(beta_high[1]-beta_high[0]))*f0_high/(1-E_frac)))
			print('')
			x_new=np.linspace(x_low.max(),x_high.min(),10)[1:-1]
			I_new=np.zeros(x_new.shape[0])
			for idx in range(I_new.shape[0]):
				x0=x_new[idx]/np.cos(inc.value)
				x0_l=x0+(100*sig/np.cos(inc.value))
				x0_h=x0-(100*sig/np.cos(inc.value))
				pairs=list()
				if x0_l<x0_crit[0]:
					x_max=sheet(z_list[0].max(),*S_par)
					pairs.append((z_list[0].max(),max(((x_max-x0_l)/np.tan(inc.value),zmax))))
				elif x0_l>x0_crit[-1]:
					x_max=sheet(z_list[-1].min(),*S_par)
					pairs.append((min(((x_max-x0_l)/np.tan(inc.value),zmin)),z_list[0].min()))
				else:
					ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0_l].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0_l].max(),:][0,:]
					full=np.concatenate((ar1,ar2))
					int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
					full=full[full.argsort()]
					diff=np.diff(int_set)
					ints=full[1:][np.abs(diff)>0]
					ints=np.concatenate((full[:1],ints))
					pairs=list()
					for i in range(ints.shape[0]-1):
						pairs.append((ints[i],ints[i+1]))
				ints_low=np.zeros(len(pairs))
				for i in range(len(pairs)):
					ints_low[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0_l,inc,S_par,sheet))
				pairs=list()
				if x0_h<x0_crit[0]:
					x_max=sheet(z_list[0].max(),*S_par)
					pairs.append((z_list[0].max(),max(((x_max-x0_h)/np.tan(inc.value),zmax))))
				elif x0_h>x0_crit[-1]:
					x_max=sheet(z_list[-1].min(),*S_par)
					pairs.append((min(((x_max-x0_h)/np.tan(inc.value),zmin)),z_list[0].min()))
				else:
					ar1,ar2=np.array(z_list)[x0_crit==x0_crit[x0_crit>x0_h].min(),:][0,:],np.array(z_list)[x0_crit==x0_crit[x0_crit<x0_h].max(),:][0,:]
					full=np.concatenate((ar1,ar2))
					int_set=np.concatenate((np.ones(ar1.shape[0]),np.ones(ar2.shape[0])*2))[full.argsort()]
					full=full[full.argsort()]
					diff=np.diff(int_set)
					ints=full[1:][np.abs(diff)>0]
					ints=np.concatenate((full[:1],ints))
					pairs=list()
					for i in range(ints.shape[0]-1):
						pairs.append((ints[i],ints[i+1]))
				ints_high=np.zeros(len(pairs))
				for i in range(len(pairs)):
					ints_high[i]=brenth(dist,pairs[i][0],pairs[i][1],args=(x0_h,inc,S_par,sheet))
				ints_tot=np.sort(np.concatenate((ints_low,ints_high)))
				for i in range(ints_tot.shape[0]//2):
					I_new[idx]+=quad(igrand,ints_tot[::2][i],ints_tot[1::2][i],args=(x0,inc,sig,S_par,sheet,sheet_dl),epsrel=1.e-16)[0]
			I_new*=T/(2*np.pi*(sig**2))
			if x3.shape[0]>0:
				I_new=np.concatenate((I3[x3<x_new.min()],I_new,I3[x3>x_new.max()]))
				x_new=np.concatenate((x3[x3<x_new.min()],x_new,x3[x3>x_new.max()]))
			else:
				x_new=np.concatenate((x_low[:-10],x_new,x_high[:10]))
				I_new=np.concatenate((I[np.isin(x.value,x_low)][:-10],I_new,I[np.isin(x.value,x_high)][:10]))
			I_new=I_new[x_new.argsort()]
			x_new=x_new[x_new.argsort()]
			alpha_new=(((-(n-n0)*(np.gradient(I_new,x_new)))))
			beta_new=(x_new/Ds.value+s*alpha_new)*(u.rad.to(u.mas))
			theta_new=(x_new/Ds.value)*(u.rad.to(u.mas))
			grad_new=np.gradient(beta_new,theta_new,edge_order=2)[2:-2]
			beta_new=beta_new[2:-2]
			theta_new=theta_new[2:-2]
			x_new=x_new[2:-2]
			I_new=I_new[2:-2]
			if grad_new[0]>0:
				x_low=x_new[x_new<x_new[grad_new<0].min()]
				x_high=x_new[x_new>x_new[grad_new>0].max()]
				beta_low=beta_new[x_new<x_new[grad_new<0].min()]
				beta_high=beta_new[x_new>x_new[grad_new>0].max()]
				theta_low=theta_new[x_new<x_new[grad_new<0].min()]
				theta_high=theta_new[x_new>x_new[grad_new>0].max()]
			else:
				x_low=x_new[x_new<x_new[grad_new>0].min()]
				x_high=x_new[x_new>x_new[grad_new<0].max()]
				beta_low=beta_new[x_new<x_new[grad_new>0].min()]
				beta_high=beta_new[x_new>x_new[grad_new<0].max()]
				theta_low=theta_new[x_new<x_new[grad_new>0].min()]
				theta_high=theta_new[x_new>x_new[grad_new<0].max()]
			mu=1./np.gradient(beta_new,theta_new)
			mu_low=mu[np.isin(x_new,x_low)]
			#mu_low=np.gradient(theta_low,beta_low,edge_order=2)
			muI_low=cumtrapz(np.abs(mu_low),-np.abs(beta_low-beta_low[-1]))
			Err_low=((muI_low[-1]-muI_low[-2])**2)/(2*muI_low[-2]-muI_low[-4]-muI_low[-1])
			#mu_high=np.gradient(theta_high,beta_high)
			mu_high=mu[np.isin(x_new,x_high)]
			muI_high=cumtrapz(np.abs(mu_high)[::-1],-np.abs(beta_high[::-1]-beta_high[0]))
			Err_high=((muI_high[-1]-muI_high[-2])**2)/(2*muI_high[-2]-muI_high[-4]-muI_high[-1])
			I3=np.concatenate((I3,I_new[np.isin(x_new,x3,invert=True)*np.isin(x_new,x_high)],I_new[np.isin(x_new,x3,invert=True)*np.isin(x_new,x_low)]))
			x3=np.concatenate((x3,x_new[np.isin(x_new,x3,invert=True)*np.isin(x_new,x_high)],x_new[np.isin(x_new,x3,invert=True)*np.isin(x_new,x_low)]))
			I3=I3[x3.argsort()]
			x3=x3[x3.argsort()]
		x2=np.concatenate((x2,x3))
		I2=np.concatenate((I2,I3))
		Err[2*turn_idx+1]=Err_low
		Err[2*turn_idx+2]=Err_high
	I2=I2[np.argsort(x2)]
	x2=x2[np.argsort(x2)]
	return(x2,I2,Err)

def zbnd_find(zmin,zmax,nz,sheet_dir,sheet,S_par,inc):
	z0=np.linspace(zmin,zmax,nz)
	grad=sheet_dir(z0,inc,*S_par)
	z_list=list()
	idx=np.linspace(0,z0.shape[0]-1,z0.shape[0]).astype(int)
	idx2=np.linspace(0,38,39).astype(int)
	if grad.max()>0 and grad.min()<0:
		rough=idx[np.abs(np.concatenate((np.zeros(1),np.diff(np.sign(grad)))))==2]
		z0_rough=z0[rough]
		z_crit=np.zeros(z0_rough.shape)
		for i in range(z_crit.shape[0]):
			z_crit[i]=brenth(sheet_dir,z0[rough[i]-1],z0[rough[i]],args=(inc,*S_par))
		x_crit=sheet(z_crit,*S_par)
		x0_crit=x_crit-np.tan(inc.value)*z_crit
	elif np.std(grad)==0:
		x0_crit=np.zeros(1)
		z_crit=np.zeros(1)
		rough=np.array([idx[z0>z_crit].min()])
	elif grad.max()<0:
		z_peak=fmin(lambda x: -sheet_dir(x,inc,*S_par),z0[grad==grad.max()][0])[0]
		if sheet_dir(z_peak,inc,*S_par)>0:
			z_crit=np.zeros(2)
			z_crit[0]=brenth(sheet_dir,z0[z0<z_peak].max(),z_peak,args=(inc,*S_par))
			z_crit[1]=brenth(sheet_dir,z_peak,z0[z0>z_peak].min(),args=(inc,*S_par))
			z_crit=np.unique(z_crit)
			x_crit=sheet(z_crit,*S_par)
			x0_crit=x_crit-np.tan(inc.value)*z_crit
			x0_crit=np.unique(x0_crit)
			if x0_crit.shape[0]==1:
				z_crit=z_crit[:1]
			z0=np.concatenate((np.linspace(z0[z0<z_peak].max(),z_peak,20),np.linspace(z_peak,z0[z0>z_peak].min(),20)[1:]))
			grad=sheet_dir(z0,inc,*S_par)
			rough=idx2[np.abs(np.concatenate((np.zeros(1),np.diff(np.sign(grad)))))==2]
		else:
			z_crit=np.array([z_peak])
			x_crit=sheet(z_crit,*S_par)
			x0_crit=x_crit-np.tan(inc.value)*z_crit
			rough=np.array([idx[z0>z_crit].min()])
	elif grad.min()>0:
		z_peak=fmin(lambda x: sheet_dir(x,inc,*S_par),z0[grad==grad.max()][0])[0]
		if sheet_dir(z_peak,inc,*S_par)<0:
			z_crit=np.zeros(2)
			z_crit[0]=brenth(sheet_dir,z0[z0<z_peak].max(),z_peak,args=(inc,*S_par))
			z_crit[1]=brenth(sheet_dir,z_peak,z0[z0>z_peak].min(),args=(inc,*S_par))
			z_crit=np.unique(z_crit)
			x_crit=sheet(z_crit,*S_par)
			x0_crit=x_crit-np.tan(inc.value)*z_crit
			x0_crit=np.unique(x0_crit)
			if x0_crit.shape[0]==1:
				z_crit=z_crit[:1]	
			z0=np.concatenate((np.linspace(z0[z0<z_peak].max(),z_peak,20),np.linspace(z_peak,z0[z0>z_peak].min(),20)[1:]))
			grad=sheet_dir(z0,inc,*S_par)
			rough=idx2[np.abs(np.concatenate((np.zeros(1),np.diff(np.sign(grad)))))==2]
		else:
			z_crit=np.array([z_peak])
			x_crit=sheet(z_crit,*S_par)
			x0_crit=x_crit-np.tan(inc.value)*z_crit
			rough=np.array([idx[z0>z_crit].min()])
	TGT=np.array([np.ones(z0.shape[0]),np.gradient(sheet(z0,*S_par),z0)])
	NORM=np.array([np.ones(z0.shape[0])*np.sin(inc.value),-np.ones(z0.shape[0])*np.cos(inc.value)])
	dot=(TGT*NORM).sum(0)
	type_crit=np.sign(dot[rough-1])
	type_crit=type_crit[x0_crit.argsort()].astype(int)
	if type_crit.shape[0]==1:
		type_crit[0]=0
	z_crit=z_crit[x0_crit.argsort()]
	x0_crit=x0_crit[x0_crit.argsort()]
	for i in range(z_crit.shape[0]):
		n_cross=1+(type_crit[:i].sum()*2)+type_crit[i]
		cross=np.zeros(n_cross)
		cross[0]=z_crit[i]
		bnds=np.sort(np.concatenate((np.array([zmin,zmax]),np.array([-100*x0_crit[0],-10*x0_crit[-1]])/np.tan(inc.value),z_crit[:i],z_crit[i+1:])))
		k=1
		for j in range(bnds.shape[0]-1):
			if z_crit[i]<bnds[j] or z_crit[i]>bnds[j+1]:
				z_temp=np.linspace(bnds[j],bnds[j+1],nz)
				rough=np.linspace(0,z_temp.shape[0]-1,z_temp.shape[0])[np.abs(np.concatenate((np.zeros(1),np.diff(np.sign(dist(z_temp,x0_crit[i],inc,S_par,sheet))))))==2].astype(int)
				if rough.shape[0]>0:
					cross[k]=brenth(dist,z_temp[rough[0]-1],z_temp[rough[0]],args=(x0_crit[i],inc,S_par,sheet))
					k+=1
		z_list.append(cross)
	return(x0_crit,z_list)
