from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian,exp_op # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
from quspin.tools.Floquet import Floquet,Floquet_t_vec # Floquet Hamiltonian
from quspin.basis.photon import coherent_state # HO coherent state
import numpy as np # generic math functions
import sys
#
##### define model parameters #####
Nph_max=0 #22 # maximum photon occupation 
Nph=0 #18
L=4
Jzz=1.0
hz=0.809
hx=0.9045
Omega=10.0 # drive frequency
nT=1
nT0=50
A=hz # spin-photon coupling strength (drive amplitude) 

def drive(t,Omega):
	return np.sin(Omega*t)
#
##### set up photon-atom Hamiltonian #####
# define operator site-coupling lists
ph_energy=[[Omega]] # photon energy

absorb=[[A/(2.0*np.sqrt(Nph)),i] for i in range(L)] # absorption term	
emit=[[A/(2.0*np.sqrt(Nph)),i] for i in range(L)] # emission term

z_field=[[hz,i] for i in range(L)] # atom energy
x_field=[[hx,i] for i in range(L)] # atom energy
J_nn=[[Jzz,i,(i+1)%L] for i in range(L)] # atom energy

# define static and dynamics lists
static=[["|n",ph_energy],["x|-",absorb],["x|+",emit],["z|",z_field],["x|",x_field],["zz|",J_nn]]
static_s=[["z|",z_field],["x|",x_field],["zz|",J_nn]]
# compute atom-photon basis
#basis=photon_basis(spin_basis_1d,L=L,Nph=Nph_max)
basis_full=photon_basis(spin_basis_1d,L=L,Nph=Nph_max)
basis_full_sp=spin_basis_1d(L=L)

basis=photon_basis(spin_basis_1d,L=L,Nph=Nph_max,kblock=0)
basis_sp=spin_basis_1d(L=L,kblock=0)

#basis=photon_basis(spin_basis_1d,L=L,Nph=Nph_max)
#basis_sp=spin_basis_1d(L=L)

P = basis.get_proj(dtype=np.complex128)
P_sc = basis_sp.get_proj(dtype=np.complex128)

# compute atom-photon Hamiltonian H
H = hamiltonian(static,[],dtype=np.float64,basis=basis)
H_s = hamiltonian(static_s,[],dtype=np.float64,basis=basis)
print("full H-space dim", H.Ns)
#
static_sp=[["z",z_field],["x",x_field],["zz",J_nn]]
dynamic_sp=[["x",x_field,drive,[Omega]]]



H_sp=hamiltonian(static_sp,dynamic_sp,dtype=np.float64,basis=basis_sp) 

print("spin H-space dim", H_sp.Ns)

#### define observables #####
# in atom-photon Hilbert space
ph_args={"basis":basis_full,"check_symm":False,"check_herm":False,"check_pcon":False,"dtype":np.float64}
sc_args={"basis":basis_full_sp,"check_symm":False,"check_herm":False,"check_pcon":False,"dtype":np.float64}

n =hamiltonian([["|n",[[1.0, ]] ]],[],basis=basis)

sz_ph_list=[hamiltonian([["z|",[[1.0,i]] ]],[],**ph_args) for i in range(L)]
sz_ph_list2=[hamiltonian([["z|",[[1.0,i]] ]],[],**ph_args).project_to(P) for i in range(L)]
sz_sc_list=[hamiltonian([["z",[[1.0,i]] ]],[],**sc_args).project_to(P_sc) for i in range(L)]

print np.around((P*P.T).todense())
#a = (P.T.todense().dot(sz_ph_list[0].astype(np.complex128).todense())).dot(P.todense())
#print np.around(a,2) 
exit()
print sz_ph_list[0]# (sz_ph_list[0]*sz_ph_list[0]).project_to(P)
print sz_ph_list2[0]#*sz_ph_list2[0] 
exit()


##### define initial state #####
# define atom ground state
E_sp,V_sp=H_sp.eigsh(k=2,time=0,which='BE',maxiter=1E10)
W = np.diff(E_sp).squeeze()
psi_sp_i=V_sp[:,0].ravel()
print("spin MB bandwidth is %s" %(W) )

# define photon Flock state containing Nph_max photons
psi_ph_i=np.zeros((Nph_max+1,),dtype=np.float64)
psi_ph_i[Nph]=1.0
#psi_ph_i = coherent_state(np.sqrt(Nph),Nph_max+1)
#psi_ph_i /= np.linalg.norm(psi_ph_i)
# compute atom-photon initial state as a tensor product
psi_i=np.kron(psi_sp_i,psi_ph_i)

#
##### calculate time evolution #####
# define time vector over 100 driving cycles with 100 points per period
t=Floquet_t_vec(Omega,nT,len_T=10) # t.i = initial time, t.T = driving period 
t0=nT0*t.T # time from which correlator is measured
# evolve atom-photon state up to time t0

psi_sc_t0 = H_sp.evolve(psi_sp_i,0,t0)
psi_ph_t0 = exp_op(H,a=-1j*t0).dot(psi_i)



n_t0 = n.matrix_ele(psi_ph_t0,psi_ph_t0).real
sz_ph_t0 = np.mean([sz.matrix_ele(psi_ph_t0,psi_ph_t0).real for sz in sz_ph_list])
sz_sc_t0 = np.mean([sz.matrix_ele(psi_sc_t0,psi_sc_t0).real for sz in sz_sc_list])

expH = exp_op(H,a=-1j,start=0,stop=nT*t.T,num=10*nT+1,endpoint=True,iterate=True)

ph_generators = [expH.dot(psi_ph_t0)]
sc_generators = [H_sp.evolve(psi_sc_t0,t0,t,iterate=True)]

npsi = n.dot(psi_ph_t0)
ph_generators.append(expH.dot(npsi))

for sz_sc,sz_ph in zip(sz_sc_list,sz_ph_list):
	spsi_ph = sz_ph.dot(psi_ph_t0)
	spsi_sc = sz_sc.dot(psi_sc_t0)

	ph_generators.append(expH.dot(spsi_ph))
	sc_generators.append(H_sp.evolve(spsi_sc,t0,t,iterate=True))

ph_site = len(sz_ph_list)
sc_site = len(sz_sc_list)



O_n = np.zeros((len(t),),dtype=np.float64)
nn = np.zeros((len(t),),dtype=np.complex128)

E_ph = np.zeros_like(O_n)
E_sc = np.zeros_like(O_n)

SzSz_ph = np.zeros((ph_site,len(t)),dtype=np.complex128)
O_sz_ph = np.zeros_like(SzSz_ph,dtype=np.float64)

SzSz_sc = np.zeros((sc_site,len(t)),dtype=np.complex128)
O_sz_sc = np.zeros_like(SzSz_sc,dtype=np.float64)


for i,psi_ph,psi_sc in zip(range(len(t)),zip(*ph_generators),zip(*sc_generators)):
	print("time: ",expH.grid[i]+t0)
	O_n[i]=n.matrix_ele(psi_ph[0],psi_ph[0]).real
	nn[i]=n.matrix_ele(psi_ph[0],psi_ph[1])

	E_sc[i] = H_sp.matrix_ele(psi_sc[0],psi_sc[0]).real/L
	E_ph[i] = H_s.matrix_ele(psi_ph[0],psi_ph[0]).real/L
	for j,spsi,sz in zip(range(ph_site),psi_ph[2:],sz_ph_list):
		O_sz_ph[j,i] = sz.matrix_ele(psi_ph[0],psi_ph[0]).real
		SzSz_ph[j,i] = sz.matrix_ele(psi_ph[0],spsi)

		print SzSz_ph[j,i].real

	for j,spsi,sz in zip(range(sc_site),psi_sc[1:],sz_sc_list):
		O_sz_sc[j,i] = sz.matrix_ele(psi_sc[0],psi_sc[0]).real
		SzSz_sc[j,i] = sz.matrix_ele(psi_sc[0],spsi)



nn -= O_n*n_t0


print SzSz_ph

SzSz_ph = SzSz_ph.mean(axis=0)

#O_sz_ph = O_sz_ph.mean(axis=0)
#SzSz_ph -= sz_ph_t0*O_sz_ph

SzSz_sc = SzSz_sc.mean(axis=0)
O_sz_sc = O_sz_sc.mean(axis=0)
SzSz_sc -= sz_sc_t0*O_sz_sc

##### plot results #####
import matplotlib.pyplot as plt
import pylab
# define legend labels
str_n = "$\\langle n\\rangle/n_0,$"
str_z = "$\\langle\\sigma^z\\rangle,$"
str_zz_Re = "$\\frac{1}{2}\\langle\\{\\sigma^z(t+t_0),\\sigma^z(t_0)\\}\\rangle,$"
str_zz_Im = "$\\langle[\\sigma^z(t+t_0),\\sigma^z(t_0)]\\rangle,$"
str_nn_Re = "$\\frac{1}{2}\\langle\\{n(t+t_0),n(t_0)\\}\\rangle,$"
str_nn_Im = "$\\langle[n(t+t_0),n(t_0)]\\rangle]$"
str_e = "$\\epsilon(t)$"
# plot spin-photon data
fig = plt.figure()
#plt.plot( (t0+t.vals)/t.T,O_n/(Nph),"k",linewidth=1,label=str_n)
#plt.plot( (t0+t.vals)/t.T,O_sz_ph,"b",linewidth=1,label=str_z)
plt.plot( (t0+t.vals)/t.T,SzSz_ph.real,"m",linewidth=1,label=str_zz_Re)
#plt.plot( (t0+t.vals)/t.T,E_ph,"r",linewidth=1,label=str_e)
#plt.plot( (t0+t.vals)/t.T,2.0*SzSz.imag,"--m",linewidth=1,label=str_zz_Im)
#plt.plot( (t0+t.vals)/t.T,nn.real,"c",linewidth=1,label=str_nn_Re)
#plt.plot( (t0+t.vals)/t.T,nn.imag,"--c",linewidth=1,label=str_zz_Re)
# label axes
plt.xlabel("$t/T$",fontsize=18)
# set y axis limits
#plt.ylim([-1.1,1.4])
# display legend horizontally
plt.legend(loc="best") #,ncol=5,columnspacing=0.6,numpoints=4)
# update axis font size
plt.tick_params(labelsize=16)
# turn on grid
plt.grid(True)
# save figure
#fig.savefig('example3.pdf', bbox_inches='tight')
# show plot
plt.title("Quantum")
#plt.figure() 

"""
plt.plot( (t0+t.vals)/t.T,O_sz_sc,"b",linewidth=1,label=str_z)
plt.plot( (t0+t.vals)/t.T,SzSz_sc.real,"m",linewidth=1,label=str_zz_Re)
plt.plot( (t0+t.vals)/t.T,E_sc,"r",linewidth=1,label=str_e)
#plt.plot( (t0+t.vals)/t.T,2.0*SzSz.imag,"--m",linewidth=1,label=str_zz_Im)
#plt.plot( (t0+t.vals)/t.T,nn.real,"c",linewidth=1,label=str_nn_Re)
#plt.plot( (t0+t.vals)/t.T,nn.imag,"--c",linewidth=1,label=str_zz_Re)
# label axes
plt.xlabel("$t/T$",fontsize=18)
# set y axis limits
#plt.ylim([-1.1,1.4])
# display legend horizontally
plt.legend(loc="best") #,ncol=5,columnspacing=0.6,numpoints=4)
# update axis font size
plt.tick_params(labelsize=16)
# turn on grid
plt.grid(True)
# save figure
#fig.savefig('example3.pdf', bbox_inches='tight')
# show plot
plt.title("Semi-Classi")
"""
plt.show() 