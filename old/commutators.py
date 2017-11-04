import sys,os
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian, commutator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time, diag_ensemble # t_dep measurements
from quspin.tools.Floquet import Floquet, Floquet_t_vec # Floquet Hamiltonian
import numpy as np # generic math functions
#
##### define model parameters #####
L=14 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
Omega=4.5 # drive frequency
#
##### set up alternating Hamiltonians #####
# define time-reversal symmetric periodic step drive
def drive(t,Omega):
	return np.sign(np.cos(Omega*t))
drive_args=[Omega]
# compute basis in the 0-total momentum and +1-parity sector
basis=spin_basis_1d(L=L,a=1,kblock=0,pblock=1)
# define PBC site-coupling lists for operators
x_field_pos=[[+g,i]	for i in range(L)]
x_field_neg=[[-g,i]	for i in range(L)]
z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
dynamic=[["zz",J_nn,drive,drive_args],
		 ["z",z_field,drive,drive_args],["x",x_field_neg,drive,drive_args]]
# compute Hamiltonians
H=0.5*hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
#
##### set up second-order van Vleck Floquet Hamiltonian #####
# zeroth-order term
Heff_0=0.5*hamiltonian(static,[],dtype=np.float64,basis=basis)
static=[["zz",J_nn],["z",z_field],["x",x_field_neg]]
Htilde=0.5*hamiltonian(static,[],dtype=np.float64,basis=basis)
# second-order term: site-coupling lists
Heff2_term_1=[[+J**2*g,i,(i+1)%L,(i+2)%L] for i in range(L)] # PBC
Heff2_term_2=[[+J*g*h, i,(i+1)%L] for i in range(L)] # PBC
Heff2_term_3=[[-J*g**2,i,(i+1)%L] for i in range(L)] # PBC
Heff2_term_4=[[+J**2*g+0.5*h**2*g,i] for i in range(L)]
Heff2_term_5=[[0.5*h*g**2,		  i] for i in range(L)]
# define static list
Heff_static=[["zxz",Heff2_term_1],
			 ["xz",Heff2_term_2],["zx",Heff2_term_2],
			 ["yy",Heff2_term_3],["zz",Heff2_term_2],
			 ["x",Heff2_term_4],
			 ["z",Heff2_term_5]							] 
# compute van Vleck Hamiltonian
Heff_2=hamiltonian(Heff_static,[],dtype=np.float64,basis=basis)
Heff_2*=-np.pi**2/(12.0*Omega**2)
# zeroth + second order van Vleck Floquet Hamiltonian
Heff_02=Heff_0+Heff_2
#
##### set up second-order van Vleck Kick operator #####
Keff2_term_1=[[J*g,i,(i+1)%L] for i in range(L)] # PBC
Keff2_term_2=[[h*g,i] for i in range(L)]
# define static list
Keff_static=[["zy",Keff2_term_1],["yz",Keff2_term_1],["y",Keff2_term_2]]
Keff_02=hamiltonian(Keff_static,[],dtype=np.complex128,basis=basis)
Keff_02*=-np.pi**2/(8.0*Omega**2)



comm1 = -np.pi**2/(8.0*Omega**2)*1.0/1j*commutator(Htilde, Heff_0).astype(np.complex128)

print np.linalg.norm( comm1.todense() - Keff_02.todense() )


