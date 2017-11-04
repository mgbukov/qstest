from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time, diag_ensemble, mean_level_spacing # t_dep measurements
from quspin.tools.Floquet import Floquet, Floquet_t_vec # Floquet Hamiltonian
import numpy as np # generic math functions

L=14 # system size

J=1.0 #hopping term
U=0.5 # interaction
hz=0.9 # inhomogeneous chem potential

A=1.0 # drive amplitude
Omega=10.0 # drive frequency

# construct spin basis
basis=spin_basis_1d(L=L,Nup=L/2,pzblock=1)

# define site-coupling lists

hop=[[J,i,(i+1)%L] for i in range(L)]
intn=[[U,i,(i+1)%L] for i in range(L)]
z_field=[[hz*(i-(L-1)/2.0),i] for i in range(L)]


def fun(t,A,Omega):
	return A*np.cos(Omega*t)

fun_args=(A,Omega)


static=[['+-',hop],['-+',hop],['zz',intn]]
dynamic=[['z',z_field,fun,fun_args]]

# define Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

print H.Ns

evo_dict={'H':H,'T':2.0*np.pi/Omega}
F=Floquet(evo_dict)

print F.EF



#E,V=H.eigh()

#r = mean_level_spacing(E)
#print r

