import sys,os
qspin_path = os.path.join(os.getcwd(),"../basis_1d_base/")
sys.path.insert(0,qspin_path)

import numpy as np

from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from quspin.tools.measurements import ent_entropy
#from memory_profiler import profile

#@profile
def partial_trace(rho,sub_sys_A,L,sps=2):
	rho = np.atleast_2d(rho)

	if rho.shape[0] != rho.shape[1]:
		raise ValueError("expecting square matrix")

	# calculate subsystem B to be traced out and its length
	sub_sys_A = set(sub_sys_A)
	sub_sys_B = set(range(L))-sub_sys_A

	Ns_B = (sps**len(sub_sys_B))
	Ns_A = (sps**len(sub_sys_A))

	# T_tup tells numpy how to reshuffle the indices such that when I reshape the array to the 
	# 4-Tensor rho_{ik,jl} i,j are for sub_sys_B and k,l are for sub_sys_A
	# which means I need (sub_sys_B,sub_sys_A,sub_sys_B+L,sub_sys_A+L)

	T_tup = tuple(sub_sys_B)+tuple(sub_sys_A) # 
	T_tup += tuple(s+L for s in T_tup)

	print("The following line need to be modified for symemtries")
	# DM where index is given per site as rho_v[i_1,...,i_L,j_1,...j_L]
	rho_v = rho.reshape(tuple(sps for i in range(2*L)))
	# take transpose to reshuffle indices 
	rho_v = rho_v.transpose(T_tup) 
	rho_v = rho_v.reshape((Ns_B,Ns_A,Ns_B,Ns_A)) 

	return np.einsum("ijik->jk",rho_v)



##### define model parameters #####
L=6 # system size

J=1.0 # hopping
U=np.sqrt(2) # interactions strenth


# define site-coupling lists
interaction=[[U/2.0,i,i] for i in range(L)] # PBC
chem_pot=[[-U/2.0,i] for i in range(L)] # PBC
hopping=[[J,i,(i+1)%L] for i in range(L)] # PBC

#### define hcb model
basis = boson_basis_1d(L=L,sps=3)


# Hubbard-related model
static =[["+-",hopping],["-+",hopping],["n",chem_pot],["nn",interaction]]

H=hamiltonian(static,[],basis=basis,dtype=np.float32)
E,V=H.eigh()

psi=V[:,0]
chain_subsys=[0,4]

DM=ent_entropy(psi,basis,DM='chain_subsys',chain_subsys=chain_subsys)['DM_chain_subsys']
DM_pt = partial_trace( np.outer(psi.conj(),psi),chain_subsys,L,basis.sps )


np.testing.assert_allclose(DM-DM_pt,0.0,atol=1E-5,err_msg='Failed testing partial trace!')









