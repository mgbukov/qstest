import sys,os
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
sys.path.insert(0,qspin_path)

#from quspin.operators import hamiltonian, exp_op # Hamiltonians and operators
from quspin.tools import block_tools as blk
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from numpy.random import ranf,seed # pseudo random numbers
#from joblib import delayed,Parallel # parallelisation
import numpy as np # generic math functions

##### define model parameters #####
L=6 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
Omega=4.5 # drive frequency
#
##### set up alternating Hamiltonians #####
# define time-reversal symmetric periodic step drive
def drive(t,Omega):
	return np.cos(Omega*t)
def drive2(t,Omega):
	return np.sin(Omega*t)	

drive_args=(Omega,)
drive_args2=(Omega,)
# compute basis in the 0-total momentum and +1-parity sector
#basis=spin_basis_1d(L=L,a=1,kblock=0,pblock=1)
# define PBC site-coupling lists for operators
x_field_pos=[[+g,i]	for i in range(L)]
x_field_neg=[[-g,i]	for i in range(L)]
z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
dynamic=[["zz",J_nn,drive,drive_args],
		 ["z",z_field,drive2,drive_args],["x",x_field_neg,drive,drive_args2]]


blocks=[{'kblock':0,'pblock':1},{'kblock':0,'pblock':-1}]
#blocks=[{'kblock':0}]
basis_con=basis=spin_basis_1d
basis_args=(L,)
dtype=np.float64

H_block=blk.block_ops(blocks,static,dynamic,basis_con,basis_args,dtype,save_previous_data=True,compute_all_blocks=True)

psi0=np.eye(2**L)[0] 
psi_t = H_block.evolve(psi0,0,10.0)
psi_t2 = H_block.expm(psi0,H_time_eval=10.0)

P,H = blk.block_diag_hamiltonian(blocks,static,dynamic,basis_con,basis_args,dtype,check_symm=True,check_herm=True,check_pcon=True)




