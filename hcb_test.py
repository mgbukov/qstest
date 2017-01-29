import sys,os
qspin_path = os.path.join(os.getcwd(),"../hardcore_bosons/")
sys.path.insert(0,qspin_path)


from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, hcb_basis_1d, fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

##### define model parameters #####
L=10 # system size
J=1.0 # spin interaction
h=0.9045 # parallel field

# compute basis in the 0-total momentum and +1-parity sector
basis_spin=spin_basis_1d(L=L,a=1,Nup=L/2,kblock=0,pblock=1,zblock=1,pauli=False)
basis_hcb = hcb_basis_1d(L=L,a=1,Nup=L/2,kblock=0,pblock=1,cblock=1)
basis_fermion = fermion_basis_1d(L=L,a=1,Nup=L/2,kblock=0,pblock=1,cblock=1)


# define PBC site-coupling lists for operators
#z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
J_zz=[[h,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["+-",J_nn],["-+",J_nn],["zz",J_zz]]

H_spin=hamiltonian(static,[],basis=basis_spin)
H_hcb=hamiltonian(static,[],basis=basis_hcb,check_herm=False,check_symm=False,check_pcon=False)
H_fermion=hamiltonian(static,[],basis=basis_fermion,check_herm=False,check_symm=False,check_pcon=False)


E_spin=H_spin.eigvalsh()
E_hcb=H_hcb.eigvalsh()
E_fermion=H_fermion.eigvalsh()

print(H_spin.Ns)
print("mismatch energies are", max(abs(E_spin-E_hcb)) )
print(E_hcb)
print('-------')
print(E_fermion)



