import sys,os
qspin_path = os.path.join(os.getcwd(),"../hcb/")
sys.path.insert(0,qspin_path)


from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, hcb_basis_1d, fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

##### define model parameters #####
L=3 # system size
J=1.0 # spin interaction
h=0.0 #0.9045 # parallel field

# compute basis in the 0-total momentum and +1-parity sector
basis_spin=spin_basis_1d(L=L,pauli=False)#,a=1,kblock=0,pblock=1)
basis_hcb = hcb_basis_1d(L=L)#,a=1,kblock=0,pblock=1)
basis_fermion = fermion_basis_1d(L=L,Nup=1)#,a=1,kblock=0,pblock=1)


# define PBC site-coupling lists for operators
x_field=[[-h,i]		for i in range(L)]
n_field=[[h,i]		for i in range(L)]
p_field=[[-0.5*h,i]	for i in range(L)]

J_pm=[[-1.00*J,i,(i+1)%L] for i in range(L)] # PBC
J_pp=[[+0.00*J,i,(i+1)%L] for i in range(L)] # PBC
J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC

# static and dynamic lists
static_spin=[["zz",J_zz],["x",x_field]]
static_hcb=[["zz",J_zz],["+",p_field],["-",p_field]]
static_fermion=[["+-",J_pm],["-+",J_pm]] #,["++",J_pp],["--",J_pp],["z",n_field]]

H_spin=hamiltonian(static_spin,[],basis=basis_spin,dtype=np.float32)
H_hcb=hamiltonian(static_hcb,[],basis=basis_hcb,dtype=np.float32,check_herm=False,check_symm=False,check_pcon=False)
H_fermion=hamiltonian(static_fermion,[],basis=basis_fermion,dtype=np.float32,check_herm=False,check_symm=False,check_pcon=False)

"""
I=0
length=1
out=0
for i in range(length):
	out += ((I >> i) & 1)

print(out)
exit()
"""
print(basis_fermion)
print(H_fermion.todense() )
#print(H_fermion.todense() - H_fermion.todense().T )
exit()

E_spin=H_spin.eigvalsh()
E_hcb=H_hcb.eigvalsh()
E_fermion=H_fermion.eigvalsh()

print(H_spin.Ns,H_hcb.Ns,H_fermion.Ns)
#print("mismatch energies are", max(abs(E_spin-E_hcb)) )
#print(E_hcb)
print(E_spin)
print('-------')
print(E_fermion)



