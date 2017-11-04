from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, fermion_basis_1d
import numpy as np

L=6

basis=fermion_basis_1d(L,Nf=L//2)
basis_symm_p=fermion_basis_1d(L,Nf=L//2,cblock=1)
basis_symm_m=fermion_basis_1d(L,Nf=L//2,cblock=-1)

j=2

mu=[[0.0,i] for i in range(L)]

J_l=[[+1.0,i,(i+j)%L] for i in range(L)]
J_r=[[-1.0,i,(i+j)%L] for i in range(L)]

static=[['+-',J_l],['-+',J_r],['n',mu]]

H=hamiltonian(static,[],basis=basis,check_symm=False)
H_sym_p=hamiltonian(static,[],basis=basis_symm_p,check_symm=False)
H_sym_m=hamiltonian(static,[],basis=basis_symm_m,check_symm=False)

E=H.eigvalsh()
E_sym_p=H_sym_p.eigvalsh()
E_sym_m=H_sym_m.eigvalsh()

print(E[0],E_sym_p[0],E_sym_m[0])



E_sym=np.sort( np.concatenate((E_sym_m,E_sym_p)) )

print(E)
print(E_sym)

exit()

print( np.linalg.norm(E_nnn-E_nnn_sym) )

