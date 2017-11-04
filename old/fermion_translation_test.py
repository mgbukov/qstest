from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
try: # import python 3 zip function in python 2 and pass if already using python 3
    import itertools.izip as zip
except ImportError:
    pass 

np.set_printoptions(precision=4,suppress=True)

##### define model parameters #####
L=4 # system size
J=1.0 # uniform hopping contribution
##### construct single-particle Hamiltonian #####
# define site-coupling lists
hop_p=[[-J,i,(i+1)%L] for i in range(L)] # PBC
hop_m=[[+J,i,(i+1)%L] for i in range(L)] # PBC
# define static and dynamic lists
static=[["+-",hop_p],["-+",hop_m]]
dynamic=[]
# define basis
basis=fermion_basis_1d(L,Nf=2)
# build real-space Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
# diagonalise real-space Hamiltonian
E,V=H.eigh()
##### compute Fourier transform and momentum-space Hamiltonian #####
# define basis blocks and arguments
blocks=[dict(Nf=2,kblock=i) for i in range(L)] # only L//2 distinct momenta
basis_args = (L,)
# construct block-diagonal Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,fermion_basis_1d,
						basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
# diagonalise momentum-space Hamiltonian
Eblock,Vblock=Hblock.eigh()

print(Hblock.toarray())
exit()

print( np.linalg.norm( (Hblock.H - Hblock).toarray() ) )
print( np.linalg.norm( Hblock.H.toarray() - Hblock.toarray() ) )
exit()