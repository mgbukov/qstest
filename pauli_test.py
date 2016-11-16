import sys,os
qspin_path = os.path.join(os.getcwd(),"../qspin_dev/")
sys.path.insert(0,qspin_path)

from qspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from qspin.operators import hamiltonian # Hamiltonian and observables
from qspin.tools.measurements import ent_entropy
import numpy as np


L = 2
J=0
hz=0.5

basis = spin_basis_1d(L=L,kblock=0,pblock=1,pauli=True)
			
zz_int =[[J,i,(i+1)%L] for i in range(L)]
x_field=[[-1.0,i] for i in range(L)]
z_field=[[-hz,i] for i in range(L)]

static = [["zz",zz_int],["z",z_field]]
dynamic = [["x",x_field,lambda t: 1,[]]]

kwargs = {'dtype':np.float64,'basis':basis,'check_symm':False,'check_herm':False}
H = hamiltonian(static,dynamic,**kwargs)

print H.todense()