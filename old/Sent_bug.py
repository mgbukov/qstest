from __future__ import print_function, division

import sys,os,argparse,time
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy, _ent_entropy

import numpy as np

v = np.arange(8).reshape((2,2,2))
print(v[0,0,0])
print(v[0,0,1])
print(v[0,1,0])
print(v[0,1,1])
print(v[1,0,0])
print(v[1,0,1])
print(v[1,1,0])
print(v[1,1,1])

print('-------')
v = np.arange(8).T.reshape((2,2,2)).T
print(v[0,0,0])
print(v[1,0,0])
print(v[0,1,0])
print(v[1,1,0])
print(v[0,0,1])
print(v[1,0,1])
print(v[0,1,1])
print(v[1,1,1])

print('-------')
v = np.arange(8).reshape((2,2,2),order='F')
print(v[0,0,0])
print(v[1,0,0])
print(v[0,1,0])
print(v[1,1,0])
print(v[0,0,1])
print(v[1,0,1])
print(v[0,1,1])
print(v[1,1,1])

#exit()


L=3
basis2=spin_basis_1d(L)

vec=0.5*np.array([0.0,np.sqrt(2),np.sqrt(2),0.0,0.0,0.0,.0,0.])

sitesL=[0]
ent3=_ent_entropy(vec,basis2,chain_subsys=sitesL)
#ent3=basis2.ent_entropy(vec,sub_sys_A=sitesL)
print(ent3["Sent"])

sitesL=[1,2]
#ent1=ent_entropy(vec,basis2,chain_subsys=sitesL)
ent2=_ent_entropy(vec,basis2,chain_subsys=sitesL)
ent3=basis2.ent_entropy(vec,sub_sys_A=sitesL)
print(ent2["Sent"],ent3["Sent_A"])

###

vec=np.vstack((vec,vec)).T

sitesL=[0]
ent3=_ent_entropy({'V_states':vec},basis2,chain_subsys=sitesL)
#ent3=basis2.ent_entropy(vec,sub_sys_A=sitesL)
print(ent3["Sent"])

sitesL=[0,1]
#ent1=ent_entropy(vec,basis2,chain_subsys=sitesL)
ent2=_ent_entropy({'V_states':vec},basis2,chain_subsys=sitesL)
#ent3=basis2.ent_entropy(vec,sub_sys_A=sitesL)
print(ent2["Sent"])