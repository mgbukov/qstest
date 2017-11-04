import sys,os
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time, diag_ensemble # t_dep measurements
import numpy as np # generic math functions
#
##### define model parameters #####
L=1 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
#
##### set up alternating Hamiltonians #####
# compute basis 
basis=spin_basis_1d(L=L)
# define PBC site-coupling lists for operators
x_field_pos=[[+g,i]	for i in range(L)]
z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
# compute Hamiltonians
H=hamiltonian(static,[],dtype=np.float64,basis=basis,check_symm=False,check_herm=False)

# define initial \sigma^z_{j=0} and \sigma^z_{j=L/2} at two sites i=0 and j=L/2
z_i = hamiltonian([['z',[[1,0]]]],[],dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
z_i.as_dense_format() # cast in dense format since matrix-exp will result in a dense matrix

z_j = hamiltonian([['z',[[1.0,L/2]]]],[],dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
x_j = hamiltonian([['x',[[1.0,L/2]]]],[],dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
y_j = hamiltonian([['y',[[1.0,L/2]]]],[],basis=basis,check_symm=False,check_herm=False)

print y_j.todense()
exit()

# rotate z_i by H
start,stop,num = 0.0,10.0,11
tvec = np.linspace(start,stop,num)
# this creates a generator object which can be looped opver to obtain the sandwich at each time
z_i_time =  z_i.rotate_by(H,generator=True,a=+1j,start=start,stop=stop,num=num,iterate=True)

for op in z_i_time:
	#print np.around( op.todense() , 3)	
	c_z_j = np.trace((op*z_j).todense())
	c_x_j = np.trace((op*x_j).todense())
	c_id_j = np.trace((op*id_j).todense())
	print c_id_j, c_z_j, c_x_j,op.tocsr().nnz




