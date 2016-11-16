import sys,os
qspin_path = os.path.join(os.getcwd(),"../qspin_dev/")
sys.path.insert(0,qspin_path)

from qspin.operators import hamiltonian, exp_op # Hamiltonians and operators
from qspin.basis import spin_basis_1d # Hilbert space spin basis
from qspin.tools.measurements import obs_vs_time, diag_ensemble # t_dep measurements
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
sp = hamiltonian([['+',[[1,0]]]],[],dtype=np.complex128,basis=basis,check_symm=False,check_herm=False)


print 'STARTS HERE'
print H.todense()
print 'weird bug in the line above: L=1, but somehow the J_nn has an effect: look at the diag elmnts'
print sp.todense()
print '-------'

a=1j
print H.project_to(exp_op(a*sp).get_mat().todense()).todense()
print H.project_to(exp_op(a*sp) ).todense()
print "in the following line, the exp_op optional arguments don't do anything: maybe an error should be thornw?"
print H.rotate_by(exp_op(a*sp).get_mat().todense(),generator=False,a=20000.0).todense()
print "when u flip the daggers in the sandwich function, use the commented lines below"
#print H.rotate_by(sp.H,generator=True,a=a.conjugate()).todense()
#print exp_op(sp.H,a=a.conjugate()).sandwich(H).todense()
print H.rotate_by(sp,generator=True,a=a).todense()
print exp_op(sp,a=a).sandwich(H).todense()







