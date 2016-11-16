from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np

L=2
basis = spin_basis_1d(L,pblock=1)
P = basis.get_proj(dtype=np.complex128)



J = [[1.0,i,(i)%L] for i in range(L)]
h = [[1.0,i] for i in range(L)]
#print J

O1 = hamiltonian([["xz",J]],[],basis=basis,check_symm=False,check_herm=False)
#O1 = hamiltonian([["x",h]],[],basis=basis,check_symm=False)

z_list = [hamiltonian([["z",[[1.0,i]] ]],[],N=L) for i in range(L)]
x_list = [hamiltonian([["x",[[1.0,i]] ]],[],N=L) for i in range(L)]
y_list = [hamiltonian([["y",[[1.0,i]] ]],[],N=L) for i in range(L)]
xz_list = [hamiltonian([["xz",[[1.0,i,(i+1)%L]] ]],[],N=L) for i in range(L)]
x_red_list = [hamiltonian([["x",[[1.0,i]] ]],[],basis=basis,check_symm=False,check_herm=False) for i in range(L)]
y_red_list = [hamiltonian([["y",[[1.0,i]] ]],[],basis=basis,check_symm=False,check_herm=False) for i in range(L)]
z_red_list = [hamiltonian([["z",[[1.0,i]] ]],[],basis=basis,check_symm=False,check_herm=False) for i in range(L)]

print (z_list[0].project_to(P)).todense()
print 
print (z_list[0].project_to(P)*z_list[0].project_to(P)).todense()
exit()


xz_red_list = [hamiltonian([["xz",[[1.0,i,(i+1)%L]] ]],[],N=L,basis=basis,check_symm=False,check_herm=False) for i in range(L)]
sum_xz_red_list = hamiltonian([["xz",[[1.0,i,(i+1)%L] for i in range(L)] ]],[],N=L,basis=basis,check_symm=False,check_herm=False) 


print sum(xz_list).project_to(P).todense()
print
print sum_xz_red_list.todense()
print
print (x_red_list[0]*z_red_list[0] + x_red_list[1]*z_red_list[1]).todense()
print
exit()
O2 = hamiltonian([],[],basis=basis)

for x,z,y,y_red,xz in zip(x_list,z_list,y_list,y_red_list,xz_red_list):
	xx = x.project_to(P)
	zz = z.project_to(P)
	yy = y.project_to(P)
	xz = xz

	print xz.todense()
	print

#	print z.todense()
#	print zz.todense()
#	print
#	print x.todense()
#	print xx.todense()

#	print np.linalg.norm((x*z).todense() + 1j*y.todense())
#	print np.linalg.norm(y_red.todense() - yy.todense())
#	print np.linalg.norm((xx*zz).todense() + 1j*y_red.todense())

	O2 += xx*zz
#	O2 += xx


#print O1-O2

