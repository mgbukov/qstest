from __future__ import print_function, division

import sys,os
#qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
qspin_path = "/Users/mbukov/ED_python/QuSpin_dev/"
sys.path.insert(0,qspin_path)

import quspin
print('Imported example from:', quspin.__file__)

exit()

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import evolve
from quspin.tools.Floquet import Floquet_t_vec
import numpy as np # generic math functions
import scipy.sparse as sp
from scipy.special import jv

import matplotlib.pyplot as plt

import sys
import argparse

seed=19
np.random.seed(seed)


##### define model parameters #####
L=int(sys.argv[1]) # system size
if L%2==0:
	i_CM = L//2-0.5 # centre of chain
else:
	i_CM = L//2



J=1.0
mu=0.00 #0.002

g = 0.99
rho = 96*(425E-3)**2 # 27*(425E-3)^2
U = 1.0 #g/rho # Bose-Hubbard interaction strength


A=1.0
Omega=3.0 #3.13 # 
Jeff = jv(0,A/Omega)

q_vec=2*np.pi*np.fft.fftfreq(L)

hopping=[[-J,i,(i+1)%L] for i in range(L)]
trap=[[mu*(i-i_CM)**2,i] for i in range(L)]
shaking=[[A*Omega*(i-i_CM),i] for i in range(L)]

# define basis
basis = boson_basis_1d(L,Nb=1,sps=2)

### lab-frame Hamiltonian
def drive(t,Omega):
	return np.cos(Omega*t)

drive_args=[Omega]

static=[["+-",hopping],["-+",hopping],['n',trap]]
dynamic=[["n",shaking,drive,drive_args]]

### rot-frame Hamiltonian
def drive_rot(t,Omega):
	return np.exp(-1j*A*np.sin(Omega*t) )

def drive_rot_cc(t,Omega):
	return np.exp(+1j*A*np.sin(Omega*t) )

drive_args=[Omega]

static_rot=[['n',trap]]
dynamic_rot=[["+-",hopping,drive_rot,drive_args],["-+",hopping,drive_rot_cc,drive_args]]


#### calculate Hamiltonian
H_static=hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False,check_pcon=False)
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_herm=False,check_symm=False,check_pcon=False)
H_rot=hamiltonian(static_rot,dynamic_rot,basis=basis,dtype=np.complex128,check_herm=False,check_symm=False,check_pcon=False)

E,V=H_static.eigh()
E_rot,V_rot=H_rot.eigh()


#y0=V[:,0]#*np.sqrt(L)
#print(H_static.matrix_ele(y0,y0), 0.5*U*np.sum(np.abs(y0)**4)  )
#exit()


'''
def GPE(time,V,H,U):

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[Ns:] =  H.static.dot(V[:Ns])
	V_dot[:Ns] = -H.static.dot(V[Ns:])


	# static GPE interaction
	V_dot_2 = (np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2)
	V_dot[Ns:] += U*V_dot_2*V[:Ns]
	V_dot[:Ns] -= U*V_dot_2*V[Ns:]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[Ns:] += f(time,*f_args)*Hd.dot(V[:Ns])
		V_dot[:Ns] -= f(time,*f_args)*Hd.dot(V[Ns:])

	return V_dot

def GPE_imag_time(time,V,H,U):

	"""
	\dot y = - ([f+ih]Hy + g abs(y)^2 y)

	y = u + iv

	\dot u + i\dot v = - { [f+ih]H(u + iv) + U( abs(u)^2 + abs(v)^2 )(u + iv) }
					 = - { fHu - hHv + i(fHv + hHu) + U( abs(u)^2 + abs(v)^2 )(u + iv)  }

	\dot u = - { fHu - hHv + U( abs(u)^2 + abs(v)^2 )u }
	\dot v = - { fHv + hHu + U( abs(u)^2 + abs(v)^2 )v }
	"""

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[:Ns] = H.static.dot(V[:Ns]).real
	V_dot[Ns:] = H.static.dot(V[Ns:]).real
	
	# static GPE interaction 
	V_dot_2 = (np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2)
	V_dot[:Ns] += U*V_dot_2*V[:Ns]
	V_dot[Ns:] += U*V_dot_2*V[Ns:]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[:Ns] += ( (f(time,*f_args).real)*Hd.dot(V[:Ns]) - (f(time,*f_args).imag)*Hd.dot(V[Ns:]) ).real
		V_dot[Ns:] += ( (f(time,*f_args).real)*Hd.dot(V[Ns:]) + (f(time,*f_args).imag)*Hd.dot(V[:Ns]) ).real

	return -V_dot
'''
def GPE_cpx(time,V,H,U):

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[:Ns] =  H.static.dot(V[Ns:]).real
	V_dot[Ns:] = -H.static.dot(V[:Ns]).real


	# static GPE interaction
	V_dot_2 = np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2
	V_dot[:Ns] += U*V_dot_2*V[Ns:]
	V_dot[Ns:] -= U*V_dot_2*V[:Ns]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[:Ns] +=  ( +(f(time,*f_args).real)*Hd.dot(V[Ns:]) + (f(time,*f_args).imag)*Hd.dot(V[:Ns]) ).real
		V_dot[Ns:] +=  ( -(f(time,*f_args).real)*Hd.dot(V[:Ns]) + (f(time,*f_args).imag)*Hd.dot(V[Ns:]) ).real

	return V_dot


def GPE_imag_time2(time,V,H,U):
	return -( H.static.dot(V) + U*np.abs(V)**2*V )






##### imaginary-time evolution
y00=V[:,0]*np.sqrt(L)
t_imag=Floquet_t_vec(Omega,20,len_T=1)

GPE_params = (H_static,U) #
y0_t = evolve(y00,t_imag.i,t_imag.vals,GPE_imag_time2,real=True,iterate=True,imag_time=True,f_params=GPE_params)

E_old=0.
for y0 in y0_t:
	E_new=(H_static.matrix_ele(y0,y0) + 0.5*U*np.sum(np.abs(y0)**4) ).real
	#print("energy:", E_new )
	
	E_old=E_new
print('finished calculating GS w/ conv error', E_old-E_new)


print("GS energy:",     (H_static.matrix_ele(y00,y00) + np.sum(0.5*U*np.abs(y00)**4) ) )
#print("GS kin energy:", 1.0/L*np.sum( H_static.matrix_ele(V[:,0],V[:,0]) )    )
#print("GS int energy:", 1.0/L*np.sum( 0.5*g*np.abs(V[:,0])**4)     )


print("GPE energy:",    (H_static.matrix_ele(y0,y0) + np.sum(0.5*U*np.abs(y0)**4) ) )
#print("GPE kin energy:", 1.0/L*np.sum( H_static.matrix_ele(y,y) ) )
#print("GPE int energy:", 1.0/L*np.sum( 0.5*g*np.abs(y)**4) )


#exit()

"""
plt.scatter(np.arange(L)-i_CM, abs(y00)**2, color='green' )
plt.scatter(np.arange(L)-i_CM, abs(y0)**2, color='red' )
plt.show()

plt.scatter(q_vec, abs(np.fft.fft(y00)/L)**2, color='green' )
plt.scatter(q_vec, abs(np.fft.fft(y0)/L)**2, color='red' )
plt.show()
exit()
#"""


#"""
##### real-time evolution

N=35
t=Floquet_t_vec(Omega,N,len_T=1)

#plt.scatter(np.arange(L)-i_CM, abs(y0)**2,color='b')

noise=1.0/L*np.random.uniform(size=L)
norm=np.linalg.norm(y0)

y0+= noise
y0=y0/np.linalg.norm(y0)*norm

#plt.scatter(np.arange(L)-i_CM, abs(y0)**2,color='r')
#plt.show()

print(abs(np.fft.fft(y0)/L)**2)
plt.scatter(q_vec, abs(np.fft.fft(y0)/L)**2, color='red' )
plt.show()





GPE_params = (H_rot,U) #
y_t = evolve(y0,t.i,t.f,GPE_cpx,stack_state=True,f_params=GPE_params)

#"""
y = y_t#[:,-1]
plt.scatter(q_vec, abs(np.fft.fft(y)/L)**2, color='blue',marker='o' )
#plt.ylim([0.0,0.1])
plt.show()
#"""
exit()


'''
y_t = evolve(y0,t.i,t.vals,GPE_cpx,stack_state=True,iterate=True,f_params=GPE_params)

print('starting real-time evolution...')
E=[]
for i,y in enumerate(y_t):
	E.append( (H_static.matrix_ele(y,y) + 0.5*U*np.sum(np.abs(y)**4) ).real )
	print("(N_T,E)=:", (t.vals[i]/t.T,E[-1]-E[0]) )

	#plt.scatter(np.arange(L)-i_CM, abs(y)**2, color='blue' )
	#plt.show()
	"""
	plt.scatter(q_vec, abs(np.fft.fft(y))**2/L**2, color='blue',marker='o' )
	plt.ylim([0.0,0.1])
	plt.title('$\\mathrm{period}\\ l=%i$'%(i))
	
	plt.draw()
	plt.pause(0.005)
	plt.clf()
	"""
#plt.close()

plt.plot(t.vals/t.T,E-E[0])
plt.show()
'''



##### ToF
x_ToF=np.arange(-1.0*L//2,1.0*L//2,1)
def ToF(L,d,alpha,beta,corr):
	"""
	calculates density after time of flight. The parameters are
	L: number of lattice sites
	d: lattice spacing
	x = r/d: dim'less position
	alpha = 1.0/( (sigma/d)**2 + (hbar*t/(m*sigma*d))**2 )
	beta = alpha*hbar*t/(m*sigma**2)
	corr = \langle a^\dagger_j a_l \rangle: two-point function
	"""
	from numpy import exp

	xx=np.arange(-L//2,L//2,1)
	
	prefactor=1.0/d*np.sqrt(alpha/np.pi)

	n_ToF= np.zeros((len(x_ToF),))
	for i,x in enumerate(x_ToF):
		"""
		S=0.0j
		for j,xj in enumerate(np.arange(-L//2,L//2,1)):
			for l,xl in enumerate(np.arange(-L//2,L//2,1)):

				S+=corr[j,l]*exp(-alpha*x**2 + alpha*(xl+xj)*x - 0.5*alpha*(xl**2 + xj**2) - 1j*beta*x*(xl - xj) + 0.5j*beta*x*(xl**2 - xj**2) )
		S=S.real
		"""
		S = np.sum( corr*exp(- alpha*x**2 								\
						  	 + alpha*x*(xx[:,np.newaxis] + xx) 			\
							 - 0.5*alpha*(xx[:,np.newaxis]**2 + xx**2)  \
							 - 1.0j*beta*x*(xx[:,np.newaxis] - xx) 		\
						   	 + 0.5j*beta*x*(xx[:,np.newaxis]**2 - xx**2)   )).real


		n_ToF[i]=prefactor*S

	return n_ToF/np.sum(np.abs(n_ToF))


"""
# ToF for a Thpmas-Fermi profile

N_TF=27
c_mu = np.zeros((L,))
for i,mu in enumerate(range(L)):
	mu-=i_CM
	if abs(mu)<N_TF:
		c_mu[i]=np.sqrt(1.0 - (mu/N_TF)**2 )
norm=np.linalg.norm(c_mu)
c_mu /=norm

corr_TF=np.outer(c_mu,c_mu)/L**2

#plt.scatter(np.arange(L)-i_CM, abs(c_mu)**2 )
#plt.show()
"""


for t_ToF in [1e-4,0.001,0.005,0.01,0.05,0.1,1.0,10.0]:

	# in SI units
	sigma=50e-9
	d=425e-9
	hbar=1.0545718e-34
	m=6.476106e-26


	alpha=1.0/( (sigma/d)**2 + (hbar*t_ToF/(m*sigma*d))**2 )
	beta = alpha*hbar*t_ToF/(m*sigma**2)
	
	corr_0=np.outer(y0.conj(),y0)
	corr=np.outer(y.conj(),y)

	#print('ToF alpha, beta', alpha, beta)

	n_ToF_0=ToF(L,d,alpha,beta,corr_0)
	n_ToF=ToF(L,d,alpha,beta,corr)
	

	plt.scatter(x_ToF, n_ToF_0, color='red' )
	plt.scatter(x_ToF, n_ToF, color='blue' )
	#plt.plot(q_vec, abs(np.fft.fft(y))**2/L, color='red',marker='o' )
	#plt.ylim([0.0,0.1])
	plt.show()

