import sys,os
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
sys.path.insert(0,qspin_path)

#chain_subsys==list(range(len(chain_subsys))):

from quspin.operators import hamiltonian, exp_op # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy # entropies
from numpy.random import ranf,seed # pseudo random numbers
from joblib import delayed,Parallel # parallelisation
import numpy as np # generic math functions
from time import time # timing package
#
##### define simulation parameters #####
n_real=1 # number of disorder realisations
n_jobs=1 # number of spawned processes used for parallelisation
#
##### define model parameters #####
L=10 # system size
Jcoup=1.0 # interaction
h_MBL=10 # MBL disorder strength
#
##### times #####
texpmin=-2
texpmax=+2
steps=10
time_steps=np.array([10**n for n in np.linspace(texpmin,texpmax,steps,endpoint=False)])
#
##### set up Heisenberg Hamiltonian  #####
# compute basis in the 0-total magnetisation sector (requires L even)
basis = spin_basis_1d(L,Nup=L//2,pauli=True)
# define operators with OBC using site-coupling lists
J_Heis = [[Jcoup,i,i+1] for i in range(L-1)] # OBC
# static and dynamic lists
static = [["xx",J_Heis],["yy",J_Heis],["zz",J_Heis]] #
#dynamic =[["zz",J_zz,ramp,ramp_args]]
dynamic = []
# compute the time-dependent Heisenberg Hamiltonian
H_Heis = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
# compute observable
ImbDiag = hamiltonian([["z",[[(-1)**i,i] for i in range(L)]]],[],basis=basis,dtype=np.float64).tocsr().diagonal()
# Initial state is neal state
psi_0=np.zeros(len(ImbDiag))
psi_0[np.argmax(ImbDiag)]=1


##### calculate diagonal and entanglement entropies #####
#@profile
def realization(H_Heis,psi_0,basis,diag_op,times,real):
        """
        This function computes the entropies for a single disorder realisation.
        --- arguments ---
        vs: vector of ramp speeds
        H_Heis: static Heisenberg Hamiltonian
        basis: spin_basis_1d object containing the spin basis
        n_real: number of disorder realisations; used only for timing
        """
        ti = time() # get start time
        #
        seed() # the random number needs to be seeded for each parallel process
        #
        # draw random field uniformly from [-1.0,1.0] for each lattice site
        unscaled_fields=-1+2*ranf((basis.L,))
        # define z-field operator site-coupling list
        h_z=[[unscaled_fields[i],i] for i in range(basis.L)]
        # static list
        disorder_field = [["z",h_z]]
        # compute disordered z-field Hamiltonian
        no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
        Hz=hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,**no_checks)
        # compute the MBL and ETH Hamiltonians for the same disorder realisation
        H_MBL=H_Heis+h_MBL*Hz
        expO = exp_op(H_MBL)
        psi_t = psi_0
        time_old=0
        imb=[]
        Sent=[]
        sub_sizes=[1]
        for a_time in times:
                expO.set_a(-1j*(a_time-time_old))
                time_old=a_time
                psi_t=expO.dot(psi_t)
                imb.append(np.einsum('i,i,i',np.conj(psi_t),diag_op,psi_t))
                Sent.append([[ent_entropy(psi_t,basis,chain_subsys=range(x,x+size))["Sent"] for x in range(L-size+1)] for size in sub_sizes])
                
                print(np.around(Sent,3))
                
#       imb_at_t = lambda time: diag_op_exp(expO,psi_0,diag_op,time)
#       imb = list(map(imb_at_t,times))
#       imb = list(map(imb_at_t,times))
        # show time taken
        print("realization {0}/{1} took {2:.2f} sec".format(real+1,n_real,time()-ti))
        #
        return (np.real(imb),Sent)

mean_Imb , mean_ents = realization(H_Heis,psi_0,basis,ImbDiag,time_steps,0)
