from quspin.operators import hamiltonian
import numpy as np # generic math functions
from scipy.sparse.linalg import expm_multiply
from multiprocessing import Process,Queue,Event
from joblib import Parallel,delayed
import scipy.sparse as sp
import sys


from memory_profiler import profile




def worker(gen_func,args_list,q,e):
	from itertools import izip
	gens = []
	for arg in args_list:
		gens.append(gen_func(*arg))

	generator = izip(*gens)
	for s in generator:
		e.clear()
		q.put(s)
		e.wait()

	q.close()




def generate_parallel(n_process,n_iter,gen_func,args_list):
	n_items = len(args_list)

	if n_items == n_process:
		n_pp = 1
		n_left = 1
	elif n_items < n_process and n_process > 0:
		n_process = n_items
		n_pp = 1
		n_left = 1
	elif n_items > n_process and n_process > 0:
		n_pp = n_items//n_process
		n_left = n_pp + n_items%n_process		


	if n_process <= 1:
		from itertools import izip
		gens = []
		for arg in args_list:
			gens.append(gen_func(*arg))

		generator = izip(*gens)

		for s in generator:
			yield s

		return 

	sub_lists = [args_list[0:n_left]]
	sub_lists.extend([ args_list[n_left + i*n_pp:n_left + (i+1)*n_pp] for i in range(n_process-1)])

	es = []
	qs = []
	ps = []
	for i in range(n_process):
		e = Event()
		q = Queue(1)
		p = Process(target=worker, args=(gen_func,sub_lists[i],q,e))
		es.append(e)
		qs.append(q)
		ps.append(p)


	for p in ps:
		p.start()


	for i in range(n_iter):
		s = []

		for q,e in zip(qs,es):
			s.extend(q.get())
			e.set()

		yield tuple(s)

	for p in ps:
		p.join()



def block_hamiltonian(blocks,static,dynamic,basis_con,basis_args,dtype):
	H_list = []
	P_list = []

	dynamic_list = [([],f,f_args) for _,_,f,f_args in dynamic]
	static_mats = []

	for block in blocks:
		b = basis_con(*basis_args,**block)
		P = b.get_proj(dtype)
		if b.Ns > 0:
			P_list.append(P)
		H = hamiltonian(static,dynamic,basis=b,dtype=dtype)

		static_mats.append(H.static.tocoo())
		for i,(Hd,_,_) in enumerate(H.dynamic):
			dynamic_list[i][0].append(Hd.tocoo())


	static = [sp.block_diag(static_mats,format="csr")]
	dynamic = []
	for mats,f,f_args in dynamic_list:
		mats = sp.block_diag(mats,format="csr")
		dynamic.append([mats,f,f_args])

	P = sp.hstack(P_list,format="csr")
	return P,hamiltonian(static,dynamic,copy=False)








def _evolve_gen(psi,H,t0,times,solver_name,solver_args):
	for psi in H.evolve(psi,t0,times,solver_name=solver_name,iterate=True,**solver_args):
		yield psi



def _block_evolve_iter(psi_blocks,H_list,P,t0,times,solver_name,solver_args,n_jobs):
	args_list = [(psi_blocks[i],H_list[i],t0,times,solver_name,solver_args) for i in range(len(H_list))]

	for psi_blocks in generate_parallel(n_jobs-1,len(times),_evolve_gen,args_list):
		psi_t = np.hstack(psi_blocks)
		yield P.dot(psi_t)		




def block_evolve(blocks,static,dynamic,dtype,basis_con,basis_args,psi_0,t0,times,iterate=False,n_jobs=1,solver_name="dop853",**solver_args):
	if np.linalg.norm(psi_0) == 0:
		raise ValueError("Expecting non-zero array for psi_0.")
	list_all = []
	blocks = iter(blocks)
	checks_off = {"check_symm":False,"check_herm":False,"check_pcon":False}
	for block in blocks:
		b = basis_con(*basis_args,**block)
		if b.Ns > 0:
			p = b.get_proj(dtype)
			psi = p.T.conj().dot(psi_0)
			if np.linalg.norm(psi) > 1000*np.finfo(dtype).eps:	
				H = hamiltonian(static,dynamic,basis=b,dtype=dtype,**checks_off)
				list_all.append((H.Ns,H,psi,p))

#	list_all.sort(key=lambda x:x[0])
#	list_all.reverse()
	size,H_list,psi_blocks,P = zip(*list_all)
	H_list = list(H_list)
	psi_blocks = list(psi_blocks)
	P = sp.hstack(P,format="csr")
	if iterate:
		return _block_evolve_iter(psi_blocks,H_list,P,t0,times,solver_name,solver_args,n_jobs)
	else:
		psi_t = Parallel(n_jobs = n_jobs)(delayed(_block_evolve_helper)(H,psi,t0,times,solver_name,solver_args) for psi,H in zip(psi_blocks,H_list))
		psi_t = np.hstack(psi_t).T
		psi_t = P.dot(psi_t).T
		return psi_t


def _expm_gen(psi,H,times,dt):
	if times[0] != 0:
		H *= times[0]
		psi = expm_multiply(H,psi)
		H /= times[0]

	yield psi

	H *= dt
	for t in times[1:]:
		psi = expm_multiply(H,psi)
		yield psi
	H /= dt


def _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs):
	times,dt = np.linspace(start,stop,num=num,endpoint=endpoint,retstep=True)
	args_list = [(psi_blocks[i],H_list[i],times,dt) for i in range(len(H_list))]

	for psi_blocks in generate_parallel(n_jobs-1,len(times),_expm_gen,args_list):
		psi_t = np.hstack(psi_blocks)
		yield P.dot(psi_t)	


#@profile
def block_expm(blocks,static,dtype,basis_con,basis_args,psi_0,start,stop,iterate=False,n_jobs=1,endpoint=True,num=50,a=-1j):
	if np.linalg.norm(psi_0) == 0:
		raise ValueError("Expecting non-zero array for psi_0.")
	list_all = []
	blocks = iter(blocks)
	checks_off = {"check_symm":False,"check_herm":False,"check_pcon":False}
	for block in blocks:
		b = basis_con(*basis_args,**block)
		if b.Ns > 0:
			p = b.get_proj(dtype)
			psi = p.T.conj().dot(psi_0)
			if np.linalg.norm(psi) > 1000*np.finfo(dtype).eps:
				print b.Ns,block
				H = a*hamiltonian(static,[],basis=b,dtype=dtype,**checks_off).tocsr()
				list_all.append((H.shape[0],H,psi,p))


	size,H_list,psi_blocks,P = zip(*list_all)
	H_list = list(H_list)
	psi_blocks = list(psi_blocks)
	P = sp.hstack(P,format="csr")
	if iterate:
		return _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs)
	else:
		raise NotImplementedError("until Scipy v0.19.0 is released this function will not work property for real time.")
		psi_t = Parallel(n_jobs = n_jobs)(delayed(expm_multiply)(H,psi,start=start,stop=stop,num=num,endpoint=endpoint) for psi,H in zip(psi_blocks,H_list))
		psi_t = np.hstack(psi_t).T
		psi_t = P.dot(psi_t).T
		return psi_t






