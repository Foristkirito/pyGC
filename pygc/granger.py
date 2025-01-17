########################################################################################
# Module with functions to compute GC
########################################################################################
import numpy as np

def granger_causality(S, H, Z):

	N = H.shape[2]

	Hxx = H[0,0,:]
	Hxy = H[0,1,:]
	Hyx = H[1,0,:]
	Hyy = H[1,1,:]

	Hxx_tilda = Hxx + (Z[0,1]/Z[0,0]) * Hxy
	Hyx_tilda = Hyx + (Z[0,1]/Z[0,0]) * Hxx
	Hyy_circf = Hyy + (Z[1,0]/Z[1,1]) * Hyx

	Syy = Hyy_circf*Z[1,1]*np.conj(Hyy_circf) + Hyx*(Z[0,0]-Z[1,0]*Z[1,0]/Z[1,1]) * np.conj(Hyx)
	Sxx = Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda) + Hxy*(Z[1,1]-Z[0,1]*Z[0,1]/Z[0,0]) * np.conj(Hxy)

	Ix2y = np.log( Syy/(Hyy_circf*Z[1,1]*np.conj(Hyy_circf)) )
	Iy2x = np.log( Sxx/(Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda)) )

	Ixy  = np.zeros(N)

	for i in range(N):
		Ixy[i]  = np.log( (Hxx_tilda[i]*Z[0,0]*np.conj(Hxx_tilda[i]))*(Hyy_circf[i]*Z[1,1]*np.conj(Hyy_circf[i])/np.linalg.det(S[:,:,i])) ).real
	
	return Ix2y.real, Iy2x.real, Ixy.real

def conditional_granger_causality(S, f, fs, Niterations=100, tol=1e-12, verbose=True):
	'''
		Computes the conditional Granger Causality
	'''

	from .non_parametric import wilson_factorization

	nvars = S.shape[0]

	_, _, Znew  = wilson_factorization(S, f, fs, Niterations, tol, verbose)

	LSIG        = np.log(np.diag(Znew))

	F           = np.zeros([nvars, nvars])

	for j in range(nvars):
		print('j = ' + str(j))

		# Reduced regression
		j0        = np.concatenate( (np.arange(0,j), np.arange(j+1, nvars)), 0) 

		S_aux     = np.delete(S, j, 0)
		S_aux     = np.delete(S_aux, j, 1)
		_, _, Zij = wilson_factorization(S_aux, f, fs, Niterations, tol, verbose)

		LSIGj     = np.log(np.diag(Zij))

		for ii in range(nvars-1):
			i = j0[ii]
			F[i,j] = LSIGj[ii] - LSIG[i]

	return F

def conditional_spec_granger_causality(S, f, fs, Niterations=100, tol=1e-12, verbose=True):
	'''
		Computes the conditional Granger Causality
	'''

	from .non_parametric import wilson_factorization

	nvars = S.shape[0]

	_, Hnew, Znew  = wilson_factorization(S, f, fs, Niterations, tol, verbose)

	SIG = np.diag(Znew)

	GC = np.zeros([nvars,nvars,len(f)])

	for j in range(nvars):
		print('j = ' + str(j))

		# Reduced regression
		j0        = np.concatenate( (np.arange(0,j), np.arange(j+1, nvars)), 0) 

		S_aux     = np.delete(S, j, 0)
		S_aux     = np.delete(S_aux, j, 1)
		_, Hij, Zij = wilson_factorization(S_aux, f, fs, Niterations, tol, verbose)

		SIGj = np.diag(Zij)


		G = np.zeros([nvars, nvars, len(f)]) * (1+1j)

		for i in range(len(f)):
			aux = np.insert(Hij[:,:,i], j, np.zeros(nvars-1), axis=1)
			aux = np.insert(aux, j, np.zeros(nvars), axis=0)
			G[:,:,i] = aux
		G[j,j,:] = 1
		
		Q = np.zeros([nvars, nvars, len(f)]) * (1+1j)

		for i in range(len(f)):
			Q[:,:,i] = np.matmul( np.linalg.inv(G[:,:,i]), Hnew[:,:,i] )

		for ii in range(nvars-1):
			i = j0[ii]
			div     = Q[i,i,:]*Znew[i,i]*np.conj(Q[i,i,:]).T
			GC[j,i] = np.log( SIGj[ii] / np.abs( div ) )

	return GC
