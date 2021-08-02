import numpy as np 
import neo
import elephant
from quantities import s, Hz
from mne.time_frequency import tfr_array_morlet


def compute_freq(N, Fs):
	# Simulated time
	# 5000, 200
	# 25
	T = N / Fs
	print(f"N {N}, Fs {Fs}")
	# Frequency array
	f = np.linspace(1/T,Fs/2-1/T,int(N/2+1))

	return f

def cxy(X, Y=[], f=None, Fs=1):
	# Number of data points
	N = X.shape[0]

	if len(Y) > 0:
		Xfft = np.fft.fft(X)[1:len(f)+1]
		Yfft = np.fft.fft(Y)[1:len(f)+1]
		Pxy  = Xfft*np.conj(Yfft) / N
		return Pxy
	else:
		Xfft = np.fft.fft(X)[1:len(f)+1]
		Pxx  = Xfft*np.conj(Xfft) / N
		return Pxx


def morlet(X, f, Fs=1):
	N = X.shape[0]
	
	# X = neo.AnalogSignal(X.T, t_start=0*s, nco = 3.0, sampling_rate=Fs*Hz, units='dimensionless')
	# return elephant.signal_processing.wavelet_transform(X,f,fs=Fs).reshape((N,len(f)))
	X = np.expand_dims(X, axis=0)
	X = np.expand_dims(X, axis=0)
	tmp = tfr_array_morlet(X, Fs, f, n_cycles=f/2, output="complex")[0, 0, :, :]
	return tmp.T


def morlet_power(X, Y=[], f=None, Fs=1):
	N = X.shape[0]

	if len(Y) > 0:
		Wx = morlet(X=X, f=f, Fs=Fs)
		Wy = morlet(Y=Y, f=f, Fs=Fs)
		return Wx*np.conj(Wy) / N
	else:
		Wx = morlet(X=X, f=f, Fs=Fs)
		return Wx*np.conj(Wx) / N
