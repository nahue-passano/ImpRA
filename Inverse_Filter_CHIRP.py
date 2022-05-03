# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:04:07 2022

@author: Cori
"""

import soundfile as sf
import numpy as np
from scipy.signal import chirp
import array

fs = 44100

def Sine_Sweep_Generator(f_min, f_max, T, fs):
    """

    Parameters
    ----------
    f_min : Frecuencia inicial del sine-sweep [Hz].
    f_max : Frecuencia final de sine-sweep [Hz].
    T : Duración del sine-sweep [s].
    fs : Frecuencia de muestreo [Hz].

    Returns
    -------
    None.

    """
    t = np.arange(0, T, 1/fs)
    ss = chirp(t, f0=f_min, f1=f_max, t1=T, method='logarithmic')
    sf.write('sine_sweep.wav', ss, fs)
    
    ss_inv = ss[::-1]
    m = 1/(2*np.pi*np.exp(t*np.log(f_max/f_min)/T))
    k = m*ss_inv
    
    sf.write('inverse_filter.wav', k, fs)
    
    import matplotlib.pyplot as plt
    ss_fft = np.fft.rfft(ss)
    freq_ss = np.fft.rfftfreq(ss.size, 1/fs)
    plt.xlim([0, 1000])
    plt.plot(freq_ss, 20*np.log10(np.abs(ss_fft)))
    plt.show()
    
    k_fft = np.fft.rfft(k)
    freq_ss_inv = np.fft.rfftfreq(ss_inv.size, 1/fs)
    plt.xlim([0, 1000])
    plt.plot(freq_ss_inv, 20*np.log10(np.abs(k_fft)))
    plt.show()
    
    return k

filtro_inverso = Sine_Sweep_Generator(100, 1000, 10, 44100)