# -*- coding: utf-8 -*-
"""
Created on Fri May  6 00:04:40 2022

@author: Cori
"""
import soundfile as sf
import numpy as np
from scipy.signal import chirp
# import matplotlib.pyplot as plt

def RIR_Generator(f_min, f_max, T, path):
    """
    Parameters
    ----------
    f_min : Frecuencia inicial del sine-sweep [Hz].
    f_max : Frecuencia final de sine-sweep [Hz].
    T : Duración del sine-sweep [s].

    Returns
    -------
    RIR : ndarray
        Respuesta al impulso

    """
    sine_sweep, fs = sf.read(path)                                          # Importa sine-sweep grabado
    d = len(sine_sweep)/fs                                                  # Duración de la grabación
    
    t = np.arange(0, T, 1/fs)                                               # Vector de tiempos
    ss = chirp(t, f0=f_min, f1=f_max, t1=T, method='logarithmic')           # Genera sine-sweep para obtener filtro inverso (normalizado)
    inv_ss = ss[::-1]   
    m = 1/(2*np.pi*np.exp(t*np.log(f_max/f_min)/T))                         # Factor de modulación
    inv_filt = m * inv_ss                                                   # Filtro inverso
    inv_filt = inv_filt/np.amax(abs(inv_filt))                              # Normalizado
    inv_filt = np.pad(inv_filt, (0, int((d-T)*fs)), constant_values=(0, 0)) # Agregando zero padding
    
    sine_sweep_fft = np.fft.rfft(sine_sweep)
    inv_filt_fft = np.fft.rfft(inv_filt)
    RIR_fft = sine_sweep_fft * inv_filt_fft
    RIR = np.fft.ifft(RIR_fft)                                                # Repuesta al impulso
    RIR = RIR/np.max(abs(RI))
    # RI = RI[np.max(abs(RI)):(3*fs)]                                        # Se recorta el audio desde el máximo hasta 3 s
    RIR = RIR[np.where(abs(RIR) == np.max(abs(RIR))):(3*fs)]
    return RIR

if __name__ == '__main__': # hace que lo siguiente solo se ejecute al estar con ESTE script abierto, y NO llamando desde otro script
  RI = RIR_Generator(100, 1000, 10)