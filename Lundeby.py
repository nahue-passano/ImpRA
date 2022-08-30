# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:32:33 2022

@author: Cori
"""
# Chequeado segun código de Marino da bien. Usé la media cuadratica, con la aritmética algo sale mal


import numpy as np
import soundfile as sf

path = 'D:/Desktop/UNTREF/Instrumentos y Mediciónes Acústicas/TP 10 - RiRs/Código/RIRs_prueba/RI_6.wav'

# path = 'D:/Desktop/UNTREF/Instrumentos y Mediciónes Acústicas/TP 10 - RiRs/Trabajos anteriores/TP10_Atin/Example RIRs/IR_1.wav'

# def Lundeby(path):

IR, fs = sf.read(path)                                                         # Import IR audio
IR = IR[np.argmax(IR):len(IR)]

# 1. AVERAGE SQUARED IMPULSE RESPONSE IN LOCAL TIME INTERVALS (10 ms)

w = int(0.01 * fs)                                                             # 10 ms window
t = int(len(IR)/w)                                                             # Steps

RMS = lambda signal: np.sqrt(np.mean(signal**2))
dB = lambda signal: 10 * np.log10(abs(signal/max(IR)))

# # CON MEDIA ARITMÉTICA
# def arithmetic_mean(w, signal, t):
#     IR_prom = np.zeros(t)
#     for i in range(t):
#         IR_prom[i] = np.mean(signal[i*w:(i+1)*w]) 
#     return dB(IR_prom)

# IR_prom_dB = arithmetic_mean(w, IR, t)

# CON MEDIA CUADRÁTICA
def root_mean_square(w, signal, t):
    IR_RMS = np.zeros(t)
    for i in range(0, t):
        IR_RMS[i] = RMS(IR[i*w:(i+1)*w])
    return dB(IR_RMS)
        
IR_RMS_dB = root_mean_square(w, IR, t)


# 2. ESTIMATE BACKGROUND NOISE USING THE TAIL (square average of the last 10 %)
noise = IR[int(0.9*len(IR)):len(IR)]

# noise_mean = np.mean(noise)                                                    # CON MEDIA ARITMÉTICA
# noise_dB = dB(noise_mean)

noise_RMS = RMS(noise)                                                         # CON MEDIA CUADRÁTICA
noise_dB = dB(noise_RMS)


# 3. ESTIMATE SLOPE OF DECAY FROM 0 dB TO NOISE LEVEL + 10 dB
fit_end = int(max(np.argwhere(IR_RMS_dB > noise_dB + 10)))

x_axis = np.arange(0, (len(IR)/w))
                  
for i in range(0, t):
    x_axis[i] = int((w/2) + (i*w))

m, b = np.polyfit(x_axis[0:fit_end], IR_RMS_dB[0:fit_end],1)  # Linear regression

linear_fit = lambda x: (m * x) + b


# 4. FIND PRELIMINARY CROSSPOINT (where the lineal regression meets the estimated noise)
crosspoint = round((noise_dB - b)/m)         
print(crosspoint)

error = 1
max_tries = 5                                                                 # According to Lundeby, 5 iterations is enough in all cases
tries = 1

while tries <= max_tries and error > 0.001:

   # 5. FIND NEW LOCAL TIME INTERVAL LENGTH
    delta = abs(10/m)                                                          # (X2 - X1) = (Y2 - Y1)/m  --> delta = Time interval required for a 10 dB drop
    p = 10                                                                     # Number of steps every 10 dB (Lundeby recommends between 3 and 10)
    w = int(delta / p)                                                         # Window: 10 steps (p) every 10 dB (delta)
    
    # t = int(len(IR)/w) 
    
    if (crosspoint - delta) > len(IR):
        t = int(len(IR) / w)
    else:
        t = int(len(IR[0:int(crosspoint - delta)]) / w)
    
        
    # 6. AVERAGE SQUARED IMPULSE RESPONSE IN NEW LOCAL TIME INTERVALS
    # IR_prom_dB = arithmetic_mean(w, IR, t)                                   # CON MEDIA ARITMÉTICA    
    IR_RMS_dB = root_mean_square(w, IR, t)                                     # CON MEDIA CUADRÁTICA

    
    # 7. ESTIMATE BACKGROUND NOISE LEVEL 
    noise = IR[int(crosspoint + delta): len(IR):]                              # 10 dB safety margin from crosspoint
    
    if len(noise) < (0.1 * len(IR)):                                           # Lundeby indicates to use at least 10% of the signal to estimate background noise level
        noise = IR[int(0.9 * len(IR)): len(IR):]
        
    # noise_mean = np.mean(noise)                                              # CON MEDIA ARITMÉTICA
    # noise_dB = dB(noise_mean)
    
    noise_RMS = RMS(noise)                                                     # CON MEDIA CUADRÁTICA
    noise_dB = dB(noise_RMS)
        
    # 8. ESTIMATE LATE DECAY SLOPE

    x_axis = np.arange(0, t)
                       
    for i in range(0, t):
        x_axis[i] = int((w/2) + (i*w))
        
    m, b = np.polyfit(x_axis, IR_RMS_dB, 1)

    # fit_start = int(max(np.argwhere(IR_RMS_dB > noise_dB + 30)))
    # fit_end = int(max(np.argwhere(IR_RMS_dB > noise_dB + 10)))
    
    # m, b = np.polyfit(x_axis[fit_start:fit_end], IR_RMS_dB[fit_start:fit_end], 1)
    
    error = abs(crosspoint - ((noise_dB - b)/m))/crosspoint
    
    # 9. FIND NEW CROSSPOINT
    crosspoint = round((noise_dB - b)/m)
    print(crosspoint)
    tries += 1
