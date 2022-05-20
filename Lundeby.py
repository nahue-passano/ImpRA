# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:14:53 2022

@author: Cori
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

path = 'D:/Desktop/UNTREF/Instrumentos y Mediciónes Acústicas/TP10 - RiRs/Código/RIRs_prueba/RI_6.wav'
# def Lundeby(path):

IR, fs = sf.read(path)                                 # Import IR audio
IR = IR[np.argmax(IR):len(IR)]

w = int(0.01 * fs)                                          # 10 ms window
t = int(len(IR)/w)                                          # Steps

RMS = lambda signal: np.sqrt(np.mean(signal**2))
dB = lambda signal: 10 * np.log10(abs(signal/max(IR)))

media = np.zeros(t)
eje_tiempo = np.zeros(t)

# Divide in sections and calculate the mean

for i in range(0, t):
    media[i] = np.mean(IR[i * w:(i + 1) * w])
    eje_tiempo[i] = int(np.ceil(w/2)) + (i*w)
    
# Calculate noise level of the last 10% of the signal

noise_rms_dB = 10 * np.log10(abs(np.sum(IR[round(0.9 * len(IR)):]) / (0.1 * len(IR)) / max(IR)))
mediadB = dB(media)

# Se busca la regresión lineal del intervalo de 0dB y la media mas proxima al ruido + 10dB.
# Calculate linear regression between the 0 dB 

try:
    r = int(max(np.argwhere(mediadB > noise_rms_dB + 10)))
       
    if np.any(mediadB[0:r] < noise_rms_dB+10):
        r = min(min(np.where(mediadB[0:r] < noise_rms_dB + 10)))
    if np.all(r==0) or r<10:
        r=10
except:
    r = 10

m, b = np.polyfit(eje_tiempo[0:r], mediadB[0:r], 1) # Linear regression

crosspoint = int((noise_rms_dB-b)/m)

linear_fit = lambda x: (m * x) + b

# Insufficient SNR

if noise_rms_dB > -20:
    punto = len(IR)
    C = None
    
else:
    error = 1
    veces = 1
    max_tries = 5 # According to Lundeby, 5 iterations is enough in all cases      
    
    while error > 0.0004 and veces <= max_tries:
        
        p = 10                      # Number of steps every 10 dB (Lundeby recommends between 3 and 10)        
        delta = int(abs(10/m))      # (X2 - X1) = (Y2 - Y1)/m  --> Time interval required for a 10 dB drop
        
        w = int(np.floor(delta/p))  # Window: 10 steps (p) every 10 dB (delta)
        t = int(np.floor(len(IR[:int(crosspoint - delta)])/w))
        
        if t < 2:
            t = 2
        elif np.all(t == 0):
            t = 2

        media = np.zeros(t)
        eje_tiempo = np.zeros(t)
        
        for i in range(0, t):
            media[i] = np.mean(IR[i*w:(i + 1) * w])
            eje_tiempo[i] = np.ceil(w / 2) + int(i * w)
            
        mediadB = dB(media)
        m, b = np.polyfit(eje_tiempo, mediadB, 1)

        # Nueva media de energia de ruido, comenzando desde desde el punto de la linea de 
        # decaimiento, 10 dB por debajo del punto de cruce
        
        # New noise average level, starting from the point of the 
        # decay curve, 10 dB below the intersection.
        
        noise = IR[int(abs(crosspoint + delta)):]
        
        if len(noise) < round(0.1 * len(IR)):
            noise = IR[round(0.9 * len(IR)):]
            
        noise_rms_dB = dB(sum(noise)/ len(noise))

        # New intersection index
        
        error = abs(crosspoint - (noise_rms_dB - b) / m) / crosspoint
        crosspoint = round((noise_rms_dB - b) / m)
        
        
        x = np.arange(len(mediadB))
        
        plt.plot(eje_tiempo, mediadB)
        plt.plot(eje_tiempo, linear_fit(x))
        plt.show()
        veces += 1
               
# Output validation
        
if crosspoint > len(IR):
    punto = len(IR)
else:
    punto = int(crosspoint)
    
C = max(IR) * 10 ** (b / 10) * np.exp(m/10/np.log10(np.exp(1))*crosspoint) / (
    -m / 10 / np.log10(np.exp(1)))

print(punto)


    