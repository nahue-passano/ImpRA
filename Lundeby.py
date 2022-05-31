# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:32:33 2022

@author: Cori
"""
# Lundeby dice que hay que usar la media cuadrática, Maite y Marino usan la media aritmética, 3
# Rebora usa la cuadrática. Dan resultados distintos. Dejé ambas opciones (comentar la indeseada).

# Problema en el while cuando pongo la condición "error > 0.0001 and ..."

# No puedo graficar bien lo que está pasando. No se si es que estoy haciendolo mal o graficándolo mal.

# Segun el paper, no hay que iterar el promedio cuadrático (paso 6.), pero vi que Maite y Marino lo metieron en el loop.


import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import math

path = 'D:/Desktop/UNTREF/Instrumentos y Mediciónes Acústicas/TP10 - RiRs/Código/RIRs_prueba/RI_6.wav'
# def Lundeby(path):

IR, fs = sf.read(path)                                      # Import IR audio
IR = IR[np.argmax(IR):len(IR)]

# 1. AVERAGE SQUARED IMPULSE RESPONSE IN LOCAL TIME INTERVALS (10 ms)

w = int(0.01 * fs)                                          # 10 ms window
t = int(len(IR)/w)                                          # Steps

RMS = lambda signal: np.sqrt(np.mean(signal**2))
dB = lambda signal: 10 * np.log10(abs(signal/max(IR)))

# CON MEDIA ARITMÉTICA
IR_prom = np.zeros(int(len(IR)/w))

eje_tiempo =np.zeros(t)
for i in range(int(len(IR)/w)):
    IR_prom[i] = np.mean(IR[i*w:(i+1)*w])
    eje_tiempo[i] = math.ceil(w / 2) + (i * w)    
IR_RMS_dB = dB(IR_prom)

# CON MEDIA CUADRÁTICA
# IR_RMS = np.zeros(t)
# eje_tiempo = np.zeros(t)
# for i in range(0, t):
#     IR_RMS[i] = RMS(IR[i*w:(i+1)*w])
#     eje_tiempo[i] = math.ceil(w / 2) + (i * w)
    
# IR_RMS_dB = dB(IR_RMS)


# 2. ESTIMATE BACKGROUND NOISE USING THE TAIL (square average of the last 10 %)
noise = IR[int(0.9*len(IR)):len(IR)]
# noise_RMS = RMS(noise)          # CON MEDIA CUADRÁTICA
noise_RMS = np.mean(noise)    # CON MEDIA ARITMÉTICA
noise_dB = dB(noise_RMS)

# 3. ESTIMATE SLOPE OF DECAY FROM 0 dB TO NOISE LEVEL + 10 dB
not_noise_index = int(max(np.argwhere(IR_RMS_dB > noise_dB + 10)))
x = np.arange(0, len(IR))
m, b = np.polyfit(x[0:not_noise_index], IR_RMS_dB[0:not_noise_index],1)            # Linear regression    

linear_fit = lambda x: (m * x) + b

# 4. FIND PRELIMINARY CROSSPOINT (where the lineal regression meets the estimated noise)
crosspoint = (noise_dB - b)/m         
 
# plt.plot(eje_tiempo, IR_RMS_dB)
# plt.plot(eje_tiempo, linear_fit(eje_tiempo))
# plt.grid()
# plt.show()                     # ESTOS GRÁFICOS NO ESTAN BIEN, PERO NO SE POR QUÉ, ANTES ESTABAN BIEN

error = 1
max_tries = 2                    # According to Lundeby, 5 iterations is enough in all cases
tries = 0

while tries <= max_tries:        # ME EXPLOTA TODO CUANDO PONGO LA CONDICIÓN error > 0.001

    # 5. FIND NEW LOCAL TIME INTERVAL LENGTH
    delta = abs(10/m)            # (X2 - X1) = (Y2 - Y1)/m  --> Time interval required for a 10 dB drop
    p = 10                       # Number of steps every 10 dB (Lundeby recommends between 3 and 10)
    w = int(delta / p)           # Window: 10 steps (p) every 10 dB (delta)
    
    if (crosspoint - delta) > len(IR):
        t = int(len(IR) / w)
    else:
        t = int(len(IR[0:int(crosspoint - delta)]) / w)
    if t < 2:
        t = 2
        
    # 6. AVERAGE SQUARED IMPULSE RESPONSE IN NEW LOCAL TIME INTERVALS
    IR_RMS_w = np.zeros(t)
    eje_tiempo_w = np.zeros(t)
    for i in range(0, t):
        # IR_RMS_w[i] = RMS(IR[i*w:(i+1)*w])                # CON MEDIA CUADRÁTICA
        IR_RMS_w[i] = np.mean(IR[i*w:(i+1)*w])          # CON MEDIA ARITMÉTICA
        eje_tiempo_w[i] = math.ceil(w / 2) + (i * w)
    IR_RMS_dB_w = dB(IR_RMS_w)

    # SEGÚN INTERPRETÉ DEL PAPER, LOS PUNTOS 5. Y 6. NO DEBERÍAN ESTAR DENTRO DEL LOOP, LA ITERACIÓN ES SOLO DE 7., 8. Y 9.
    # PERO MAITE Y MARINO LO METIERON ADENTRO. AL FINAL COMO NO ME SALÍAN LAS COSAS ME RENDÍ Y TRATÉ DE HACERLO COMO ELLOS (TAMPOCO FUNCIONÓ JE)

    # 7. ESTIMATE BACKGROUND NOISE LEVEL 
    noise = IR[int(crosspoint + delta): len(IR):]
    
    if len(noise) < (0.1 * len(IR)):                   # Lundeby indicates to use at least 10% of the signal to estimate background noise level
        noise = IR[int(0.9 * len(IR)): len(IR):]
    
    # noise_RMS = RMS(noise)              # CON MEDIA CUADRÁTICA
    noise_RMS = np.mean(noise)        # CON MEDIA ARITMÉTICA
    noise_dB = dB(noise_RMS)
    
    # 8. ESTIMATE LATE DECAY SLOPE
    m, b = np.polyfit(eje_tiempo_w, IR_RMS_dB_w, 1)  
    
    
    # ACÁ DEJO COMENTADA OTRA FORMA DE ENCONTRAR LOS LÍMITES PARA EL POLYFIT. PARA MÍ ES ASÍ COMO PIDE LUNDEBY, PERO NO ME FUNCIONA BIEN.
    # fit_start = int(crosspoint - (2 * delta))        # Linear regression starts 20 dB above crosspoint
    # fit_end = int(crosspoint - delta)                # Linear regression ends 10 dB above crosspoint
    # x = np.arange(0, (fit_end - fit_start))    
    # m,b = np.polyfit(x, IR_RMS_dB[fit_start : fit_end:], 1) # ESTO SALE MAL PORQUE ME QUEDA NEGATIVO EL FIT_START WAESRDTFGYHJKML
    
    error = abs(crosspoint - ((noise - b)/m))/crosspoint
    
    # 9. FIND NEW CROSSPOINT
    crosspoint = (noise_dB - b)/m
    
    tries += 1

if crosspoint > len(IR):
    crosspoint = len(IR)

# GRÁFICO

N = np.arange(len(eje_tiempo))
for i in N:
    N[i] = noise_dB

# eje_tiempo = eje_tiempo / fs
# plt.plot(eje_tiempo, IR_RMS_dB)
# plt.plot(eje_tiempo, N)
# plt.plot(eje_tiempo, N + 10)
# plt.plot(eje_tiempo, linear_fit(eje_tiempo)) # DESFASADA Y CON MUCHA MAS PENDIENTE DE LO QUE DEBERIA. 
                                               # No se si es problema de la regresión o si es problema de como está interpretado el gráfico
# plt.ylim(-100, 0)
# plt.scatter(crosspoint, noise_dB, marker = 'o')
# plt.scatter(crosspoint - delta, noise_dB, marker = 'o')
# plt.plot(x, N + 10)

# plt.grid()
# plt.show()