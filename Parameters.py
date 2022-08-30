# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:45:20 2022

@author: Cori
"""
# T10 (de -5 dB a -15 dB)
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

path = 'D:/Desktop/UNTREF/Instrumentos y Mediciónes Acústicas/TP 10 - RiRs/Código/RIRs_prueba/RI_3.wav'

crosspoint = 31218

IR, fs = sf.read(path)
IR = IR[np.argmax(IR):len(IR)]

for i in range(len(IR)):
    IR[i] = IR[i]**2

schroeder = np.pad(np.flip(np.cumsum(np.flip(IR[0:crosspoint:]))), (0, (len(IR) - crosspoint)))

with np.errstate(divide='ignore', invalid='ignore'):
    schroeder_dB = 10 * np.log10(schroeder / max(schroeder))
    IR_dB = 10 * np.log10(IR/max(IR))

x_axis = np.arange(len(IR))

# EDT
x_min = np.max(np.argwhere(IR_dB > -1))
x_max = np.max(np.argwhere(IR_dB > -10))
m, b = np.polyfit((x_min, x_max), (schroeder_dB[x_min], schroeder_dB[x_max]),1)

y = lambda x: (m*x)+b

EDT = (x_max - x_min) / fs

# T20
x_min = np.max(np.argwhere(IR_dB > -5))
x_max = np.max(np.argwhere(IR_dB > -25))
m, b = np.polyfit((x_min, x_max), (schroeder_dB[x_min], schroeder_dB[x_max]),1)

y = lambda x: (m*x)+b

T20 = (x_max - x_min) / fs


# T30
x_min = np.max(np.argwhere(IR_dB > -5))
x_max = np.max(np.argwhere(IR_dB > -35))
m, b = np.polyfit((x_min, x_max), (schroeder_dB[x_min], schroeder_dB[x_max]),1)

y = lambda x: (m*x)+b

T30 = (x_max - x_min) / fs

plt.plot(y(x_axis))
plt.plot(IR_dB)
plt.plot(schroeder_dB)
plt.xlim(0, 35000)
plt.ylim(-90, 0)
plt.show()


# C50
energia = IR # está elevada al 2
C50 = 10 * np.log10(np.cumsum(energia[0: (50 * fs)]) / np.cumsum(energia[(50 * fs):len(energia)]))
C80 = 10 * np.log10(np.cumsum(energia[0: (80 * fs)]) / np.cumsum(energia[(80 * fs):len(energia)]))



