# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 16:01:02 2022

@author: Cori
"""

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
    schroeder = 10 * np.log10(schroeder / max(schroeder))
    IR = 10 * np.log10(IR/max(IR))

plt.plot(IR)
plt.plot(schroeder)
plt.show()




