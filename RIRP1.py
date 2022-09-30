# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:38:27 2022

@author: Cori
"""

import numpy as np
import os
from scipy import signal
import json
import soundfile as sf
from scipy.signal import chirp, butter, sosfiltfilt
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
import time

class RIRP:
    def __init__(self):
        self.ir = None
        self.fs = None

    def load_signal(self, signal_path):
        """ Loads the signal (ir or sinesweep) to be procesed
        Args:
            signal_path (str): Path of the signal
        """
        self.ir, self.fs = sf.read(signal_path)
    
    def get_ir_from_sinesweep(self, f_min, f_max, T):
        """ Generates the ir from a sinesweep measurement
        Args:
            f_min (int): Initial frequency of the sweep [Hz]
            f_max (int): Final frequency of the sweep [Hz]
            T (int): Length of the sweep [s]
        """

        # Time lenght and array
        d = len(self.ir)/self.fs                                               
        t = np.arange(0, T, 1/self.fs)                                          
        
        # Generating chirp
        ss = chirp(t, f0=f_min, f1=f_max, t1=T, method='logarithmic')           
        inv_ss = ss[::-1]                                                               # Inverse chirp
        m = 1/(2*np.pi*np.exp(t*np.log(f_max/f_min)/T))                                 # Modulation
        inv_filt = m * inv_ss                                                           # Inverse filter
        inv_filt = inv_filt/np.amax(abs(inv_filt))                              
        inv_filt = np.pad(inv_filt, (0, int((d-T)*self.fs)), constant_values=(0, 0))    # Padding
        
        # Frequency operation to obtain ir
        sine_sweep_fft = np.fft.rfft(self.ir)
        inv_filt_fft = np.fft.rfft(inv_filt)
        ir_fft = sine_sweep_fft * inv_filt_fft
        ir = np.fft.ifft(ir_fft)                                                
        self.ir = ir/np.max(np.abs(ir))

    def get_reversed_ir(self):
        """ Temporarily flips the ir to avoid low-frequency overlap between
        the filter's impulse responses and the ir
        Returns:
            reversed_ir (numpy.array): Temporarily fliped ir
        """

        if np.ndim(self.ir) == 1:    # Case of single ir to be reversed
            reversed_ir = np.flip(self.ir)
        else:                   # Case of matrix of ir filtered to be reversed
            reversed_ir = np.flip(self.ir,axis = 1)

        return reversed_ir
    
    def get_ir_filtered(self, ir, bands_per_oct, bw_ir = None):
        """Filters the impulse response in octave or third octave bands with 
        6th order (in the case of the octave band) and 8th order (in the case
        of the third octave band) butterworth bandpass filters according to
        NORMA
        Args:
            ir (numpy array): Array of the impulse response
            self.fs (int): Sample rate
            bands_per_oct (int): Bands per octave
            bw_ir (list) (optional): Bandwidth of the impulse response to be filtered. 
            bw_ir[0]: lower boundary, bw_ir[1]: upper boundary. Default to None. 
        Returns:
            filtered_ir (numpy array): Array of size m*n where each row is the impulse
            response filtered with the m-th bandpass filter
            center_freqs (numpy array): Array with the center frequencies of the filters
        """
        if bands_per_oct == 1:
            filename = 'filters/octave_band_butterworths.json'
            filter_order = 6
            center_freqs = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
            
        elif bands_per_oct == 3:
            filename = 'filters/third_octave_band_butterworths.json'
            filter_order = 8
            center_freqs = np.array([31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                            400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                            4000, 5000, 6300, 8000, 10000, 12500, 16000])
            
        if bw_ir != None:
            center_freqs = center_freqs[bw_ir[0]:bw_ir[1] + 1]

        # Upper and lower boundaries frequencies
        jump_freq = np.power(2, 1 / (2 * bands_per_oct))
        lower_boundary_freqs = center_freqs / jump_freq
        upper_boundary_freqs = center_freqs * jump_freq
        upper_boundary_freqs[np.where(upper_boundary_freqs > self.fs/2)] = self.fs/2 - 1
        filtered_ir = np.zeros((len(center_freqs), len(ir)))

        # Generation of the bandpass filters and filtering of the IR
        if os.path.exists(filename):
            # Loading filters
            print('Loading filters')
            with open(filename) as json_obj:
                butterworth_filters = json.load(json_obj)

        else:
            # Generating and saving filters
            print('generating filters')
            butterworth_filters = {}
            for lower, upper in zip(lower_boundary_freqs, upper_boundary_freqs):  
                band_i = str(int(lower*jump_freq))
                butterworth_filter_i = butter(N = filter_order, Wn = np.array([lower, upper]), 
                                            btype='bandpass', analog=False , 
                                            output='sos', fs = self.fs)              # Generates the bandpass
                butterworth_filters[band_i] = butterworth_filter_i.tolist()
            with open(filename, 'w') as json_obj:
                json.dump(butterworth_filters, json_obj, indent = 4)
        
        bands = list(butterworth_filters.keys())

        for i in range(len(bands)):
            band_i = bands[i]
            filter_i = np.array(butterworth_filters[band_i])
            filtered_ir[i, :] = sosfiltfilt(filter_i, ir)           # Filters the IR

        return filtered_ir, center_freqs

    def get_chu_compensation(self, ir, percentage = 10):
        """ Calculates the noise RMS value according to Chu's method
        Args:
            ir (numpy.array): Array of the ir
            percentage (int, optional): Percentage of ir tail to be considered. Defaults to 10.
        Returns:
            noise_rms: Noise RMS value of ir tail
        """

        ir_len = len(ir)
        noise_start = int(np.round((1 - percentage / 100) * ir_len))
        ir_trimmed = ir[noise_start:]  # Trims the signal keeping only the last x% of itself as specified.
        noise_rms = np.mean(ir_trimmed)  # Calculates the mean squared value
        
        return noise_rms
    
    def get_lundeby_limit(self, ir):
        w = int(0.01 * self.fs)                                                             # 10 ms window
        t = int(len(ir)/w)                                                             # Steps
        
        RMS = lambda Signal: np.sqrt(np.mean(Signal**2))
        dB = lambda Signal: 10 * np.log10(abs(Signal/max(ir)))
        
        def root_mean_square(w, Signal, t):
            IR_RMS = np.zeros(t)
            for i in range(0, t):
                IR_RMS[i] = RMS(Signal[i*w:(i+1)*w])
            return dB(IR_RMS)
                
        IR_RMS_dB = root_mean_square(w, ir, t)
        
        # # 2. ESTIMATE BACKGROUND NOISE USING THE TAIL (square average of the last 10 %)
        # noise = ir[int(0.9*len(ir)):len(ir)]
        # noise_RMS = RMS(noise)                                                         
        # 
        
        noise_RMS = self.get_chu_compensation(ir)
        noise_dB = dB(noise_RMS)
        
        # 3. ESTIMATE SLOPE OF DECAY FROM 0 dB TO NOISE LEVEL + 10 dB
        fit_end = int(max(np.argwhere(IR_RMS_dB > noise_dB + 10)))
        
        x_axis = np.arange(0, (len(ir)/w))
        for i in range(0, t):
            x_axis[i] = int((w/2) + (i*w))
        
        m, b = np.polyfit(x_axis[0:fit_end], IR_RMS_dB[0:fit_end],1)                  # Linear regression
        
        # linear_fit = lambda x: (m * x) + b
        
        
        # 4. FIND PRELIMINARY CROSSPOINT (where the lineal regression meets the estimated noise)
        crosspoint = round((noise_dB - b)/m)         

        error = 1
        max_tries = 5                                                                 # According to Lundeby, 5 iterations is enough in all cases
        tries = 0
        
        while tries <= max_tries and error > 0.001:
        
           # 5. FIND NEW LOCAL TIME INTERVAL LENGTH
            delta = abs(10/m)                                                          # (X2 - X1) = (Y2 - Y1)/m  --> delta = Time interval required for a 10 dB drop
            p = 10                                                                     # Number of steps every 10 dB (Lundeby recommends between 3 and 10)
            w = int(delta / p)                                                         # Window: 10 steps (p) every 10 dB (delta)
            
            if (crosspoint - delta) > len(ir): 
                t = int(len(ir) / w)
            else:
                t = int(len(ir[0:int(crosspoint - delta)]) / w)
                
            # 6. AVERAGE SQUARED IMPULSE RESPONSE IN NEW LOCAL TIME INTERVALS  
            IR_RMS_dB = root_mean_square(w, ir, t)                                     
        
            # 7. ESTIMATE BACKGROUND NOISE LEVEL 
            noise = ir[int(crosspoint + delta): len(ir):]                              # 10 dB safety margin from crosspoint
            
            if len(noise) < (0.1 * len(ir)):                                           # Lundeby indicates to use at least 10% of the signal to estimate background noise level
                noise = ir[int(0.9 * len(ir)): len(ir):]
            noise_RMS = RMS(noise)                                                     
            noise_dB = dB(noise_RMS)
                
            # 8. ESTIMATE LATE DECAY SLOPE
            x_axis = np.arange(0, t)
            for i in range(0, t):
                x_axis[i] = int((w/2) + (i*w))
                
            m, b = np.polyfit(x_axis, IR_RMS_dB, 1)
            
            error = abs(crosspoint - ((noise_dB - b)/m))/crosspoint
            
            # 9. FIND NEW CROSSPOINT
            crosspoint = round((noise_dB - b)/m)
            print(crosspoint)
            
            tries += 1
        
        return crosspoint

    def get_smooth_by_schroeder(self, ir_matrix, crosspoint):
        """ Smooths the ir's energy with Schroeder's method.
        Args:
            ir_matrix (numpy.array): Array of the ir
            crosspoint (float): Time crosspoint to end Schroeder's integral
        Returns:
            smoothed_energy (numpy.array): Array of the smoothed energy
        """
        
        # Get ir energy
        energy = ir_matrix**2

        ## Iterar schroeder en cada fila de la matriz
        
        schroeder = np.pad(np.flip(np.cumsum(np.flip(energy[0:crosspoint:]))), (0, (len(energy) - crosspoint)))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed_energy = 10 * np.log10(schroeder / max(schroeder))
            
        return smoothed_energy

    def get_smooth_by_median_filter(self, ir_matrix, len_window):
        """ Smooths the ir's energy with Median Filter
        Args:
            ir_matrix (numpy.array): Array of the ir
            len_window (int): Length of the window in samples
        Returns:
            smoothed_energy (numpy.array): Array of the smoothed energy 
        """
        ## Elevar al cuadrado ir_matix
        smoothed_energy = np.zeros_like(ir_matrix)
        for freq in range(np.shape(ir_matrix)[0]):
            smoothed_energy[freq,:] = median_filter(ir_matrix[freq,:],len_window)
            
        return smoothed_energy
    
    def get_acoustical_parameters(self, smoothed_energy):   
        """ Calculates all acoustical parameters available:
        > EDT
        > T20
        > T30
        Args:
            smoothed_energy (numpy.array): Smoothed energy of the ir
        Returns:
            EDT: _description_ ... TO BE DONE
        """

        EDT = self.get_EDT(smoothed_energy)

        T20 = self.get_T20(smoothed_energy)
        
        T30 = self.get_T30(smoothed_energy)

        return EDT, T20, T30

    def get_EDT(self, smoothed_energy):
        """ Calculates the Energy Decay Time (EDT) of the ir
        Args:
            smoothed_energy (numpy.array): Smoothed energy of the ir
        Returns:
            EDT (float): Energy Decay Time in seconds
        """
        x_min_EDT = np.max(np.argwhere(smoothed_energy > -1))
        x_max_EDT = np.max(np.argwhere(smoothed_energy > -10))
        EDT = (x_max_EDT - x_min_EDT) / self.fs

        # m_EDT, b_EDT = np.polyfit((x_min_EDT, x_max_EDT), (smoothed_energy[x_min_EDT], smoothed_energy[x_max_EDT]),1)
        # EDT_linear_fit = lambda x: (m_EDT * x) + b_EDT
        return EDT

    def get_T20(self, smoothed_energy):
        """ Estimates the Reverberation Time (RT) using the -5 to -25 dB 
        range of the smoothed energy
        Args:
            smoothed_energy (numpy.array): Smoothed energy of the ir
        Returns:
            T20 (float): T20 in seconds
        """
        # T20
        x_min_TR = np.max(np.argwhere(smoothed_energy > -5))
        x_max_T20 = np.max(np.argwhere(smoothed_energy > -25))
        T20 = (x_max_T20 - x_min_TR) / self.fs
        
        # m_T20, b_T20 = np.polyfit((x_min_TR, x_max_T20), (smoothed_energy[x_min_TR], smoothed_energy[x_max_T20]),1)
        # T20_linear_fit = lambda x: (m_T20 * x) + b_T20

        return T20

    def get_T30(self, smoothed_energy):
        """ Estimates the Reverberation Time (RT) using the -5 to -35 dB 
        range of the smoothed energy
        Args:
            smoothed_energy (numpy.array): Smoothed energy of the ir
        Returns:
            T30 (float): T30 in seconds
        """
        x_min_TR = np.max(np.argwhere(smoothed_energy > -5))
        x_max_T30 = np.max(np.argwhere(smoothed_energy > -35))
        T30 = (x_max_T30 - x_min_TR) / self.fs
        
        # m_T30, b_T30 = np.polyfit((x_min_TR, x_max_T30), (smoothed_energy[x_min_TR], smoothed_energy[x_max_T30]),1)
        # T30_linear_fit = lambda x: (m_T30 * x) + b_T30
        return T30


if __name__ == '__main__':
    RIRP_instance = RIRP()
    signal_path = 'audio_tests/rirs/RI_1.wav'
    f_min = 125
    f_max = 16_000
    T = 3
    # sinesweep, fs = RIRP_instance.load_signal(signal_path)
    # ir = RIRP_instance.get_ir_from_sinesweep(sine_sweep = sinesweep, fs = fs ,f_min = f_min , f_max = f_max , T = T)
    ir, fs = RIRP_instance.load_signal(signal_path)
    ir_reversed = RIRP_instance.get_reversed_ir(ir)
    filtered_ir, center_freqs =  RIRP_instance.get_ir_filtered(ir = ir_reversed, fs = fs, bands_per_oct = 1)
    filtered_ir = RIRP_instance.get_reversed_ir(filtered_ir)**2
    smoothed_ir = RIRP_instance.get_smooth_by_median_filter(filtered_ir,1000)
    crosspoint = RIRP_instance.get_lundeby_limit()
    print('hola')