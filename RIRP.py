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
        self.IR = None
        self.fs = None

    def load_signal(self, signal_path):
        """ Loads the signal (ir or sinesweep) to be procesed

        Args:
            signal_path (str): Path of the signal
        """
        self.IR, self.fs = sf.read(signal_path)
    
    def get_ir_from_sinesweep(self, f_min, f_max, T):
        """ Generates the ir from a sinesweep measurement

        Args:
            f_min (int): Initial frequency of the sweep [Hz]
            f_max (int): Final frequency of the sweep [Hz]
            T (int): Length of the sweep [s]
        """

        # Time lenght and array
        d = len(self.IR)/self.fs                                               
        t = np.arange(0, T, 1/self.fs)                                          
        
        # Generating chirp
        ss = chirp(t, f0=f_min, f1=f_max, t1=T, method='logarithmic')           
        inv_ss = ss[::-1]                                                               # Inverse chirp
        m = 1/(2*np.pi*np.exp(t*np.log(f_max/f_min)/T))                                 # Modulation
        inv_filt = m * inv_ss                                                           # Inverse filter
        inv_filt = inv_filt/np.amax(abs(inv_filt))                              
        inv_filt = np.pad(inv_filt, (0, int((d-T) * self.fs)), constant_values=(0, 0))    # Padding
        
        # Frequency operation to obtain ir
        sine_sweep_fft = np.fft.rfft(self.IR)
        inv_filt_fft = np.fft.rfft(inv_filt)
        IR_fft = sine_sweep_fft * inv_filt_fft
        IR = np.fft.ifft(IR_fft)                                                
        self.IR = IR/np.max(np.abs(IR))

    def get_reversed_IR(self):
        """ Temporarily flips the ir to avoid low-frequency overlap between
        the filter's impulse responses and the ir

        Returns:
            reversed_IR (numpy.array): Temporarily fliped IR
        """

        if np.ndim(self.IR) == 1:    # Case of single ir to be reversed
            reversed_IR = np.flip(self.IR)
        else:                   # Case of matrix of ir filtered to be reversed
            reversed_IR = np.flip(self.IR,axis = 1)

        return reversed_IR
    
    def get_ir_filtered(self, IR, bands_per_oct, bw_IR = None):
        """Filters the impulse response in octave or third octave bands with 
        6th order (in the case of the octave band) and 8th order (in the case
        of the third octave band) butterworth bandpass filters according to
        NORMA

        Args:
            IR (numpy array): Array of the impulse response
            self.fs (int): Sample rate
            bands_per_oct (int): Bands per octave
            bw_IR (list) (optional): Bandwidth of the impulse response to be filtered. 
            bw_IR[0]: lower boundary, bw_IR[1]: upper boundary. Default to None. 

        Returns:
            filtered_IR (numpy array): Array of size m*n where each row is the impulse
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
            
        if bw_IR != None:
            center_freqs = center_freqs[bw_IR[0]:bw_IR[1] + 1]

        # Upper and lower boundaries frequencies
        jump_freq = np.power(2, 1 / (2 * bands_per_oct))
        lower_boundary_freqs = center_freqs / jump_freq
        upper_boundary_freqs = center_freqs * jump_freq
        upper_boundary_freqs[np.where(upper_boundary_freqs > self.fs/2)] = self.fs/2 - 1
        filtered_IR = np.zeros((len(center_freqs), len(IR)))

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
            filtered_IR[i, :] = sosfiltfilt(filter_i, IR)           # Filters the IR

        return filtered_IR, center_freqs

    def get_chu_compensation(self, IR, percentage = 10):
        """ Calculates the noise RMS value according to Chu's method

        Args:
            IR (numpy.array): Array of the IR
            percentage (int, optional): Percentage of ir tail to be considered. Defaults to 10.

        Returns:
            noise_rms: Noise RMS value of ir tail
        """

        IR_len = len(IR)
        noise_start = int(np.round((1 - percentage / 100) * IR_len))
        IR_trimmed = IR[noise_start:]  # Trims the signal keeping only the last x% of itself as specified.
        noise_rms = np.mean(IR_trimmed)  # Calculates the mean squared value
        
        return noise_rms
    
    def get_lundeby_limit(self, IR):
        """ Calculates Schroeder's integral limit according to Lundeby's method

        Args:
            IR (numpy.array): Array of the IR

        Returns:
            crosspoint: Index from which the energy of the signal corresponds to background noise.
        """
        w = int(0.01 * self.fs)                                                        # 10 ms window
        t = int(len(IR)/w)                                                             # Steps
        
        RMS = lambda Signal: np.sqrt(np.mean(Signal**2))
        dB = lambda Signal: 10 * np.log10(abs(Signal/max(IR)))
        
        def root_mean_square(w, Signal, t):
            IR_RMS = np.zeros(t)
            for i in range(0, t):
                IR_RMS[i] = RMS(Signal[i*w:(i+1)*w])
            return dB(IR_RMS)
                
        IR_RMS_dB = root_mean_square(w, IR, t)
        
        # 2. ESTIMATE BACKGROUND NOISE USING THE TAIL (square average of the last 10 %)
        noise_RMS = self.get_chu_compensation(IR)
        noise_dB = dB(noise_RMS)
        
        # 3. ESTIMATE SLOPE OF DECAY FROM 0 dB TO NOISE LEVEL + 10 dB
        fit_end = int(max(np.argwhere(IR_RMS_dB > noise_dB + 10)))
        
        x_axis = np.arange(0, (len(IR)/w))
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
            
            if (crosspoint - delta) > len(IR): 
                t = int(len(IR) / w)
            else:
                t = int(len(IR[0:int(crosspoint - delta)]) / w)
                
            # 6. AVERAGE SQUARED IMPULSE RESPONSE IN NEW LOCAL TIME INTERVALS  
            IR_RMS_dB = root_mean_square(w, IR, t)                                     
        
            # 7. ESTIMATE BACKGROUND NOISE LEVEL 
            noise = IR[int(crosspoint + delta): len(IR):]                              # 10 dB safety margin from crosspoint
            
            if len(noise) < (0.1 * len(IR)):                                           # Lundeby indicates to use at least 10% of the signal to estimate background noise level
                noise = IR[int(0.9 * len(IR)): len(IR):]
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

    def get_smooth_by_schroeder(self, IR_matrix, crosspoint):
        """ Smooths the ir's energy with Schroeder's method.

        Args:
            IR_matrix (numpy.array): Array of the IR
            crosspoint (float): Time crosspoint to end Schroeder's integral

        Returns:
            smoothed_energy (numpy.array): Array of the smoothed energy
        """
        
        # Get ir energy
        energy = IR_matrix**2

        ## Iterar schroeder en cada fila de la matriz
        
        schroeder = np.pad(np.flip(np.cumsum(np.flip(energy[0:crosspoint:]))), (0, (len(energy) - crosspoint)))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed_energy = 10 * np.log10(schroeder / max(schroeder))
            
        return smoothed_energy

    def get_smooth_by_median_filter(self, IR_matrix, len_window):
        """ Smooths the IR's energy with Median Filter

        Args:
            IR_matrix (numpy.array): Array of the IR
            len_window (int): Length of the window in samples

        Returns:
            smoothed_energy (numpy.array): Array of the smoothed energy 
        """
        ## Elevar al cuadrado IR_matix
        smoothed_energy = np.zeros_like(IR_matrix)
        for freq in range(np.shape(IR_matrix)[0]):
            smoothed_energy[freq,:] = median_filter(IR_matrix[freq,:], len_window)
    
        return smoothed_energy

    def get_acoustical_parameters(self, smoothed_energy):   
        """ Calculates acoustical parameters from the energy time cuve
        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            Tt: Transition Time
            EDTt: Early Decay Transition Time
            EDT: Early Decay Time. Reverberation Time extrapolated from a decay of 10 dB
            T20: Reverberation Time extrapolated from a decay of 20 dB
            T30: Reverberation Time extrapolated from a decay of 20 dB
            C50: Clarity of speech
            C80: Clarity of music
            IACC_Early: Interaural Cross-Correlation Coefficient for Early integration
        """
        
        Tt = self.get_Tt_EDTt(smoothed_energy)[0]
        
        EDTt = self.get_Tt_EDTt(smoothed_energy)[1]

        EDT = self.get_EDT(smoothed_energy)

        T20 = self.get_T20(smoothed_energy)
        
        T30 = self.get_T30(smoothed_energy)
        
        C50 = self.get_C50(smoothed_energy)
        
        C80 = self.get_C80(smoothed_energy)
        
        # IACC_Early = self.get_IACC_Early(IR_L, IR_R) CHEQUEAR, NO SE SI PARA EL IACC SE ENRA CON R Y L SEPARADAS O KE

        return Tt, EDTt, EDT, T20, T30, C50, C80
    
    def get_Tt_EDTt(self, smoothed_energy):
        """ Calculates the Transition Time (Tt) of the IR

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            Tt (float): Transition time in seconds
        """
        # No estoy segura si es necesario elevar al 2 la smoothed_energy o no.
        Tt = np.max(np.where(np.cumsum(smoothed_energy**2) <= 0.99 * np.max(np.sum(smoothed_energy**2)))) / self.fs

        smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_EDTt = np.max(np.argwhere(smoothed_energy_dB > -1))
        x_max_EDTt = Tt
        EDTt = (60/(-1 - smoothed_energy_dB[int(x_max_EDTt)]))*(x_max_EDTt - x_min_EDTt) / self.fs
        
        # m_EDTt, b_EDTt = np.polyfit((x_min_EDTt, x_max_EDTt), (smoothed_energy_dB[x_min_EDTt], smoothed_energy_dB[x_max_EDTt]), 1)
        # EDTt_linear_fit = lambda x: (m_EDTt * x) + b_EDTt
        
        return Tt, EDTt       
        
    def get_EDT(self, smoothed_energy):
        """ Calculates the Energy Decay Time (EDT) of the IR

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            EDT (float): Energy Decay Time in seconds
        """
        smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_EDT = np.max(np.argwhere(smoothed_energy_dB > -1))
        x_max_EDT = np.max(np.argwhere(smoothed_energy_dB > -10))
        EDT = (60/9) * (x_max_EDT - x_min_EDT) / self.fs

        # m_EDT, b_EDT = np.polyfit((x_min_EDT, x_max_EDT), (smoothed_energy_dB[x_min_EDT], smoothed_energy_dB[x_max_EDT]), 1)
        # EDT_linear_fit = lambda x: (m_EDT * x) + b_EDT
        return EDT

    def get_T20(self, smoothed_energy):
        """ Estimates the Reverberation Time (RT) using the -5 to -25 dB 
        range of the smoothed energy

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            T20 (float): T20 in seconds
        """
        smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_T20 = np.max(np.argwhere(smoothed_energy_dB > -5))
        x_max_T20 = np.max(np.argwhere(smoothed_energy_dB > -25))
        T20 = 3 * (x_max_T20 - x_min_T20) / self.fs
        
        # m_T20, b_T20 = np.polyfit((x_min_T20, x_max_T20), (smoothed_energy_dB[x_min_T20], smoothed_energy_dB[x_max_T20]), 1)
        # T20_linear_fit = lambda x: (m_T20 * x) + b_T20

        return T20

    def get_T30(self, smoothed_energy):
        """ Estimates the Reverberation Time (RT) using the -5 to -35 dB 
        range of the smoothed energy

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            T30 (float): T30 in seconds
        """
        smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_T30 = np.max(np.argwhere(smoothed_energy_dB > -5))
        x_max_T30 = np.max(np.argwhere(smoothed_energy_dB > -35))
        T30 = 2 * (x_max_T30 - x_min_T30) / self.fs
        
        # m_T30, b_T30 = np.polyfit((x_min_T30, x_max_T30), (smoothed_energy_dB[x_min_TR], smoothed_energy_dB[x_max_T30]), 1)
        # T30_linear_fit = lambda x: (m_T30 * x) + b_T30
        
        return T30
    
    def get_C50(self, smoothed_energy):
        C50 = 10 * np.log10(np.sum(smoothed_energy[0: int(0.05 * fs)]) / np.sum(smoothed_energy[int(0.05 * fs):len(smoothed_energy)]))
        
        return C50
    
    def get_C80(self, smoothed_energy):
        C80 = 10 * np.log10(np.sum(smoothed_energy[0: int(0.08 * fs)]) / np.sum(smoothed_energy[int(0.08 * fs):len(smoothed_energy)]))
        
        return C80
    
    def get_IACC_Early(self, IR_L, IR_R):        
        # Hice la función entrando con las señales L y R separadas, pero no se si va a ser así, o si hay que entrar
        # con una matriz donde estén incluidas ambas señales y acomodar el código a eso.
        
        # En la fórmula tal como está codeada acá, el IR_L e IR_R no deben estar elevados al 2!!!!
        
        num = np.correlate(IR_L[0: int(0.8 * self.fs)], IR_R[0: int(0.08 * self.fs)])
        den = np.sqrt(np.sum(IR_L[0: int(0.8 * self.fs)]**2) * (np.sum(IR_R[0: int(0.8 * self.fs)]**2)))
        IACC_Early = np.max(np.abs(num/den))
        
        return IACC_Early
        

if __name__ == '__main__':
    RIRP_instance = RIRP()
    signal_path = 'audio_tests/rirs/RI_1.wav'
    # signal_path = 'D:/Desktop/UNTREF/Instrumentos y Mediciónes Acústicas/TP 10 - RiRs/Código/RIRs_prueba/RI_1.wav'
    # f_min = 125
    # f_max = 16000
    # T = 3
    # sinesweep, fs = RIRP_instance.load_signal(signal_path)
    # IR = RIRP_instance.get_IR_from_sinesweep(sine_sweep = sinesweep, fs = fs ,f_min = f_min , f_max = f_max , T = T)
    IR, fs = RIRP_instance.load_signal(signal_path)
    IR_reversed = RIRP_instance.get_reversed_IR(IR)
    filtered_IR, center_freqs =  RIRP_instance.get_IR_filtered(IR = IR_reversed, fs = fs, bands_per_oct = 1)
    filtered_IR = RIRP_instance.get_reversed_IR(filtered_IR)**2
    # smoothed_IR = RIRP_instance.get_smooth_by_median_filter(filtered_IR,1000)
    crosspoint = RIRP_instance.get_lundeby_limit()
    smoothed_IR = RIRP_instance.get_smooth_by_schroeder(filtered_IR, crosspoint)
    parameters = RIRP_instance.get_acoustical_parameters(smoothed_IR)
    print('hola')
    
    # # CASO DE IR BINAURAL (????)
    # signal_path = 'audio_tests/rirs/RI_1.wav'
    # IR, fs = RIRP_instance.load_signal(signal_path)
    # if np.size(self.IR, 1) == 2: 
        
        # # LEFT
        # self.IR = self.IR[:, 0]  (???) lo escribí asi intuitivamente pero no se si funcará
        # IR_L_reversed = RIRP_instance.get_reversed_IR(IR_L)
        # filtered_IR_L, center_freqs =  RIRP_instance.get_IR_filtered(IR = IR_reversed_L, fs = fs, bands_per_oct = 1)
        # filtered_IR_L = RIRP_instance.get_reversed_IR(filtered_IR_L)**2
        # # smoothed_IR_L = RIRP_instance.get_smooth_by_median_filter(filtered_IR_L,1000)
        # crosspoint_L = RIRP_instance.get_lundeby_limit()
        # smoothed_IR_L = RIRP_instance.get_smooth_by_schroeder(filtered_IR_L, crosspoint_L)
        # parameters_L = RIRP_instance.get_acoustical_parameters(smoothed_IR_L)
        
        # # RIGTH
        # self.IR = self.IR[:, 1] (???)
        # IR_R_reversed = RIRP_instance.get_reversed_IR(IR_R)
        # filtered_IR_R, center_freqs =  RIRP_instance.get_IR_filtered(IR = IR_reversed_R, fs = fs, bands_per_oct = 1)
        # filtered_IR_R = RIRP_instance.get_reversed_IR(filtered_IR_R)**2
        # # smoothed_IR_R = RIRP_instance.get_smooth_by_median_filter(filtered_IR_R,1000)
        # crosspoint_R = RIRP_instance.get_lundeby_limit()
        # smoothed_IR_R = RIRP_instance.get_smooth_by_schroeder(filtered_IR_R, crosspoint_R)
        # parameters_R = RIRP_instance.get_acoustical_parameters(smoothed_IR_R)
        
        # print('hola')
    
