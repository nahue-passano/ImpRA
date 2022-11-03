import numpy as np
import os
from scipy import signal
import json
import soundfile as sf
from scipy.signal import chirp, butter, sosfiltfilt
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

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
        
        return self.IR, self.fs
    
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
        
        return self.IR, self.fs

    def get_reversed_IR(self, IR):
        """ Temporarily flips the ir to avoid low-frequency overlap between
        the filter's impulse responses and the ir

        Returns:
            reversed_IR (numpy.array): Temporarily fliped IR
        """

        if np.ndim(IR) == 1:    # Case of single ir to be reversed
            reversed_IR = np.flip(IR)
        else:                   # Case of matrix of ir filtered to be reversed
            reversed_IR = np.flip(IR,axis = 1)

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
            print('Generating filters')
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
        N = len(IR)
        energia = IR.copy()
        media = np.zeros(int(N/(self.fs*0.01)))
        eje_tiempo = np.zeros(int(N/(self.fs*0.01)))

        # Divide in sections and calculate the mean

        t = np.floor(N/(self.fs*0.01)).astype('int')
        v = np.floor(N/t).astype('int')

        RMS = lambda signal: np.sqrt(np.mean(signal**2))
        dB = lambda signal: 10 * np.log10(abs(signal/max(energia)))

        for i in range(0, t):
            # media[i] = np.mean(energia[i * v:(i + 1) * v])
            media[i]=RMS(energia[i * v:(i + 1) * v])
            eje_tiempo[i] = np.ceil(v/2).astype('int') + (i*v)
            
        # Calculate noise level of the last 10% of the signal

        # rms_dB = 10 * np.log10(np.sum(energia[round(0.9 * N):]) / (0.1 * N) / max(energia))
        rms_dB = dB(RMS(energia[round(0.9 * N):]))
        mediadB = 10 * np.log10(media / max(energia))

        # Se busca la regresión lineal del intervalo de 0dB y la media mas proxima al ruido + 10dB.
        # Calculate linear regression between the 0 dB 

        try:
            r = int(max(np.argwhere(mediadB > rms_dB + 10)))
                
            if np.any(mediadB[0:r] < rms_dB+10):
                r = min(min(np.where(mediadB[0:r] < rms_dB + 10)))
            if np.all(r==0) or r<10:
                r=10
        except:
            r = 10

        # Least squares
            
        A = np.vstack([eje_tiempo[0:r], np.ones(len(eje_tiempo[0:r]))]).T
        m, c = np.linalg.lstsq(A, mediadB[0:r], rcond=-1)[0]
        crosspoint = int((rms_dB-c)/m)


        linear_fit = lambda x: (m * x) + c

        x = np.arange(len(mediadB))
        
        # plt.plot(eje_tiempo, mediadB)
        # plt.plot(eje_tiempo, linear_fit(x))
        # plt.show()


        # Insufficient SNR

        if rms_dB > -20:
            
            crosspoint = len(energia)
            C = None
            
        else:

            error = 1
            INTMAX = 5
            veces = 1
                    
            while error > 0.0004 and veces <= INTMAX:
                
                # Calculates new time intervals for the mean with approximately
                # p steps for each 10 dB
                
                p = 10
                
                # Number of samples for the decay slope of 10 dB
                
                delta = int(abs(10/m)) 
                
                # Interval over which the mean is calculated
                
                v = np.floor(delta/p).astype('int') 
                t = int(np.floor(len(energia[:int(crosspoint-delta)])/v))
                
                if t < 2:
                    t = 2
                elif np.all(t == 0):
                    t = 2

                media = np.zeros(t)
                eje_tiempo = np.zeros(t)
                
                for i in range(0, t):
                    media[i] = RMS(energia[i*v:(i + 1) * v])
                    # media[i] = np.mean(energia[i*v:(i + 1) * v])
                    eje_tiempo[i] = np.ceil(v / 2) + (i * v).astype('int')
                    
                mediadB = 10 * np.log10(media / max(energia))
                A = np.vstack([eje_tiempo, np.ones(len(eje_tiempo))]).T
                m, c = np.linalg.lstsq(A, mediadB, rcond=-1)[0]

                # Nueva media de energia de ruido, comenzando desde desde el crosspoint de la linea de 
                # decaimiento, 10 dB por debajo del crosspoint de crosspoint
                
                # New noise average level, starting from the point of the 
                # decay curve, 10 dB below the intersection.
                
                noise = energia[int(abs(crosspoint + delta)):]
                
                if len(noise) < round(0.1 * len(energia)):
                    noise = energia[round(0.9 * len(energia)):]
                    
                # rms_dB = 10 * np.log10(sum(noise)/ len(noise) / max(energia))
                rms_dB = dB(RMS(noise))

                # New intersection index
                
                error = abs(crosspoint - (rms_dB - c) / m) / crosspoint
                crosspoint = int(round((rms_dB - c) / m))
                
                x = np.arange(len(mediadB))

                # plt.plot(eje_tiempo, mediadB)
                # plt.plot(eje_tiempo, linear_fit(x))
                # plt.show()
                veces += 1
                        
        # Output validation
                
        if crosspoint > N:
            crosspoint = N
                    
        C = max(energia) * 10 ** (c / 10) * np.exp(m/10/np.log10(np.exp(1))*crosspoint) / (
            -m / 10 / np.log10(np.exp(1)))
        
        return crosspoint, C

    def get_smooth_by_schroeder(self, IR, crosspoint):
        """ Smooths the ir's energy with Schroeder's method.

        Args:
            IR_matrix (numpy.array): Array of the IR
            crosspoint (float): Time crosspoint to end Schroeder's integral

        Returns:
            smoothed_energy (numpy.array): Array of the smoothed energy
        """
        
        # Get ir energy
        energy = IR**2

        ## Iterar schroeder en cada fila de la matriz
        
        schroeder = np.pad(np.flip(np.cumsum(np.flip(energy[0:crosspoint:]))), (0, (len(energy) - crosspoint)))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed_energy = 10 * np.log10(schroeder / np.max(schroeder))
            
        return smoothed_energy

    def get_smooth_by_median_filter(self, IR, len_window):
        """ Smooths the IR's energy with Median Filter

        Args:
            IR (numpy.array): Array of the IR
            len_window (int): Length of the window in samples

        Returns:
            smoothed_energy (numpy.array): Array of the smoothed energy 
        """
        ## Elevar al cuadrado IR_matix
        
        smoothed_energy = median_filter(IR, len_window)
    
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
        
        acoustical_parameters = {}
        
        acoustical_parameters['Tt'] = self.get_Tt_EDTt(smoothed_energy)[0]
        
        acoustical_parameters['EDTt'] = self.get_Tt_EDTt(smoothed_energy)[1]

        acoustical_parameters['EDT'] = self.get_EDT(smoothed_energy)

        acoustical_parameters['T20'] = self.get_T20(smoothed_energy)
        
        acoustical_parameters['T30'] = self.get_T30(smoothed_energy)
        
        acoustical_parameters['C50'] = self.get_C50(smoothed_energy)
        
        acoustical_parameters['C80'] = self.get_C80(smoothed_energy)
        
        # IACC_Early = self.get_IACC_Early(IR_L, IR_R) CHEQUEAR, NO SE SI PARA EL IACC SE ENRA CON R Y L SEPARADAS O KE
        
        
        return acoustical_parameters
    
    def get_Tt_EDTt(self, smoothed_energy):
        """ Calculates the Transition Time (Tt) of the IR

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            Tt (float): Transition time in seconds
        """
        # No estoy segura si es necesario elevar al 2 la smoothed_energy o no.
        Tt = np.max(np.where(np.cumsum(smoothed_energy**2) <= 0.99 * np.max(np.sum(smoothed_energy**2)))) / self.fs
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     smoothed_energy_dB = 10 * np.log10(np.abs(smoothed_energy / np.max(smoothed_energy)))
        x_min_EDTt = np.max(np.argwhere(smoothed_energy > -1))
        x_max_EDTt = Tt.copy()
        EDTt = (60/(-1 - smoothed_energy[int(x_max_EDTt)]))*(x_max_EDTt - x_min_EDTt) / self.fs
        
        # m_EDTt, b_EDTt = np.polyfit((x_min_EDTt, x_max_EDTt), (smoothed_energy_dB[x_min_EDTt], smoothed_energy_dB[x_max_EDTt]), 1)
        # EDTt_linear_fit = lambda x: (m_EDTt * x) + b_EDTt
        
        return np.round(Tt,4), np.round(EDTt,4)
        
    def get_EDT(self, smoothed_energy):
        """ Calculates the Energy Decay Time (EDT) of the IR

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            EDT (float): Energy Decay Time in seconds
        """
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_EDT = np.max(np.argwhere(smoothed_energy > -1))
        x_max_EDT = np.max(np.argwhere(smoothed_energy > -10))
        EDT = (60/9) * (x_max_EDT - x_min_EDT) / self.fs

        # m_EDT, b_EDT = np.polyfit((x_min_EDT, x_max_EDT), (smoothed_energy_dB[x_min_EDT], smoothed_energy_dB[x_max_EDT]), 1)
        # EDT_linear_fit = lambda x: (m_EDT * x) + b_EDT
        return np.round(EDT,4)

    def get_T20(self, smoothed_energy):
        """ Estimates the Reverberation Time (RT) using the -5 to -25 dB 
        range of the smoothed energy

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            T20 (float): T20 in seconds
        """
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_T20 = np.max(np.argwhere(smoothed_energy > -5))
        x_max_T20 = np.max(np.argwhere(smoothed_energy > -25))
        T20 = 3 * (x_max_T20 - x_min_T20) / self.fs
        
        # m_T20, b_T20 = np.polyfit((x_min_T20, x_max_T20), (smoothed_energy_dB[x_min_T20], smoothed_energy_dB[x_max_T20]), 1)
        # T20_linear_fit = lambda x: (m_T20 * x) + b_T20

        return np.round(T20,4)

    def get_T30(self, smoothed_energy):
        """ Estimates the Reverberation Time (RT) using the -5 to -35 dB 
        range of the smoothed energy

        Args:
            smoothed_energy (numpy.array): Smoothed energy of the IR

        Returns:
            T30 (float): T30 in seconds
        """
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     smoothed_energy_dB = 10 * np.log10(abs(smoothed_energy / max(smoothed_energy)))
        x_min_T30 = np.max(np.argwhere(smoothed_energy > -5))
        x_max_T30 = np.max(np.argwhere(smoothed_energy > -35))
        T30 = 2 * (x_max_T30 - x_min_T30) / self.fs
        
        # m_T30, b_T30 = np.polyfit((x_min_T30, x_max_T30), (smoothed_energy_dB[x_min_TR], smoothed_energy_dB[x_max_T30]), 1)
        # T30_linear_fit = lambda x: (m_T30 * x) + b_T30
        
        return np.round(T30,4)
    
    def get_C50(self, smoothed_energy):
        t50 = np.int64(0.05*self.fs)
        y50_num = smoothed_energy[:t50]
        y50_den = smoothed_energy[t50:]
        with np.errstate(divide='ignore', invalid='ignore'):
            C50 = 10*np.log10((np.sum(y50_num))/(np.sum(y50_den)))
        return np.round(C50,4)
    
    def get_C80(self, smoothed_energy):
        t80 = np.int64(0.08*self.fs)
        y80_num = smoothed_energy[:t80]
        y80_den = smoothed_energy[t80:]
        with np.errstate(divide='ignore', invalid='ignore'):
            C80 = 10*np.log10((np.sum(y80_num))/(np.sum(y80_den)))
        return np.round(C80,4)
    
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
    
    A = RIRP_instance.get_reversed_IR(np.array([[1,2,3],[4,5,6]]))
    print(A)
    # signal_path = 'audio_tests/rirs/RI_1.wav'
    # IR, fs = RIRP_instance.load_signal(signal_path)
    # IR_reversed = RIRP_instance.get_reversed_IR(IR)
    # filtered_IR, center_freqs =  RIRP_instance.get_ir_filtered(IR = IR_reversed, bands_per_oct = 1)
    # filtered_IR = RIRP_instance.get_reversed_IR(filtered_IR)**2
    # # smoothed_IR = RIRP_instance.get_smooth_by_median_filter(filtered_IR,1000)
    # for ir_i in filtered_IR:
    #     crosspoint = RIRP_instance.get_lundeby_limit(ir_i)
    #     smoothed_IR = RIRP_instance.get_smooth_by_schroeder(ir_i, crosspoint)
    #     parameters = RIRP_instance.get_acoustical_parameters(smoothed_IR)
    #     print(parameters)
        
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
    
