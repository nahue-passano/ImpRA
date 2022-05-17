import numpy as np
from scipy import signal
import soundfile as sf
from scipy.signal import chirp, butter, sosfiltfilt
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

class RIRP:
    def __init__(self,ir_path):
        self.ir, self.fs = sf.read(ir_path)
    
    def get_ir_from_sinesweep(self, f_min, f_max, T):
        """
        Parameters
        ----------
        f_min : Frecuencia inicial del sine-sweep [Hz].
        f_max : Frecuencia final de sine-sweep [Hz].
        T : Duración del sine-sweep [s].

        Returns
        -------
        RI : ndarray
            Respuesta al impulso

        """
        # sine_sweep, fs = sf.read(path)                                          # Importa sine-sweep grabado
        sine_sweep, fs = self.ir , self.fs
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
        IR_fft = sine_sweep_fft * inv_filt_fft
        IR = np.fft.ifft(IR_fft)                                                # Repuesta al impulso
        self.ir = IR/np.amax(np.abs(IR))
        # self.ir = IR[np.amax(abs(IR)):(3*fs)]                                        # Se recorta el audio desde el máximo hasta 3 s
        # return self.ir
    
    def get_reversed_ir(self):
        self.ir = np.flip(self.ir)
    
    def get_ir_filtered(self, bands_per_oct, bw_ir = None):
        """Filters the impulse response in octave or third octave bands with 
        6th order (in the case of the octave band) and 8th order (in the case
        of the third octave band) butterworth bandpass filters according to
        NORMA

        Args:
            bands_per_oct (int): Bands per octave
            bw_ir (list) (optional): Bandwidth of the impulse response to be filtered. 
            bw_ir[0]: lower boundary, bw_ir[1]: upper boundary. Default to None. 

        Returns:
            filtered_ir (numpy array): Array of size m*n where each column is the impulse
            response filtered with the m-th bandpass filter
            center_freqs (numpy array): Array with the center frequencies of the filters
        """

        if bands_per_oct == 1:
            filter_order = 6
            center_freqs = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
            
        elif bands_per_oct == 3:
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
        filtered_ir = np.zeros((len(center_freqs), len(self.ir)))
        
        # Generation of the bandpass filters and filtering of the IR
        for lower, upper in zip(lower_boundary_freqs, upper_boundary_freqs):  
            butterworth_filters = butter(N = filter_order, Wn = np.array([lower, upper]), 
                                btype='bandpass', analog=False , 
                                output='sos', fs = self.fs)                  # Generates the bandpass

            index = np.where(lower_boundary_freqs == lower)[0] 

            filtered_ir[index, :] = sosfiltfilt(butterworth_filters, self.ir) # Filters the IR
        
        return filtered_ir, center_freqs

    def get_chu_compensation(signal, percentage = 10):
        signal_len = len(signal)
        noise_start = int(np.round((1 - percentage / 100) * signal_len))
        signal_trimmed = signal[noise_start:]  # Trims the signal keeping only the last x% of itself as specified.
        noise_rms = np.mean(signal_trimmed)  # Calculates the mean squared value
        return noise_rms

    def smooth_by_median_filter(signal, len_window):
        smoothed_signal = np.zeros(len(signal))
        for i in range(len(smoothed_signal)):
            smoothed_signal[i] = median_filter(signal[i],len_window)
        
        return smoothed_signal

if __name__ == '__main__':
    RIRP_instance = RIRP('audio_tests/sweep_1.wav')
    
    f_min = 125
    f_max = 16_000
    T = 3
    RIRP_instance.get_ir_from_sinesweep(f_min = f_min , f_max = f_max , T = T)

    plt.plot(RIRP_instance.ir)
    plt.savefig('test.png')
    