import numpy as np
from scipy import signal
import soundfile as sf
from scipy.signal import chirp, butter, sosfiltfilt
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

class RIRP:
    def load_signal(self, signal_path):
        signal, fs = sf.read(signal_path)
        return signal, fs
    
    def get_ir_from_sinesweep(self, sine_sweep, fs, f_min, f_max, T):
        """
        Parameters
        ----------
        sine_sweep : Signal of the sine sweep
        fs : Sample rate
        f_min : Frecuencia inicial del sine-sweep [Hz].
        f_max : Frecuencia final de sine-sweep [Hz].
        T : Duración del sine-sweep [s].

        Returns
        -------
        RI : ndarray
            Respuesta al impulso

        """
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
        ir_fft = sine_sweep_fft * inv_filt_fft
        ir = np.fft.ifft(ir_fft)                                                # Repuesta al impulso
        ir = ir/np.amax(np.abs(ir))

        return ir

    def get_reversed_ir(self, ir):
        if np.ndim(ir) == 1:    # Case of single ir to be reversed
            reversed_ir = np.flip(ir)
        else:                   # Case of matrix of ir filtered to be reversed
            reversed_ir = np.flip(ir,axis = 1)

        return reversed_ir
    
    def get_ir_filtered(self, ir, fs, bands_per_oct, bw_ir = None):
        """Filters the impulse response in octave or third octave bands with 
        6th order (in the case of the octave band) and 8th order (in the case
        of the third octave band) butterworth bandpass filters according to
        NORMA

        Args:
            ir (numpy array): Array of the impulse response
            fs (int): Sample rate
            bands_per_oct (int): Bands per octave
            bw_ir (list) (optional): Bandwidth of the impulse response to be filtered. 
            bw_ir[0]: lower boundary, bw_ir[1]: upper boundary. Default to None. 

        Returns:
            filtered_ir (numpy array): Array of size m*n where each row is the impulse
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
        upper_boundary_freqs[np.where(upper_boundary_freqs > fs/2)] = fs/2 - 1
        filtered_ir = np.zeros((len(center_freqs), len(ir)))
        
        # Generation of the bandpass filters and filtering of the IR
        for lower, upper in zip(lower_boundary_freqs, upper_boundary_freqs):  
            butterworth_filters = butter(N = filter_order, Wn = np.array([lower, upper]), 
                                btype='bandpass', analog=False , 
                                output='sos', fs = fs)                  # Generates the bandpass

            index = np.where(lower_boundary_freqs == lower)[0] 

            filtered_ir[index, :] = sosfiltfilt(butterworth_filters, ir) # Filters the IR
        
        return filtered_ir, center_freqs

    def get_chu_compensation(self, signal, percentage = 10):
        signal_len = len(signal)
        noise_start = int(np.round((1 - percentage / 100) * signal_len))
        signal_trimmed = signal[noise_start:]  # Trims the signal keeping only the last x% of itself as specified.
        noise_rms = np.mean(signal_trimmed)  # Calculates the mean squared value
        
        return noise_rms

    def get_smooth_by_schroeder(self):
        pass

    def get_smooth_by_median_filter(self, signal, len_window):
        smoothed_signal = np.zeros_like(signal)
        for freq in range(np.shape(signal)[0]):
            smoothed_signal[freq,:] = median_filter(signal[freq,:],len_window)
            
        return smoothed_signal


if __name__ == '__main__':
    RIRP_instance = RIRP()
    
    f_min = 125
    f_max = 16_000
    T = 3
    sinesweep, fs = RIRP_instance.load_signal('audio_tests/sweep_1.wav')
    ir = RIRP_instance.get_ir_from_sinesweep(sine_sweep = sinesweep, fs = fs ,f_min = f_min , f_max = f_max , T = T)
    ir_reversed = RIRP_instance.get_reversed_ir(ir)
    filtered_ir, center_freqs =  RIRP_instance.get_ir_filtered(ir = ir_reversed, fs = fs, bands_per_oct = 1)
    filtered_ir = RIRP_instance.get_reversed_ir(filtered_ir)**2
    smoothed_ir = RIRP_instance.get_smooth_by_median_filter(filtered_ir,1000)
    print('hola')
    