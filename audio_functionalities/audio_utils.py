import numpy as np
import soundfile as sf
import scipy.signal as signal

# TODO: Investigar si se puede reemplazar _fft_convolution por signal.fftconvolve
# TODO: Testear que funcionen en el if __name__ == "__main__"

class Utils():
    """ Audio utilities to be used among the project
    """
    
    def load_signal(self, audio_path):
        """ Loads the signal (ir or sinesweep) to be procesed

        Parameters
        ----------
        audio_path : str
            Path of the audio file

        Returns
        -------
        audio_array : numpy.ndarray
            Array of the audio file
        sr : float
            Sample rate of the audio
        """
        
        audio_array, sr = sf.read(audio_path)
        
        return audio_array, sr
    
    def _chirp_inverse_filter(self, time_len, sr, f_min, f_max):
        """_summary_

        Parameters
        ----------
        time_len : _type_
            _description_
        sr : _type_
            _description_
        f_min : _type_
            _description_
        f_max : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Time vector to evaluate the chirp
        t = np.arange(0, time_len, 1/sr)
        
        # Logarithmic chirp generation
        log_chirp = signal.chirp(t = t, 
                                 f0 = f_min,
                                 f1 = f_max,
                                 t1 = time_len,
                                 method = 'logarithmic')           
        
        inv_log_chirp = log_chirp[::-1]                         # Inverse logarithmic chirp
        
        # Applying modulation to generate inverse filter
        m = 1/(2*np.pi*np.exp(t*np.log(f_max/f_min)/time_len))  # Modulation factor
        inverse_filter = m * inv_log_chirp             
        inverse_filter /= np.max(abs(inverse_filter))           # Normalization
        
        return inverse_filter
    
    def _fft_convolution(self, array_1, array_2):
        """_summary_

        Parameters
        ----------
        array_1 : _type_
            _description_
        array_2 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        
        # Applying the FFT to the arrays
        arr_1_fft = np.fft.rfft(array_1)
        arr_2_fft = np.fft.rfft(array_2)
        
        # Frequency multiplication = Time convolution
        output_freq = arr_1_fft * arr_2_fft
        output_time = np.fft.ifft(output_freq)  # Inverse fft                                                
        output_time /= np.max(abs(output_time)) # Normalization
        
        return output_time
    
    def get_ir_from_sinesweep(self, sinesweep_measured, sr, f_min, f_max):
        """ Generates an ir from a sinesweep measurement.

        Parameters
        ----------
        sinesweep_measured : numpy.ndarray
            Array of the sinesweep measured
        sr : float
            Sample rate of the sinesweep measured
        f_min : float
            Minimum frequency of the sinesweep
        f_max : float
            Maximum frequency of the sinesweep

        Returns
        -------
        _type_
            _description_
        """

        # Time lenght of the sinesweep measured
        time_len = len(sinesweep_measured)/sr
        
        # Generating chirp inverse filter
        inverse_filter = self._chirp_inverse_filter(time_len = time_len,
                                                    sr = sr,
                                                    f_min = f_min,
                                                    f_max = f_max)
        
        # Getting ir from convolving the sinesweep with the chirp inverse filter
        ir_from_sinesweep = self._fft_convolution(sinesweep_measured, inverse_filter)
        
        return ir_from_sinesweep
    
    def reverse_array(self, array):
        """_summary_

        poner esto en la linea donde se llama a esta función:
        Temporarily flips the ir to avoid low-frequency overlap between
        the filter's impulse responses and the ir

        Parameters
        ----------
        array : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        if np.ndim(array) == 1:    # Case of 1-D array to be reversed
            reversed_array = np.flip(array)
        else:                       # Case of N-D array to be reversed
            reversed_array = np.flip(array,axis = 1)

        return reversed_array
    