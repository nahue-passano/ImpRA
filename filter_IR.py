import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filter_IR(ir, fs, bands_per_oct, bw_ir = None):
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
    filtered_ir = np.zeros((len(center_freqs), len(ir)))
    
    # Generation of the bandpass filters and filtering of the IR
    for lower, upper in zip(lower_boundary_freqs, upper_boundary_freqs):  
        butterworth_filters = signal.butter(N = filter_order, Wn = np.array([lower, upper]), 
                            btype='bandpass', analog=False , 
                            output='sos', fs = fs)                  # Generates the bandpass

        index = np.where(lower_boundary_freqs == lower)[0] 

        filtered_ir[index, :] = signal.sosfiltfilt(butterworth_filters, ir) # Filters the IR
    return filtered_ir, center_freqs

if __name__ == '__main__':
    ir = np.zeros(44100)
    fs = 44100*2
    bands_per_oct = 3
    filter_IR(ir, fs, bands_per_oct, plot_filters = True)
