import json
import numpy as np
from scipy import signal
from pathlib import Path

# TODO: Documentar bien todas las funciones
# TODO: Testear que el código sea funcional

class JsonHandler():
    def load_json(filename):
        with open(filename) as json_obj:
            json_dict = json.load(json_obj)
        
        return json_dict
    
    def write_json(filename, dict):
        with open(filename, 'w') as json_obj:
            json.dump(dict, json_obj, indent = 4)
    

class Filtering():

    def _generate_bandpass_filters(self, sr):
        
        bandpass_filters = {}
        
        for band_i in self.center_freqs:
            lower = band_i / self.jump_freq
            upper = band_i * self.jump_freq
            upper[np.where(upper > self.sr/2)] = self.sr/2 - 1
            
            # Generates the bandpass at band_i
            bandpass_filter_i = signal.butter(N = self.filter_order,
                                            Wn = np.array([lower, upper]),
                                            btype='bandpass', analog=False,
                                            output='sos', fs = sr)
            
            bandpass_filters[str(band_i)] = bandpass_filter_i.tolist()
            
        JsonHandler.write_json(self.filename, bandpass_filters)
    
        
    def _set_center_freqs_by_bandwidth(self,bw_ir):
        self.center_freqs = self.center_freqs[bw_ir[0]:bw_ir[1] + 1]
        
    
    def filter(self, ir, sr, bw_ir=None):

        if bw_ir != None:
            self._set_center_freqs_by_bandwidth

        if self.filename.is_file():
            butterworth_filters = JsonHandler.load_json(self.filename)
        else:
            butterworth_filters = self._generate_bandpass_filters(sr)
        
        filtered_ir = np.zeros((len(self.center_freqs), len(ir)))
        
        for i, filter_i in enumerate(butterworth_filters.values()):
            filter_i = np.array(filter_i)
            filtered_ir[i, :] = signal.sosfiltfilt(filter_i, ir)
            
        return filtered_ir

    
class OctaveBandFilter(Filtering):
    def __init__(self):
        self.filename = Path('filters/octave_band_butterworths.json')
        self.band_per_oct = 1
        self.filter_order = 6
        self.jump_freq = np.power(2, 1/(2*self.bands_per_oct))
        self.center_freqs = np.array([31.5, 63, 125, 250, 500, 1000,
                                      2000, 4000, 8000, 16000])

class ThirdOctaveBandFilter(Filtering):
    def __init__(self):
        self.filename = Path('filters/third_octave_band_butterworths.json')
        self.bands_per_oct = 3
        self.filter_order = 8
        self.jump_freq = np.power(2, 1/(2*self.bands_per_oct))
        self.center_freqs = np.array([31.5, 40, 50, 63, 80, 100, 125,
                                      160, 200, 250, 315, 400, 500,
                                      630, 800, 1000, 1250, 1600, 2000,
                                      2500, 3150, 4000, 5000, 6300,
                                      8000, 10000, 12500, 16000])