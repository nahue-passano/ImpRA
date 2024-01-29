from enum import Enum
import json
import numpy as np
from scipy import signal
from pathlib import Path

# Global variables for filter paths
OCTAVE_BAND_FILTER_PATH = "impra/process/filters_weights/octave_band_butterworths.json"
THIRD_OCTAVE_BAND_FILTER_PATH = (
    "impra/process/filters_weights/third_octave_band_butterworths.json"
)
SAMPLE_RATE = 48_000


class JsonHandler:
    @staticmethod
    def load_json(filename):
        with open(filename) as json_obj:
            return json.load(json_obj)

    @staticmethod
    def write_json(filename, data):
        with open(filename, "w") as json_obj:
            json.dump(data, json_obj, indent=4)


class FilterGenerator:
    def __init__(self, center_freqs, filter_order, jump_freq, sample_rate):
        self.center_freqs = center_freqs
        self.filter_order = filter_order
        self.jump_freq = jump_freq
        self.sample_rate = sample_rate

    def generate_bandpass_filters(self):
        bandpass_filters = {}
        for band_i in self.center_freqs:
            lower = band_i / self.jump_freq
            upper = band_i * self.jump_freq
            upper = np.minimum(upper, self.sample_rate / 2 - 1)
            bandpass_filter_i = signal.butter(
                N=self.filter_order,
                Wn=np.array([lower, upper]),
                btype="bandpass",
                analog=False,
                output="sos",
                fs=self.sample_rate,
            )
            bandpass_filters[str(band_i)] = bandpass_filter_i.tolist()
        return bandpass_filters


class Filtering:
    def __init__(self, filename, filter_generator):
        self.filename = Path(filename)
        self.filter_generator = filter_generator

    def filter(self, signal_array, bw_ir=None):
        if bw_ir is not None:
            self.filter_generator.center_freqs = self.filter_generator.center_freqs[
                bw_ir[0] : bw_ir[1] + 1
            ]

        if self.filename.is_file():
            butterworth_filters = JsonHandler.load_json(self.filename)
        else:
            butterworth_filters = self.filter_generator.generate_bandpass_filters()
            JsonHandler.write_json(self.filename, butterworth_filters)

        filtered_signal_array = np.zeros(
            (len(self.filter_generator.center_freqs), len(signal_array))
        )

        for i, filter_i in enumerate(butterworth_filters.values()):
            filter_i = np.array(filter_i)
            filtered_signal_array[i, :] = signal.sosfiltfilt(filter_i, signal_array)

        return filtered_signal_array


class OctaveBandFilter(Filtering):
    def __init__(self):
        center_freqs = np.array(
            [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        )
        filter_generator = FilterGenerator(
            center_freqs, 6, np.power(2, 1 / 2), SAMPLE_RATE
        )
        super().__init__(OCTAVE_BAND_FILTER_PATH, filter_generator)


class ThirdOctaveBandFilter(Filtering):
    def __init__(self):
        # fmt: off
        center_freqs = np.array(
            [
                31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                4000, 5000, 6300, 8000, 10000, 12500, 16000,
            ]
        )
        # fmt: on
        filter_generator = FilterGenerator(
            center_freqs, 8, np.power(2, 1 / 6), SAMPLE_RATE
        )
        super().__init__(THIRD_OCTAVE_BAND_FILTER_PATH, filter_generator)


class Filters(Enum):
    OCTAVE_BAND = OctaveBandFilter
    THIRD_OCTAVE_BAND = ThirdOctaveBandFilter
