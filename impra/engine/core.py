from typing import Dict, Union
from pathlib import Path

import numpy as np

from impra import utils
from impra.process.smoothing import Smoothers
from impra.process.filtering import Filters
from impra.process.parameters import ParametersCalculationWrapper

# UI output
cfg = {
    "sample_rate": 48000,
    "filtering": {
        "type": Filters.OCTAVE_BAND,
        "flip_ir": True,
    },
    "smoothing": {
        "type": Smoothers.MOVING_MEDIAN_AVERAGE,
        "args": {"window_length": 50},
    },
}


class ImpulseResponseAnalyzer:
    def __init__(self, cfg: Dict) -> None:
        self.filtering_cfg = cfg["filtering"]
        self.smoothing_cfg = cfg["smoothing"]
        self.sample_rate = cfg["sample_rate"]

    def analyze(self, audio_path: Union[Path, str]):
        signal_array, sample_rate = utils.load_signal(audio_path, self.sample_rate)

        if self.filtering_cfg["flip_ir"]:
            signal_array = utils.flip_signal(signal_array)

        # Filtering
        filter = self.filtering_cfg["type"].value()
        signal_filtered = filter.filter(signal_array)

        if self.filtering_cfg["flip_ir"]:
            signal_filtered = utils.flip_signal(signal_filtered)

        # Hilbert envelope
        hilbert_envelope = Smoothers.HILBERT_ENERGY_ENVELOPE.value()
        energy_envelope = hilbert_envelope.smooth(signal_filtered, self.sample_rate)

        # Smoothing
        smoother = self.smoothing_cfg["type"].value(**self.smoothing_cfg["args"])
        energy_smoothed = []
        for band_i in energy_envelope:
            energy_smoothed.append(smoother.smooth(band_i, self.sample_rate))
        energy_smoothed = np.array(energy_smoothed)

        # Linear to dB
        energy_smoothed_db = utils.to_db_normalized(energy_smoothed)

        # Parameters calculation
        parameters_calculator = ParametersCalculationWrapper()
        parameters = parameters_calculator(
            energy_smoothed_db, energy_envelope, sample_rate
        )

        results_df = utils.results_to_dataframe(
            parameters, filter.filter_generator.center_freqs
        )
        return results_df
