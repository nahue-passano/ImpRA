from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from impra import utils


class SmootherInterface(ABC):
    @abstractmethod
    def smooth(self, signal_array: np.ndarray, sample_rate: int) -> np.ndarray:
        pass


class MovingMedianAverageSmoother(SmootherInterface):
    def __init__(self, window_length: int) -> None:
        self.window_length = window_length

    def smooth(self, signal_array: np.ndarray, sample_rate: int) -> np.ndarray:
        return median_filter(signal_array, self.window_length)


class SchroederInverseIntegralSmoother(SmootherInterface):
    def smooth(self, signal_array: np.ndarray, sample_rate: int) -> np.ndarray:
        lundeby_limit = self.integral_limit_by_lundeby(signal_array, sample_rate)
        signal_cropped = signal_array[:lundeby_limit]
        cumulative_sum = np.cumsum(signal_cropped[::-1])[::-1]
        smoothed_signal = np.concatenate(
            (cumulative_sum, np.zeros((len(signal_array) - lundeby_limit)))
        )

        return smoothed_signal

    @staticmethod
    def integral_limit_by_lundeby(
        signal: np.ndarray, sample_rate: int
    ) -> Tuple[int, ...]:
        """
        Finds the upper limit of integration for the Schroeder inverse integral
        using Lundeby's method.

        Parameters
        ----------
        signal : np.ndarray
            2D array containing the RIR signal for each frequency band in each row.
        sample_rate : int
            Sampling frequency.

        Returns
        -------
        Tuple[int, ...]
            Indices of the upper limit of the Schroeder integral for each row.
        """
        signal_length = len(signal)
        time_step = int(sample_rate * 0.01)
        num_sections = int(signal_length / time_step)

        energy = signal**2
        avg_energy = [
            np.mean(energy[i * time_step : (i + 1) * time_step])
            for i in range(num_sections)
        ]
        time_axis = np.ceil(time_step / 2) + np.arange(num_sections) * time_step

        rms_dB = utils.calculate_average_energy(energy, 0)
        avg_energy_dB = 10 * np.log10(avg_energy / np.max(energy))

        # Linear regression calculation
        try:
            reg_end_index = int(max(np.argwhere(avg_energy_dB > rms_dB + 10)))
            if np.any(avg_energy_dB[:reg_end_index] < rms_dB + 10):
                reg_end_index = min(
                    np.where(avg_energy_dB[:reg_end_index] < rms_dB + 10)[0]
                )
            if reg_end_index == 0 or reg_end_index < 10:
                reg_end_index = 10
        except:
            reg_end_index = 10

        regression_matrix = np.vstack(
            [time_axis[:reg_end_index], np.ones(reg_end_index)]
        ).T
        slope, intercept = np.linalg.lstsq(
            regression_matrix, avg_energy_dB[:reg_end_index], rcond=None
        )[0]
        intersection_index = int((rms_dB - intercept) / slope)

        # Check if the intersection index exceeds the length of the signal
        if intersection_index > signal_length or intersection_index < 0:
            intersection_index = signal_length

        return intersection_index


class HilbertEnvelopeSmoother(SmootherInterface):
    def smooth(self, signal_array: np.ndarray, sample_rate: int) -> np.ndarray:
        hilbert_transform = signal.hilbert(signal_array, axis=1)
        energy = np.abs(hilbert_transform) ** 2
        max_energy = np.max(energy, axis=1, keepdims=True)
        return energy / max_energy


class Smoothers(Enum):
    MOVING_MEDIAN_AVERAGE = MovingMedianAverageSmoother
    SCHROEDER_INVERSE_INTEGRAL = SchroederInverseIntegralSmoother
    HILBERT_ENERGY_ENVELOPE = HilbertEnvelopeSmoother
