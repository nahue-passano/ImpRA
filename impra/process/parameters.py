from typing import Dict
import numpy as np


class ParametersCalculationWrapper:
    def __call__(
        self,
        energy_smoothed_db: np.ndarray,
        energy_envelope: np.ndarray,
        sample_rate: int,
        integration_limits: np.ndarray = None,
    ) -> Dict:
        results = {}

        reverberation_params = self.calculate_reverberation_parameters(
            energy_smoothed_db, sample_rate
        )

        clarity_params = self.calculate_clarity_parameters(
            energy_envelope, sample_rate, integration_limits
        )

        transition_params = self.calculate_transition_parameters(
            energy_envelope, energy_smoothed_db, sample_rate
        )

        results = {**reverberation_params, **clarity_params, **transition_params}
        return results

    def calculate_reverberation_parameters(self, energy_smoothed_db, sample_rate):
        """
        Calculate reverberation time parameters for each frequency band.

        Parameters
        ----------
        energy_smoothed_db : ndarray
            Array containing the filtered impulse responses, one per frequency band.
        sample_rate : int
            Sampling frequency.

        Returns
        -------
        tuple of ndarray
            Early Decay Time (EDT), T20, and T30 for each frequency band.
        """
        time_array = np.arange(len(energy_smoothed_db[0])) / sample_rate
        edt_times = [
            self._calculate_rt_i(ir, time_array, decay_start=1, decay_end=10)
            for ir in energy_smoothed_db
        ]
        t20_times = [
            self._calculate_rt_i(ir, time_array, decay_start=5, decay_end=25)
            for ir in energy_smoothed_db
        ]
        t30_times = [
            self._calculate_rt_i(ir, time_array, decay_start=5, decay_end=35)
            for ir in energy_smoothed_db
        ]
        results = {
            "EDT": np.round(edt_times, 2),
            "T20": np.round(t20_times, 2),
            "T30": np.round(t30_times, 2),
        }
        return results

    @staticmethod
    def _calculate_rt_i(energy_smoothed_db, time_array, decay_start, decay_end):
        """
        Calculate reverberation time using linear regression.

        Parameters
        ----------
        energy_smoothed_db : ndarray
            The impulse response signal of a particular frequency band.
        time_array : ndarray
            Array of time values corresponding to the impulse response.
        decay_start : float
            Initial decay in dB for reverberation time calculation.
        decay_end : float
            Final decay in dB for reverberation time calculation.

        Returns
        -------
        float
            Calculated reverberation time.
        """
        truncated_ir = energy_smoothed_db[np.argmax(energy_smoothed_db) :]
        decay_indices = np.where(
            (truncated_ir <= truncated_ir[0] - decay_start)
            & (truncated_ir > truncated_ir[0] - decay_end)
        )
        regression_time = np.vstack(
            [time_array[decay_indices], np.ones(len(time_array[decay_indices]))]
        ).T
        regression_ir = truncated_ir[decay_indices]
        slope, _ = np.linalg.lstsq(regression_time, regression_ir, rcond=None)[0]

        return -60 / slope

    def calculate_clarity_parameters(
        self,
        energy_envelope: np.ndarray,
        sample_rate: int,
        integration_limit: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the clarity parameters C50 and C80 for a given set of RIRs.

        Parameters
        ----------
        energy_envelope : np.ndarray
            2D array with filtered RIR for each frequency band in each row.
        sample_rate : int
            Sampling frequency.
        integration_limit : np.ndarray, optional
            Array containing the limits of integration. If None, defaults to the length of each row.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            C50 and C80 parameters for each frequency band.
        """
        if integration_limit is None:
            integration_limit = np.full(
                energy_envelope.shape[0], energy_envelope.shape[1]
            )

        C50 = np.array(
            [
                self._calculate_clarity_i(ir, sample_rate, 0.05, integration_limit[i])
                for i, ir in enumerate(energy_envelope)
            ]
        )
        C80 = np.array(
            [
                self._calculate_clarity_i(ir, sample_rate, 0.08, integration_limit[i])
                for i, ir in enumerate(energy_envelope)
            ]
        )

        results = {
            "C50": np.round(C50, 2),
            "C80": np.round(C80, 2),
        }

        return results

    @staticmethod
    def _calculate_clarity_i(
        energy_envelope, sample_rate, time_threshold, integration_limit
    ):
        energy_envelope = energy_envelope[np.argmax(energy_envelope) :]
        index_threshold = np.int64(time_threshold * sample_rate)
        numerator = np.sum(energy_envelope[:index_threshold])
        denominator = np.sum(energy_envelope[index_threshold:integration_limit])

        return 10 * np.log10(numerator / denominator)

    def calculate_transition_parameters(
        self,
        energy_envelope: np.ndarray,
        energy_smoothed_db: np.ndarray,
        sample_rate: int,
    ) -> Dict:
        """
        Calculate Transition Time (Tt) and Early Decay Transition Time (EDTt) from filtered impulse responses.

        Parameters
        ----------
        energy_envelope : np.ndarray
            2D array containing the energy envelope of the impulse response for each frequency band.
        energy_smoothed_db : np.ndarray
            2D array containing the smoothed energy in decibels for each frequency band.
        sample_rate : int
            Sampling frequency.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transition time (Tt) and Early Decay Transition Time (EDTt) for each frequency band.
        """
        transition_times = []
        EDTs = []

        for i, impulse_response in enumerate(energy_envelope):
            impulse_response = impulse_response[np.argmax(impulse_response) :]
            impulse_response_trimmed = impulse_response[int(5e-3 * sample_rate) :]
            transition_time = self._calculate_transition_time(
                impulse_response_trimmed, sample_rate
            )
            transition_times.append(transition_time)

            EDT = self._calculate_EDTt(
                energy_smoothed_db[i], transition_time, sample_rate
            )
            EDTs.append(EDT)

        results = {
            "Tt": np.round(transition_times, 2),
            "EDTt": np.round(EDTs, 2),
        }

        return results

    @staticmethod
    def _calculate_transition_time(
        impulse_response: np.ndarray, sample_rate: int
    ) -> float:
        """
        Calculate the transition time (Tt) of an impulse response.

        Parameters
        ----------
        impulse_response : np.ndarray
            The impulse response signal.
        sample_rate : int
            Sampling frequency.

        Returns
        -------
        float
            Transition time.
        """
        try:
            energy_content = np.cumsum(impulse_response**2)
            total_energy = np.sum(impulse_response**2)
            index = np.where(energy_content <= 0.99 * total_energy)[0][-1]
            transition_time = index / sample_rate
        except:
            transition_time = len(impulse_response)
        return transition_time

    @staticmethod
    def _calculate_EDTt(
        energy_envelope: np.ndarray, transition_time: float, sample_rate: int
    ) -> float:
        """
        Calculate the Early Decay Transition Time (EDTt) from a smoothed impulse response.

        Parameters
        ----------
        energy_envelope : np.ndarray
            The smoothed impulse response signal.
        transition_time : int
            Index up to which the signal is considered.
        sample_rate : int
            Sampling frequency.

        Returns
        -------
        float
            Early decay transition time.
        """
        index = int(transition_time * sample_rate)
        time_axis = np.arange(0, transition_time, 1 / sample_rate)
        if len(time_axis) > index:
            time_axis = time_axis[:index]

        regression_matrix = np.vstack([time_axis, np.ones(len(time_axis))]).T
        slope, _ = np.linalg.lstsq(
            regression_matrix, energy_envelope[:index], rcond=-1
        )[0]
        return -60 / slope
