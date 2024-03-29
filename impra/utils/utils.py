from typing import Tuple, Dict, List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def load_signal(audio_path: str, target_sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Load and resample an audio signal from a file to a specified sample rate.

    Parameters
    ----------
    audio_path : str
        The file path to the audio file.
    target_sample_rate : int
        The target sample rate to which the audio file will be resampled.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple containing the loaded and resampled audio signal as a numpy array (`signal_array`)
        and the target sample rate (`target_sample_rate`).
    """
    # Load the audio file in its original sample rate
    signal_array, original_sample_rate = sf.read(audio_path)

    # Resample the audio if the original sample rate is different from the target sample rate
    if original_sample_rate != target_sample_rate:
        if signal_array.ndim == 1:  # Mono
            signal_array = librosa.resample(
                signal_array, orig_sr=original_sample_rate, target_sr=target_sample_rate
            )
        else:  # Stereo
            # Resample each channel separately and stack them back together
            signal_array = np.vstack(
                [
                    librosa.resample(
                        signal_array[:, ch],
                        orig_sr=original_sample_rate,
                        target_sr=target_sample_rate,
                    )
                    for ch in range(signal_array.shape[1])
                ]
            ).T

    return signal_array, target_sample_rate


def flip_signal(signal_array: np.ndarray) -> np.ndarray:
    """
    Flip a signal array along its time axis.

    If the input array is one-dimensional, it is reversed.
    If it is multi-dimensional, it is reversed along its second axis.

    Parameters
    ----------
    signal_array : np.ndarray
        The numpy array representing the signal to be flipped. Can be 1-D or multi-dimensional.

    Returns
    -------
    np.ndarray
        The flipped numpy array.
    """
    if np.ndim(signal_array) == 1:
        reversed_array = np.flip(signal_array)
    else:
        reversed_array = np.flip(signal_array, axis=1)
    return reversed_array


def to_db_normalized(signal_array: np.ndarray) -> np.ndarray:
    """
    Normalize the given signal array to decibels relative to its maximum value per row.

    The function computes the logarithmic (decibel) value of each element in the array,
    normalized to the maximum value of each row. This operation is performed row-wise.

    Parameters
    ----------
    signal_array : np.ndarray
        A 2D NumPy array where each row represents a signal.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of the same shape as `signal_array`, with each element
        representing the decibel value normalized to the row's maximum value.
    """
    max_values = np.max(signal_array, axis=1, keepdims=True)
    return 10 * np.log10(signal_array / max_values)


def calculate_average_energy(signal: np.ndarray, decibel_offset: float) -> float:
    """
    Calculate the average energy of the signal after a specified decibel offset.

    Parameters
    ----------
    signal : np.ndarray
        The signal array.
    decibel_offset : float
        The decibel offset to start the calculation.

    Returns
    -------
    float
        Average energy level in decibels.
    """
    start_index = int(round(0.9 * len(signal)))
    noise_signal = signal[start_index:]
    max_signal = np.max(signal)
    return (
        10 * np.log10(np.sum(noise_signal) / len(noise_signal) / max_signal)
        + decibel_offset
    )


def results_to_dataframe(
    parameters: Dict[str, np.ndarray], band_freqs: np.ndarray
) -> pd.DataFrame:
    """
    Convert reverberation time parameters and band frequencies to a DataFrame.

    Parameters
    ----------
    parameters : Dict[str, np.ndarray]
        A dictionary with keys as parameter names and values as numpy arrays of their values.
    band_freqs : np.ndarray
        An array of band frequencies.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame where rows represent parameters and columns represent band frequencies.
    """
    formatted_columns = format_freqs(band_freqs)
    results_df = pd.DataFrame(parameters, index=formatted_columns).T
    return results_df


def format_freqs(band_freqs: float) -> str:
    """
    Convert an array of frequency values into a formatted string list.

    Parameters
    ----------
    band_freqs : array_like
        An array-like object containing frequency values (in Hz). Each element should be a number
        representing a frequency.

    Returns
    -------
    list of str
        A list of strings, each representing the formatted frequency value. Frequencies below 1000 Hz
        are suffixed with 'Hz', and frequencies of 1000 Hz and above are converted to kHz and suffixed
        with 'kHz'.

    Examples
    --------
    >>> format_freqs([31.5, 1000, 1250, 16000])
    ['31.5 Hz', '1 kHz', '1.2 kHz', '16 kHz']
    """
    string_list = []
    for value in band_freqs:
        if value < 1000:
            string_list.append(f"{value} Hz")
        else:
            kHz_value = value / 1000
            formatted_kHz_value = f"{kHz_value:.1f}".rstrip("0").rstrip(".")
            string_list.append(f"{formatted_kHz_value} kHz")

    return string_list


def average_stereo_parameters(parameters_df_list: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parameters_df_list).groupby(level=0).mean()
