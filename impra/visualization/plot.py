from typing import Union
from pathlib import Path

import numpy as np
import librosa
import plotly.graph_objs as go


def plot_impulse_response(audio_path: Union[Path, str]) -> go.Figure:
    signal_array, sample_rate = librosa.load(audio_path, sr=None)
    time_array = np.arange(0, len(signal_array)) / sample_rate

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_array, y=signal_array, mode="lines"))

    fig.update_layout(
        title="Impulse response visualizer",
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
        template="plotly_white",
    )

    return fig
