from typing import Union
from pathlib import Path

import numpy as np
import plotly.graph_objs as go

from impra.utils import load_signal
from impra.engine.core import ImpulseResponseAnalyzer, cfg


def plot_energy_curve(audio_path: Union[Path, str]) -> go.Figure:
    signal_array, sample_rate = load_signal(audio_path, 48000)

    if signal_array.ndim > 1:
        signal_array = signal_array.T[0, :]

    impra_engine = ImpulseResponseAnalyzer(cfg)
    filtered_signal, _ = impra_engine._filter_single_signal(signal_array)
    energy_smoothed_db, energy_envelope = impra_engine._energy_single_signal(
        filtered_signal
    )

    loc = 5

    energy_smoothed_db_1k = energy_smoothed_db[loc]
    energy_envelope_1k = energy_envelope[loc]

    time_array = np.arange(0, len(energy_envelope_1k)) / sample_rate

    hover_text = [
        f"Time [s]: {time:.2f}<br>Energy [dB]: {energy:.2f}"
        for time, energy in zip(time_array, energy_smoothed_db_1k)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=energy_smoothed_db_1k,
            mode="lines",
            hoverinfo="text",
            text=hover_text,  # Assign the hover text for each point
        )
    )

    fig.update_layout(
        title="Impulse response visualizer",
        xaxis_title="Time [s]",
        yaxis_title="Energy [dB]",
        template="plotly_white",
    )

    return fig
