import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from impra.visualization import plot
from impra.process.filtering import Filters
from impra.process.smoothing import Smoothers
from impra.engine.core import ImpulseResponseAnalyzer

st.set_page_config(layout="wide")

filtering_map = {
    "Octave": Filters.OCTAVE_BAND,
    "Third-Octave": Filters.THIRD_OCTAVE_BAND,
}

smoothing_map = {
    "Schroeder": Smoothers.SCHROEDER_INVERSE_INTEGRAL,
    "Moving Median Filter": Smoothers.MOVING_MEDIAN_AVERAGE,
}

# Create a .tmp directory if it doesn't exist
if not os.path.exists(".tmp"):
    os.makedirs(".tmp")

# Initialize columns
col1, col2 = st.columns([1, 3])

# First column elements
with col1:
    st.markdown(
        "<h3 style='text-align: center;'>ImpRA</h3>",
        unsafe_allow_html=True,
    )
    st.divider()
    uploaded_file = st.file_uploader(
        "Load impulse response", type=["wav", "mp3", "ogg"]
    )

    st.divider()
    filter_type = st.radio(
        "Filter Type",
        ("Octave", "Third-Octave"),
        horizontal=True,
    )
    flip_impulse = st.checkbox(
        "Flip impulse response for filtering",
        value=True,
        key="flip_impulse",
    )
    st.divider()
    smoothing_args = {}
    smooth_by = st.radio(
        "Smooth by",
        ("Schroeder", "Moving Median Filter"),
        # label_visibility="hidden",
        horizontal=True,
    )
    if smooth_by == "Moving Median Filter":
        window_length = st.text_input("Window Length", value="50", key="window_length")
        smoothing_args["window_length"] = int(window_length)

    st.divider()
    analyze_button = st.button("Analyze", use_container_width=True)

# Second column placeholders (to be filled later)
with col2:
    fig_placeholder = st.empty()
    df_placeholder = st.empty()

# Callback when 'Analyze' is clicked
if analyze_button:
    if uploaded_file is not None:
        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Define the file path
        file_path = f".tmp/{timestamp}.wav"

        # Write the file to the .tmp directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        fig = plot.plot_impulse_response(file_path)
        fig_placeholder.plotly_chart(fig, use_container_width=True)

        cfg = {
            "sample_rate": 48_000,
            "filtering": {
                "type": filtering_map[filter_type],
                "flip_ir": flip_impulse,
            },
            "smoothing": {
                "type": smoothing_map[smooth_by],
                "args": smoothing_args,
            },
        }
        analyzer = ImpulseResponseAnalyzer(cfg)
        results_df = analyzer.analyze(file_path)
        df_placeholder.dataframe(results_df, use_container_width=True)
