import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from typing import Literal

from ma.utils import (
    _segment_participant_data,
    _bandpass_filter,
)


def plot_participant(
    df_participant: pd.DataFrame,
    mark_segments: list[int] = None,
    segment_type: Literal["HR", "RR"] = "HR",
    filtered: bool = False,
    title: str = None,
    **kwargs,
) -> Figure:
    fig, ax = plt.subplots()
    if filtered:
        df_participant = _bandpass_filter(df_participant=df_participant)
    if not kwargs:
        for col in df_participant.columns:
            ax.plot(df_participant.index, df_participant[col], label=col)

    if kwargs:
        for col in kwargs.values():
            ax.plot(df_participant.index, df_participant[col], label=col)

    if mark_segments:
        segmented_signal = _segment_participant_data(
            df_participant=df_participant, class_name=segment_type
        )
        for segment in mark_segments:
            segment = segmented_signal[segment - 1]
            start_time = segment.index[0]
            stop_time = segment.index[-1]
            ax.axvline(
                x=start_time, linestyle="--", linewidth=2, color="red", alpha=0.3
            )
            ax.axvline(x=stop_time, linestyle="--", linewidth=2, color="red", alpha=0.3)
    ax.grid()
    fig.legend()
    if title:
        fig.suptitle(title)
    return fig


def plot_participant_segments(
    df_participant: pd.DataFrame,
    segment_nos: list,
    class_name: Literal["HR", "RR"],
    title: str = None,
    **kwargs,
) -> Figure:
    assert isinstance(
        segment_nos, list
    ), f"segment_nos should be a list, segment_nos:{segment_nos}"

    segmented_dfs = _segment_participant_data(
        df_participant=df_participant, class_name=class_name
    )

    # Plot
    fig, ax = plt.subplots()
    for segment_no in segment_nos:
        df_segment = segmented_dfs[segment_no - 1]
        if not kwargs:
            col = df_segment.columns
            for col in df_segment.columns:
                ax.plot(df_segment.index, df_segment[col], label=col)
        if kwargs:
            for col in kwargs.values():
                ax.plot(df_segment.index, df_segment[col], label=col)
    ax.grid()
    fig.legend()
    if title:
        fig.suptitle(title)
    return fig
