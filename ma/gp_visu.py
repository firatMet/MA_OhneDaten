import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd


def plot_signals_with_hr(
    df_parti_segmented: list[pd.DataFrame],
    df_parti_hr: pd.DataFrame,
    df_ref: pd.DataFrame,
    df_ref_hr: pd.DataFrame,
    df_parti: pd.DataFrame,
    peaks_parti: list[int],
    peaks_ref: list[int],
    class_name: str,
    fig=None,
    axes=None,
) -> Figure:
    if axes is None:
        fig = plt.figure(figsize=(20, 9))
        axes = fig.subplots(nrows=2, ncols=1, sharex=True)

    match class_name:
        case "HR":
            label_prefix = "Heart Rate"
            bpm_label = "Heart Rate [bpm]"
        case "RR":
            label_prefix = "Respiratory Rate"
            bpm_label = "Respiratory Rate [bpm]"

    ref_signal = df_ref.columns[0]

    _ = plot_signals_with_r_peaks(
        df_parti_segmented=df_parti_segmented,
        df_ref=df_ref,
        df_parti=df_parti,
        peaks_parti=peaks_parti,
        peaks_ref=peaks_ref,
        ref_signal=ref_signal,
        fig=fig,
        ax=axes[0],
    )
    signal_name = df_parti_segmented[0].columns[0]

    axes[1].plot(
        df_parti_hr.index, df_parti_hr[class_name.lower()],  color="green", label=f"{label_prefix} - {signal_name}", marker="x"
    )
    axes[1].plot(
        df_ref_hr.index,
        df_ref_hr[class_name.lower()],
        color="orange",
        alpha=0.3,
        label=f"{label_prefix}- Reference Signal",
        marker="o",
    )

    axes[1].set_ylabel(bpm_label)
    axes[1].set_xlabel("Time [s]")
    axes[0].set_xlabel("")

    axes[1].grid()
    axes[1].legend()

    return fig


def plot_signals_with_r_peaks(
    df_parti_segmented: list[pd.DataFrame],
    df_ref: list[pd.DataFrame],
    df_parti: pd.DataFrame,
    peaks_parti: list[int],
    peaks_ref: list[int],
    ref_signal: str,
    fig=None,
    ax=None,
) -> Figure:
    signal_name = df_parti.columns[0]
    if ax is None:
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(111)
    # Signals:
    for segment in df_parti_segmented:
        color = _get_color_linestyle(segment=segment)
        ax.plot(segment.index, segment.values, color=color)
        _shade_ma_area(segment=segment, fig=fig)

    ax.plot(df_ref.index, df_ref.values, color="orange", alpha=0.3)

    # R-Peaks:
    signal_name = df_parti.columns[0]
    peaks_parti_flattened = [value + len(df_parti_segmented[0]) * idx for idx, sublist in enumerate(peaks_parti[0]) for value in sublist]
    peaks_ref_flattened = [value + len(df_parti_segmented[0])* idx for idx, sublist in enumerate(peaks_ref[0]) for value in sublist]
    ax.scatter(
        df_parti.index[peaks_parti_flattened],
        df_parti.iloc[peaks_parti_flattened][signal_name].values,
        color="black",
    )
    ax.scatter(
        df_ref.index[peaks_ref_flattened],
        df_ref.iloc[peaks_ref_flattened][ref_signal].values,
        color="black",
        alpha=0.3,
    )

    ax.grid()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [mV]")
    fig = _add_legend(fig=fig, signal_name=signal_name)

    return fig


def _get_color_linestyle(segment: pd.DataFrame) -> str:
    signal_name = segment.columns[0]
    match segment.attrs[f"{signal_name}_sq"]:
        # If the label is unknown:
        case -1:
            color = "purple"
        # Good Quality: Green
        case 0:
            color = "green"
        # Bad Quality: Blue
        case 1:
            color = "blue"
        # Noisy Quality: Red
        case 2:
            color = "red"

    return color


def _add_legend(fig: Figure, signal_name: str) -> Figure:
    ax = fig.get_axes()[0]

    # SQ
    (label_sq_unkown,) = ax.plot(
        [], [], linestyle="-", color="purple", label=f"{signal_name} - Unknown SQ Label"
    )
    (label_good,) = ax.plot(
        [], [], linestyle="-", color="green", label=f"{signal_name} - Good SQ"
    )
    (label_sq_bad,) = ax.plot(
        [], [], linestyle="-", color="blue", label=f"{signal_name} - Bad SQ"
    )
    (label_sq_noisy,) = ax.plot(
        [], [], linestyle="-", color="red", label=f"{signal_name} - Noisy SQ"
    )
    (label_ref_signal,) = ax.plot(
        [], [], linestyle="-", color="orange", label="Reference Signal"
    )
    # MA
    label_ma_unknown = Line2D(
        [0], [0], color="purple", alpha=0.4, linewidth=10, label="Unknown MA Label"
    )
    label_no_ma = Line2D(
        [0], [0], color="white", alpha=0.4, linewidth=10, label="No Motion Artifact"
    )
    label_ma = Line2D(
        [0], [0], color="lightgray", alpha=0.4, linewidth=10, label="Motion Artifact"
    )

    # label_ma_unknown = ax.axvspan(
    #     None, None, color="purple", alpha=0.4, label="Unknown MA Label"
    # )
    # label_no_ma = ax.axvspan(
    #     None, None, color="white", alpha=0.4, label="No Motion Artifact"
    # )
    # label_ma = ax.axvspan(
    #     None, None, color="lightgray", alpha=0.4, label="Motion Artifact"
    # )
    # R-Peaks
    label_r_peaks = ax.scatter([], [], marker="o", color="black", label="R-Peaks")

    sq_legends = ax.legend(
        handles=[
            label_sq_unkown,
            label_good,
            label_sq_bad,
            label_sq_noisy,
            label_ref_signal,
        ],
        loc="upper right",
    )
    ax.add_artist(sq_legends)

    ma_legends = ax.legend(
        handles=[label_ma_unknown, label_ma, label_no_ma, label_r_peaks],
        loc="upper left",
    )
    ax.add_artist(ma_legends)

    # fig.legend()

    return fig


def _shade_ma_area(segment: pd.DataFrame, fig: Figure) -> None:
    signal_name = segment.columns[0]
    match segment.attrs[f"{signal_name}_ma"]:
        # Unknown MA Label:
        case -1:
            ax = fig.get_axes()[0]
            ax.axvspan(segment.index[0], segment.index[-1], color="purple", alpha=0.4)
            return
        # No MA:
        case 0:
            return
        # Signal with MA:
        case 1:
            ax = fig.get_axes()[0]
            ax.axvspan(
                segment.index[0], segment.index[-1], color="lightgray", alpha=0.4
            )
            return
