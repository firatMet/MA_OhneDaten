from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Literal

plt.rcParams['pgf.preamble'] = r'\usepackage{tikz}'

#############
#    SQI    #
#############

def plot_confusion_matrix(cm, classes, title: str, cmap=plt.cm.Blues) -> Figure:
    fig, ax = plt.subplots()
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Adding color bar
    # fig.colorbar(cax)
    
    # Setting tick marks and labels
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Adding text annotations
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    # Setting labels and title
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.suptitle(title)

    # Adjust layout
    fig.tight_layout()
    
    return fig

def feature_scatter_plot(
    feature1: str, feature2: str, sqi_df: pd.DataFrame, label=Literal["ma", "sq"], save_image:bool = False,
) -> Figure:
    assert label in ["ma", "sq"]
    # Called from interactive plot
    if not isinstance(sqi_df, pd.DataFrame):
        sqi_df = sqi_df.value
    fig, ax = plt.subplots(figsize=(16, 9))

    match label:
        case "sq":
            good_df = sqi_df[sqi_df[label] == 0]
            bad_df = sqi_df[sqi_df[label] == 1]
            noisy_df = sqi_df[sqi_df[label] == 2]

            ax.scatter(good_df[feature1], good_df[feature2], label="Good")
            ax.scatter(bad_df[feature1], bad_df[feature2], label="Bad")
            ax.scatter(noisy_df[feature1], noisy_df[feature2], label="Noisy")

            fig.suptitle(
                f"Feature Scatter Plot Between {feature1.upper()} & {feature2.upper()} | # Good: {len(good_df)} - # Bad: {len(bad_df)} - # Noisy: {len(noisy_df)}"
            )
        case "ma":
            no_ma_df = sqi_df[sqi_df[label] == 0]
            ma_df = sqi_df[sqi_df[label] == 1]

            ax.scatter(no_ma_df[feature1], no_ma_df[feature2], label="No Ma")
            ax.scatter(ma_df[feature1], ma_df[feature2], label="MA")

            fig.suptitle(
                f"Feature Scatter Plot Between {feature1.upper()} & {feature2.upper()} | # No MA: {len(no_ma_df)} - # MA : {len(ma_df)}"
            )

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend()
    ax.grid()

    plt.show()

    if save_image:
        fig.savefig("feature_scatter_plot.pgf")


def feature_violin_plot(
    feature1: str, sqi_df: pd.DataFrame, label=Literal["ma", "sq"], save_image:bool = False,
) -> Figure:
    assert label in ["ma", "sq"]

    # Called from interactive plot
    if not isinstance(sqi_df, pd.DataFrame):
        sqi_df = sqi_df.value
    fig, ax = plt.subplots(figsize=(16, 9))

    match label:
        case "sq":
            good_df = sqi_df[sqi_df[label] == 0]
            bad_df = sqi_df[sqi_df[label] == 1]
            noisy_df = sqi_df[sqi_df[label] == 2]

            ax.scatter([0] * len(good_df), good_df[feature1], label="Good")
            ax.scatter([1] * len(bad_df), bad_df[feature1], label="Bad")
            ax.scatter([2] * len(noisy_df), noisy_df[feature1], label="Noisy")

            ax.violinplot(good_df[feature1], positions=[0])
            ax.violinplot(bad_df[feature1], positions=[1])
            ax.violinplot(noisy_df[feature1], positions=[2])

            fig.suptitle(
                f"Violin Plot of the Feature: {feature1}"#| # Good: {len(good_df)} - # Bad: {len(bad_df)} - # Noisy: {len(noisy_df)}"
            )
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Good", "Bad", "Noisy"])

        case "ma":
            no_ma_df = sqi_df[sqi_df[label] == 0]
            ma_df = sqi_df[sqi_df[label] == 1]

            ax.scatter([0] * len(no_ma_df), no_ma_df[feature1], label="No Ma")
            ax.scatter([1] * len(ma_df), ma_df[feature1], label="MA")

            ax.violinplot(no_ma_df[feature1], positions=[0])
            ax.violinplot(ma_df[feature1], positions=[1])

            fig.suptitle(
                f"Violin Plot of the Feature {feature1}"#| # No Ma: {len(no_ma_df)} - # MA: {len(ma_df)}"
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No Ma", "MA"])

    ax.legend()
    ax.grid()

    if save_image:
        fig.savefig("feature_violin_plot.pgf")
