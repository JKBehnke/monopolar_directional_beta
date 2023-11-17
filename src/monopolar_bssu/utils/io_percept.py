""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram

from ..utils import find_folders as find_folders

GROUP_RESULTS_PATH = find_folders.get_local_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_local_path(folder="GroupFigures")

HEMISPHERES = ["Right", "Left"]

SUBJECTS = [
    "017",
    "019",
    "021",
    "024",
    "025",
    "028",
    "029",
    "030",
    "031",
    "032",
    "033",
    "036",
    "040",
    "041",
    "045",
    "047",
    "048",
    "049",
    "050",
    "052",
    "055",
    "059",
    "060",
    "061",
    "062",
    "063",
    "065",
    "066",
]
# excluded subjects (ECG artifacts): "026", "038",


def save_result_dataframe_as_pickle(data: pd.DataFrame, filename: str):
    """
    Input:
        - data: must be a pd.DataFrame()
        - filename: str, e.g."externalized_preprocessed_data"

    picklefile will be written in the group_results_path:

    """

    group_data_path = os.path.join(GROUP_RESULTS_PATH, f"{filename}.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(data, file)

    print(f"{filename}.pickle", f"\nwritten in: {GROUP_RESULTS_PATH}")


def save_fig_png_and_svg(path: str, filename: str, figure=None):
    """
    Input:
        - path: str
        - filename: str
        - figure: must be a plt figure

    """

    figure.savefig(
        os.path.join(path, f"{filename}.svg"),
        bbox_inches="tight",
        format="svg",
    )

    figure.savefig(
        os.path.join(path, f"{filename}.png"),
        bbox_inches="tight",
    )

    print(f"Figures {filename}.svg and {filename}.png", f"\nwere written in: {path}.")
