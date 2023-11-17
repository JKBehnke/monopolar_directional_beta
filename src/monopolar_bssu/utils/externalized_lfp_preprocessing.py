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

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

HEMISPHERES = ["Right", "Left"]

# list of subjects with no BIDS transformation yet -> load these via poly5reader instead of BIDS
SUBJECTS_NO_BIDS = ["24", "28", "29", "48", "49", "56"]


# get index of each channel and get the corresponding LFP data
# plot filtered channels 1-8 [0]-[7] Right and 9-16 [8]-[15]
# butterworth filter: band pass -> filter order = 5, high pass 5 Hz, low-pass 95 Hz
def band_pass_filter_externalized(fs: int, signal: np.array):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a band pass filter to the signal
        - 5 Hz high pass
        - 95 Hz low pass
        - filter order: 3

    """
    # parameters
    filter_order = 3  # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 5  # 5Hz high-pass filter
    frequency_cutoff_high = 95  # 95 Hz low-pass filter

    # create and apply the filter
    b, a = scipy.signal.butter(
        filter_order,
        (frequency_cutoff_low, frequency_cutoff_high),
        btype="bandpass",
        output="ba",
        fs=fs,
    )
    band_pass_filtered = scipy.signal.filtfilt(b, a, signal)

    return band_pass_filtered


def high_pass_filter_externalized(fs: int, signal: np.array):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a band pass filter to the signal
        - 1 Hz high pass
        - filter order: 3
    """
    # parameters
    filter_order = 5  # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 1  # 1Hz high-pass filter

    # create and apply the filter
    b, a = scipy.signal.butter(
        filter_order, (frequency_cutoff_low), btype="highpass", output="ba", fs=fs
    )
    band_pass_filtered = scipy.signal.filtfilt(b, a, signal)

    return band_pass_filtered


# notch filter: 50 Hz
def notch_filter_externalized(fs: int, signal: np.array):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a notch filter to the signal


    """

    # parameters
    notch_freq = 50  # 50 Hz line noise in Europe
    Q = 30  # Q factor for notch filter

    # apply notch filter
    b, a = scipy.signal.iirnotch(w0=notch_freq, Q=Q, fs=fs)
    filtered_signal = scipy.signal.filtfilt(b, a, signal)

    return filtered_signal
