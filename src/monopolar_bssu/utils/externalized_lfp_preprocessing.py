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


def cut_lfp_in_20_sec_chunks(time_series: np.ndarray):
    """
    Input:
        - time_series: LFP data of one channel, sfreq 250 Hz, 2 min recording (30000 samples)

    Cut the LFP data into 20 sec chunks á 5000 samples

    Output:
        - result_dict: dictionary with 6 keys (1-6), each key has a value of 5000 samples of the LFP data

    """

    result_dict = {}

    for i in range(1, 7):  # list from 1 to 6
        start_index = (i - 1) * 5000  # 0, 5000, 10000, 15000, 20000, 25000
        end_index = i * 5000  # 5000, 10000, 15000, 20000, 25000, 30000
        result_dict[i] = time_series[start_index:end_index]

    return result_dict

def cut_lfp_in_short_epochs(time_series: np.ndarray, fourier_transform:str):
    """
    Input:
        - time_series: LFP data of one channel, sfreq 250 Hz, 2 min recording (30000 samples)
        - fourier_transform: "yes" or "no"

    Cut the LFP data into 
        - 2 chunks á 8750 samples = 35 sec
        - 3 chunks á 7500 samples = 30 sec
        - 4 chunks á 6250 samples = 25 sec
        - 5 chunks á 5000 samples = 20 sec
        - 5 chunks á 3750 samples = 15 sec
        - 5 chunks á 2500 samples = 10 sec
        - 5 chunks á 1250 samples = 5 sec
    """

    seconds_list = [35, 30, 25, 20, 15, 10, 5]
    samples_per_chunk = [sec * 250 for sec in seconds_list] # number of samles to get the desired seconds
    result_dict = {}

    for s, sec in enumerate(seconds_list):
        n_samples = samples_per_chunk[s]    
        n_chunks = int(len(time_series) / n_samples)

        for i in range(1, n_chunks + 1):
            start_index = (i - 1) * n_samples
            end_index = i * n_samples
            result_dict[f"power_spectrum_{sec}_sec_{i}"] = time_series[start_index:end_index]
    
    if fourier_transform == "yes":
        for key, value in result_dict.items():
            result_dict[key] = fourier_transform_to_psd(sfreq=250, lfp_data=value)["average_Zxx"]
    
    return result_dict # dictionary with keys: 35_sec_1, 35_sec_2, 30_sec_1, 30_sec_2, 30_sec_3, ...
    


def fourier_transform_to_psd(sfreq: int, lfp_data: np.ndarray):
    """
    Input:
        -

    Requirements:
        - 2 min recording
        - artefracts removed

    calculate the power spectrum:
        - window length = 250 # 1 second window length
        - overlap = window_length // 4 # 25% overlap
        - window = hann(window_length, sym=False)
        - frequencies, times, Zxx = scipy.signal.spectrogram(band_pass_filtered, fs=fs, window=window, noverlap=overlap, scaling="density", mode="psd", axis=0)


    Output:
        - frequencies
        - times
        - Zxx
        - average_Zxx
        - std_Zxx
        - sem_Zxx


    """

    ######### short time fourier transform to calculate PSD #########
    window_length = sfreq  # 1 second window length
    overlap = window_length // 4  # 25% overlap

    # Calculate the short-time Fourier transform (STFT) using Hann window
    window = hann(window_length, sym=False)

    frequencies, times, Zxx = scipy.signal.spectrogram(
        lfp_data,
        fs=sfreq,
        window=window,
        noverlap=overlap,
        scaling="density",
        mode="psd",
        axis=0,
    )
    # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
    # times: len=161, 0, 0.75, 1.5 .... 120.75
    # Zxx: 126 arrays, each len=161

    # average PSD across duration of the recording
    average_Zxx = np.mean(Zxx, axis=1)
    std_Zxx = np.std(Zxx, axis=1)
    sem_Zxx = std_Zxx / np.sqrt(Zxx.shape[1])

    return {
        "frequencies": frequencies,
        "times": times,
        "Zxx": Zxx,
        "average_Zxx": average_Zxx,
        "std_Zxx": std_Zxx,
        "sem_Zxx": sem_Zxx,
    }

