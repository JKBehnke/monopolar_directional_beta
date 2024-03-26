""" Analysis of 20 sec chunks of 2 min externalized recordings """


import os
import pickle

import fooof
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof.plts.spectra import plot_spectrum
from scipy.stats import shapiro

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import loadResults as loadResults
from ..utils import sub_recordings_dict as sub_recordings_dict
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..externalized_lfp import feats_ssd as feats_ssd

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

# patient_metadata = load_data.load_patient_metadata_externalized()
PATIENT_METADATA = io_externalized.load_excel_data(filename="patient_metadata")
HEMISPHERES = ["Right", "Left"]
DIRECTIONAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]
BSSU_CHANNELS = [
    "01",
    "12",
    "23",
    "1A2A",
    "1B2B",
    "1C2C",
    "1A1B",
    "1A1C",
    "1B1C",
    "2A2B",
    "2A2C",
    "2B2C",
]

BSSU_RING_CHANNELS = [
    "01",
    "12",
    "23",
]

BSSU_SEGM_INTER_CHANNELS = [
    "1A2A",
    "1B2B",
    "1C2C",
]

BSSU_SEGM_INTRA_CHANNELS = [
    "1A1B",
    "1A1C",
    "1B1C",
    "2A2B",
    "2A2C",
    "2B2C",
]

FILTER = {
    "notch_and_band_pass_filtered": "filtered_lfp_250Hz",
    "unfiltered": "lfp_resampled_250Hz",
    "only_high_pass_filtered": "only_high_pass_lfp_250Hz",
}

SPECTRA_LIST = ["power_spectrum_2min", 
                "power_spectrum_20sec_1", 
                "power_spectrum_20sec_2", 
                "power_spectrum_20sec_3", 
                "power_spectrum_20sec_4", 
                "power_spectrum_20sec_5"]

FREQ_BAND_LIST = ["beta", "high_beta", "low_beta"]


# Data: io_externalized.load_externalized_pickle(filename="power_spectra_BSSU_externalized_20sec_group_notch_and_band_pass_filtered")
# column "chunks": each row (sub, hem, channel) with dictionary keys 1-6, values are time series of 20 seconds (5000 values)


def load_20sec_time_series(filtered:str):
    """
    Input: 
        - filtered: str "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
    """

    # Load data
    data = io_externalized.load_externalized_pickle(filename=f"power_spectra_BSSU_externalized_20sec_group_{filtered}")
    
    return data

def fourier_transform_per_20sec(dict_20_sec_time_series:dict):
    """
    For each subject, hemisphere, BSSU channel
    Input:
        - dict_20_sec_chunks: dictionary with keys (1-6) and values are time series of 20 seconds (5000 values)
    """

    dict_20_sec_power_spectra = {}

    # for each 20 sec recording, perform fourier transform
    for c in dict_20_sec_time_series:

        # check if recording exists
        if len(dict_20_sec_time_series[c]) == 0:
            print(f"20 sec rec number {c} does not exist")
            continue

        # get time series
        time_series = dict_20_sec_time_series[c]

        # fourier transform
        fourier_transform = externalized_lfp_preprocessing.fourier_transform_to_psd(sfreq=250, lfp_data=time_series)

        # save power spectrum
        dict_20_sec_power_spectra[c] = fourier_transform["average_Zxx"] # len 126, frequencies 0-125 Hz

    return dict_20_sec_power_spectra


def reorganize_data(filtered:str):
    """ 
    1. Load correct data with externalized BSSU channels and 20 sec chunks
    2. Run through each subject, hemisphere, channel
    3. Perform fourier transform on each 20 sec chunk to get power spectrum of each 20 sec
    4. save all 20 second power spectra and the 2min power spectrum in a dataframe

    -> useful for Cluster analysis!

    """
    dataframe_all_power_spectra = pd.DataFrame()

    loaded_data = load_20sec_time_series(filtered=filtered) 

    sub_list = loaded_data["subject"].unique() # list of subjects

    for sub in sub_list:

        sub_data = loaded_data[loaded_data["subject"] == sub]

        for hem in HEMISPHERES:

            hem_data = sub_data[sub_data["hemisphere"] == hem]

            for ch in BSSU_CHANNELS:

                chan_data = hem_data[hem_data["channel"] == ch]

                # get the 2 min power spectrum
                data_2min = chan_data.fourier_transform_2min.values[0] # dictionary with keys: "frequencies", "times", "average_Zxx", "std_Zxx", "sem_Zxx"

                power_spectrum_2min = data_2min["average_Zxx"]
                frequencies = data_2min["frequencies"]

                # get the 20 sec power spectra
                dict_20_sec_time_series = chan_data.chunks.values[0] # dictionary with keys 1-6, values are time series of 20 seconds (5000 values)

                dict_20_sec_power_spectra = fourier_transform_per_20sec(dict_20_sec_time_series=dict_20_sec_time_series)
                # dictionary with keys 1-6, values are power spectra of 20 sec recordings

                # save in dataframe
                one_row_data = {
                    "subject": [sub],
                    "hemisphere": [hem],
                    "channel": [ch],
                    "power_spectrum_2min": [power_spectrum_2min],
                    "frequencies": [frequencies],
                    "power_spectrum_20sec_1": [dict_20_sec_power_spectra[1]],
                    "power_spectrum_20sec_2": [dict_20_sec_power_spectra[2]],
                    "power_spectrum_20sec_3": [dict_20_sec_power_spectra[3]],
                    "power_spectrum_20sec_4": [dict_20_sec_power_spectra[4]],
                    "power_spectrum_20sec_5": [dict_20_sec_power_spectra[5]],
                }

                dataframe_one_row = pd.DataFrame(one_row_data)

                dataframe_all_power_spectra = pd.concat([dataframe_all_power_spectra, dataframe_one_row], ignore_index=True)
        
    return dataframe_all_power_spectra


def caculate_freq_band_mean_sd(power_spectrum=None, frequencies=None):
    """
    Calculate mean and standard deviation of power in frequency bands
        - beta power (13-30 Hz)
        - high beta power (21-35 Hz)
        - low beta power (13-20 Hz)
    """

    beta_power = power_spectrum[(frequencies >= 13) & (frequencies <= 35)]
    high_beta_power = power_spectrum[(frequencies >= 21) & (frequencies <= 35)]
    low_beta_power = power_spectrum[(frequencies >= 13) & (frequencies <= 20)]
    
    beta_power_mean = np.mean(power_spectrum[(frequencies >= 13) & (frequencies <= 35)])
    high_beta_power_mean = np.mean(power_spectrum[(frequencies >= 21) & (frequencies <= 35)])
    low_beta_power_mean = np.mean(power_spectrum[(frequencies >= 13) & (frequencies <= 20)])

    beta_power_sd = np.std(power_spectrum[(frequencies >= 13) & (frequencies <= 35)])
    high_beta_power_sd = np.std(power_spectrum[(frequencies >= 21) & (frequencies <= 35)])
    low_beta_power_sd = np.std(power_spectrum[(frequencies >= 13) & (frequencies <= 20)])

    beta_power_freq = frequencies[(frequencies >= 13) & (frequencies <= 35)]
    high_beta_power_freq = frequencies[(frequencies >= 21) & (frequencies <= 35)]
    low_beta_power_freq = frequencies[(frequencies >= 13) & (frequencies <= 20)]

    return {
        "beta_power": beta_power,
        "high_beta_power": high_beta_power,
        "low_beta_power": low_beta_power,
        "beta_power_mean": beta_power_mean, 
        "high_beta_power_mean": high_beta_power_mean, 
        "low_beta_power_mean": low_beta_power_mean, 
        "beta_power_sd": beta_power_sd, 
        "high_beta_power_sd": high_beta_power_sd, 
        "low_beta_power_sd": low_beta_power_sd,
        "beta_power_freq": beta_power_freq,
        "high_beta_power_freq": high_beta_power_freq,
        "low_beta_power_freq": low_beta_power_freq
    }



def frequency_band_mean_sd(filtered:str):
    """
    From all power spectra, get:
        - beta power (13-30 Hz) mean ± SD
        - high beta power (21-35) mean ± SD
        - low beta power (13-20) mean ± SD


    Output is a dictionary with 3 dataframes: keys "beta_band_result", "high_beta_band_result", "low_beta_band_result"
    These can be used for group analysis: Friedman, repeated measures ANOVA, etc.
        
    """

    # load data
    power_spectra_data = reorganize_data(filtered=filtered)

    # get beta power mean ± SD
    beta_band_result = pd.DataFrame()
    high_beta_band_result = pd.DataFrame()
    low_beta_band_result = pd.DataFrame()


    # loop through each row
    sub_list = power_spectra_data["subject"].unique() # list of subjects

    for sub in sub_list:

        sub_data = power_spectra_data[power_spectra_data["subject"] == sub]

        for hem in HEMISPHERES:

            hem_data = sub_data[sub_data["hemisphere"] == hem]

            for ch in BSSU_CHANNELS:

                chan_data = hem_data[hem_data["channel"] == ch]

                frequencies = chan_data.frequencies.values[0]

                # get freq band power mean ± SD
                result_dictionary_per_row = {}

                for s, spectrum in enumerate(SPECTRA_LIST):

                    power_spectrum = chan_data[spectrum].values[0]

                    power_result_dict = caculate_freq_band_mean_sd(power_spectrum=power_spectrum, frequencies=frequencies)

                    result_dictionary_per_row[s] = power_result_dict

                for freq in FREQ_BAND_LIST:

                    # save in dataframe
                    one_row_dict = {
                        "subject": [sub],
                        "hemisphere": [hem],
                        "channel": [ch],
                        "frequencies": [result_dictionary_per_row[0][f"{freq}_power_freq"]],
                        "power_spectrum_2min": [result_dictionary_per_row[0][f"{freq}_power"]],
                        "power_spectrum_20sec_1": [result_dictionary_per_row[1][f"{freq}_power"]],
                        "power_spectrum_20sec_2": [result_dictionary_per_row[2][f"{freq}_power"]],
                        "power_spectrum_20sec_3": [result_dictionary_per_row[3][f"{freq}_power"]],
                        "power_spectrum_20sec_4": [result_dictionary_per_row[4][f"{freq}_power"]],
                        "power_spectrum_20sec_5": [result_dictionary_per_row[5][f"{freq}_power"]],
                        "power_spectrum_2min_mean": [result_dictionary_per_row[0][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_1_mean": [result_dictionary_per_row[1][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_2_mean": [result_dictionary_per_row[2][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_3_mean": [result_dictionary_per_row[3][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_4_mean": [result_dictionary_per_row[4][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_5_mean": [result_dictionary_per_row[5][f"{freq}_power_mean"]],
                        "power_spectrum_2min_sd": [result_dictionary_per_row[0][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_1_sd": [result_dictionary_per_row[1][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_2_sd": [result_dictionary_per_row[2][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_3_sd": [result_dictionary_per_row[3][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_4_sd": [result_dictionary_per_row[4][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_5_sd": [result_dictionary_per_row[5][f"{freq}_power_sd"]],
                    }

                    one_row_dataframe = pd.DataFrame(one_row_dict)

                    if freq == "beta":
                        beta_band_result = pd.concat([beta_band_result, one_row_dataframe], ignore_index=True)
                    
                    elif freq == "high_beta":
                        high_beta_band_result = pd.concat([high_beta_band_result, one_row_dataframe], ignore_index=True)
                    
                    elif freq == "low_beta":
                        low_beta_band_result = pd.concat([low_beta_band_result, one_row_dataframe], ignore_index=True)

    return {
        "beta_band_result": beta_band_result,
        "high_beta_band_result": high_beta_band_result,
        "low_beta_band_result": low_beta_band_result
    }


def rank_power_2min(filtered:str, freq_band:str, channel_group:str):
    """
    This function ranks the 2min power for each subject and hemisphere

    Input:
        - filtered: str "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
        - freq_band: str "beta", "high_beta", "low_beta"
        - channel_group: str "all", "ring", "segm_inter", "segm_intra"

    """

    # save only the maximal power channel rows
    maximal_power_data = pd.DataFrame()
    ranked_power_2min = pd.DataFrame()

    if channel_group == "all":
        channels = BSSU_CHANNELS

    elif channel_group == "ring":
        channels = BSSU_RING_CHANNELS
    
    elif channel_group == "segm_inter":
        channels = BSSU_SEGM_INTER_CHANNELS
    
    elif channel_group == "segm_intra":
        channels = BSSU_SEGM_INTRA_CHANNELS

    freq_band_data_all = frequency_band_mean_sd(filtered=filtered)

    spec_freq_band_data = freq_band_data_all[f"{freq_band}_band_result"]

    # loop through each row
    sub_list = spec_freq_band_data["subject"].unique() # list of subjects

    for sub in sub_list:

        sub_data = spec_freq_band_data[spec_freq_band_data["subject"] == sub]

        for hem in HEMISPHERES:

            hem_data = sub_data[sub_data["hemisphere"] == hem]
            chan_data = hem_data[hem_data["channel"].isin(channels)]
            chan_data_copy = chan_data.copy()

            # rank power
            chan_data_copy["rank_2min"] = chan_data_copy["power_spectrum_2min_mean"].rank(ascending=False)
            ranked_power_2min = pd.concat([ranked_power_2min, chan_data_copy], ignore_index=True)

            # only keep maximal power channel
            maximal_power_channel_2min = chan_data_copy[chan_data_copy["rank_2min"] == 1]
            maximal_power_data = pd.concat([maximal_power_data, maximal_power_channel_2min], ignore_index=True)
    
    maximal_power_data["patient_id"] = maximal_power_data["subject"] + "_" + maximal_power_data["hemisphere"] + "_" + maximal_power_data["channel"]
    patient_ids = maximal_power_data['patient_id'].unique()

    return {
        "ranked_power_2min": ranked_power_2min, 
        "maximal_power_data": maximal_power_data,
        "maximal_power_patient_ids": patient_ids}


def shapiro_wilk_means_distribution(filtered:str, freq_band:str, channel_group:str):
    """
    Test if the power means of repeated measurements are normally distributed within each subject hemisphere, maximal power channel

    """

    shapiro_wilk_result = pd.DataFrame()

    # load the maximal power data for each sub, hem
    maximal_power_data = rank_power_2min(filtered=filtered, freq_band=freq_band, channel_group=channel_group)["maximal_power_data"]

    # merge together columns "subject", "hemisphere", "channel" to get unique patient ids
    maximal_power_data["patient_id"] = maximal_power_data["subject"] + "_" + maximal_power_data["hemisphere"] + "_" + maximal_power_data["channel"]
    patient_ids = maximal_power_data['patient_id'].unique()

    for patient_id in patient_ids:
        patient_data = maximal_power_data[maximal_power_data['patient_id'] == patient_id]
        
        measurements = []
        for i in range(1, 6):
            column_name = f'power_spectrum_20sec_{i}_mean'
            measurements.append(patient_data[column_name].values[0])

        stat, p = shapiro(np.array(measurements))

        if p > 0.05:
            print(f'Data for patient {patient_id}, power_spectrum_20sec_{i}_mean is normally distributed (p={p})')
        else:
            print(f'Data for patient {patient_id}, power_spectrum_20sec_{i}_mean is not normally distributed (p={p})')

        shapiro_wilk_one_row = {
            "patient_id": [patient_id],
            "shapiro_wilk_stat": [stat],
            "shapiro_wilk_p": [p],
            "normal_distribution": [p > 0.05],
        }

        shapiro_wilk_one_row_df = pd.DataFrame(shapiro_wilk_one_row)
        shapiro_wilk_result = pd.concat([shapiro_wilk_result, shapiro_wilk_one_row_df], ignore_index=True)
    
    return shapiro_wilk_result

