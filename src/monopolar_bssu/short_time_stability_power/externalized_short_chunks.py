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

BSSU_SEGM_CHANNELS = [
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

FILTER = {
    "notch_and_band_pass_filtered": "filtered_lfp_250Hz",
    "unfiltered": "lfp_resampled_250Hz",
    "only_high_pass_filtered": "only_high_pass_lfp_250Hz",
}

SECONDS_LIST = [35, 30, 25, 20, 15, 10, 5]

SPECTRA_LIST = ["power_spectrum_2_min", 
                "power_spectrum_20_sec_1", 
                "power_spectrum_20_sec_2", 
                "power_spectrum_20_sec_3", 
                "power_spectrum_20_sec_4", 
                "power_spectrum_20_sec_5"]

FREQ_BAND_LIST = ["beta", "high_beta", "low_beta"]


# Data: io_externalized.load_externalized_pickle(filename="power_spectra_BSSU_externalized_20sec_group_notch_and_band_pass_filtered")
# column "chunks": each row (sub, hem, channel) with dictionary keys 1-6, values are time series of 20 seconds (5000 values)

def exclude_patients():
    """
    Exclude patients with bad recordings
        - 069_Left: recording too short because too many artifacts were taken out
    """

    exclude_sub_hem = ["069_Left"] 

    return exclude_sub_hem

def load_2min_and_short_epochs_power_spectra(incl_sub:list, filtered:str, sec_per_epoch:int):
    """
    This function cuts the 2min time series of externalized BSSU data into short epochs

    Output: 
        - dataframe with 2 min power spectra and power spectra of short epochs of the given seconds per epoch

    """
    filter_name = FILTER[filtered]

    # load the BSSU externalized data
    extern_bssu_data = io_externalized.load_externalized_pickle(
        filename="externalized_directional_bssu_channels",
        reference="bipolar_to_lowermost",
    )

    short_epochs_results_df = pd.DataFrame()
    
    if incl_sub == ["all"]:
        bids_id_unique = list(extern_bssu_data["BIDS_id"].unique())
    
    else:
        bids_id_unique = []
        for sub in incl_sub:
            sub_bids_id = sub_recordings_dict.get_bids_id_from_sub(sub)
            bids_id_unique.append(sub_bids_id)
    
    for bids_id in bids_id_unique:
        sub_data = extern_bssu_data[extern_bssu_data["BIDS_id"] == bids_id]
        sub = sub_recordings_dict.get_sub_from_bids_id(bids_id)
        
        for hem in HEMISPHERES:
            hem_data = sub_data[sub_data["hemisphere"] == hem]
            
            for ch in BSSU_CHANNELS:
                chan_data = hem_data[hem_data["bipolar_channel"] == ch]
                
                # get the 2 min time series
                time_series_2min = chan_data[filter_name].values[0]

                # fourier transform
                fourier_transform_2min = externalized_lfp_preprocessing.fourier_transform_to_psd(
                    sfreq=250, 
                    lfp_data=time_series_2min)
                
                power_spectrum_2min = fourier_transform_2min["average_Zxx"]
                frequencies = fourier_transform_2min["frequencies"]
                
                # cut 2 min time series in multiple short epochs
                short_epochs_all = externalized_lfp_preprocessing.cut_lfp_in_short_epochs(time_series=time_series_2min, fourier_transform="yes")
                # dictionary with keys: 35_sec_1, 35_sec_2, 30_sec_1, 30_sec_2, 30_sec_3, ... 
                # and values are the power spectra

                # save the data
                sub_results = {
                    "bids_id": [bids_id],
                    "subject": [sub],
                    "hemisphere": [hem],
                    "channel": [ch],
                    "power_spectrum_2_min": [power_spectrum_2min],
                    "frequencies": [frequencies],
                    }
                
                # only keep the epochs of interest
                for key, value in short_epochs_all.items():

                    if f"power_spectrum_{str(sec_per_epoch)}_sec" in key:
                        sub_results[f"{key}"] = [value]
                

                # save the data in dataframe
                sub_results_df = pd.DataFrame(sub_results)
                short_epochs_results_df = pd.concat([short_epochs_results_df, sub_results_df], ignore_index=True)
    
    # save the dataframe as pickle
    io_externalized.save_result_dataframe_as_pickle(data=short_epochs_results_df, 
                                             filename=f"power_spectra_BSSU_externalized_{filtered}_2min_and_{sec_per_epoch}sec")
    
    return short_epochs_results_df


def  choose_epochs(filtered:str, sec_per_epoch:int):
    """
    1. Check the number of epochs for each subject, hemisphere, channel
    2. depending on the number of epochs, choose which epochs to include in the analysis
    - if > 4 epochs: choose only 4 epochs, evenly distributed
    """

    data = io_externalized.load_externalized_pickle(filename=f"power_spectra_BSSU_externalized_{filtered}_2min_and_{sec_per_epoch}sec")

    short_power_spectra_columns = []

    for column in data.columns:
        if "power_spectrum" and "sec" in column:
            short_power_spectra_columns.append(column)

    number_epochs = len(short_power_spectra_columns)     

    if number_epochs < 4:
        print(f"Number of epochs too small for {sec_per_epoch}: {number_epochs}")  
        columns_to_include = []

    elif number_epochs == 4:
        columns_to_include = short_power_spectra_columns     
    
    elif number_epochs > 4:
        columns_to_include = []

        # choose 4 epochs evenly distributed, with the condition that all chosen columns do not contain NaN values
        # first check which column data does not contain NaN values
        columns_without_nan = []
        for col in short_power_spectra_columns:
            if data[col].isnull().values.any() == False:
                columns_without_nan.append(col)
        
        # check again if now the number of columns without NaN values is > 4
        if len(columns_without_nan) < 4:
            print(f"Number of columns without NaN values too small for {sec_per_epoch}: {len(columns_without_nan)}")
            columns_to_include = []

        elif len(columns_without_nan) == 4:
            columns_to_include = columns_without_nan
        
        else:
            step = len(columns_without_nan) // 4 # double slash for integer division
            for i in range(1, 5):
                index = int(i * step)

                if index == len(columns_without_nan):
                    index = index - 1
                    
                columns_to_include.append(columns_without_nan[index])
    
    # get the last integer of each column name in the columns to include
    if len(columns_to_include) > 0:
        sub_strings = [col.split("_")[-1] for col in columns_to_include]
    else:
        sub_strings = []
    
    # only keep the columns to include and check if there are now NaN values
    kept_data = data[["subject", "hemisphere", "channel"] + columns_to_include]
    if kept_data.isnull().values.any():
        print(f"Kept data contains NaN values")


    return {
        "columns_to_include": columns_to_include,
        "number_epochs": number_epochs,
        "sub_strings": sub_strings,
        "kept_data": kept_data
        }




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


def calculate_freq_band_mean_sd(power_spectrum=None, frequencies=None):
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



def frequency_band_mean_sd(filtered:str, sec_per_epoch:int):
    """
    From all power spectra, get:
        - beta power (13-30 Hz) mean ± SD
        - high beta power (21-35) mean ± SD
        - low beta power (13-20) mean ± SD


    Output is a dictionary with 3 dataframes: keys "beta_band_result", "high_beta_band_result", "low_beta_band_result"
    These can be used for group analysis: Friedman, repeated measures ANOVA, etc.
        
    """

    # load data
    #power_spectra_data = load_2min_and_short_epochs_power_spectra(incl_sub=["all"], filtered=filtered, sec_per_epoch=sec_per_epoch)
    power_spectra_data = io_externalized.load_externalized_pickle(filename=f"power_spectra_BSSU_externalized_{filtered}_2min_and_{sec_per_epoch}sec")
    
    # get info how which epochs to include
    epochs_info = choose_epochs(filtered=filtered, sec_per_epoch=sec_per_epoch)
    columns_to_include = epochs_info["columns_to_include"]
    all_columns_with_2min = ["power_spectrum_2_min"] + columns_to_include

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

            # exclude patient hemispheres if necessary
            if f"{sub}_{hem}" in exclude_patients():
                continue

            for ch in BSSU_CHANNELS:

                chan_data = hem_data[hem_data["channel"] == ch]

                frequencies = chan_data.frequencies.values[0]

                # get freq band power mean ± SD
                result_dictionary_per_row = {}

                for s, spectrum in enumerate(all_columns_with_2min):

                    power_spectrum = chan_data[spectrum].values[0]

                    if type(power_spectrum) == float:
                        print(f"Power spectrum {spectrum} of {sub} {hem} {ch} is a float!: {power_spectrum}")

                    power_result_dict = calculate_freq_band_mean_sd(power_spectrum=power_spectrum, frequencies=frequencies)

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
                        "power_spectrum_2min_mean": [result_dictionary_per_row[0][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_1_mean": [result_dictionary_per_row[1][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_2_mean": [result_dictionary_per_row[2][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_3_mean": [result_dictionary_per_row[3][f"{freq}_power_mean"]],
                        "power_spectrum_20sec_4_mean": [result_dictionary_per_row[4][f"{freq}_power_mean"]],
                        "power_spectrum_2min_sd": [result_dictionary_per_row[0][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_1_sd": [result_dictionary_per_row[1][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_2_sd": [result_dictionary_per_row[2][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_3_sd": [result_dictionary_per_row[3][f"{freq}_power_sd"]],
                        "power_spectrum_20sec_4_sd": [result_dictionary_per_row[4][f"{freq}_power_sd"]],
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


def rank_power_2min(filtered:str, freq_band:str, channel_group:str, sec_per_epoch:int):
    """
    This function ranks the 2min power for each subject and hemisphere

    Input:
        - filtered: str "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
        - freq_band: str "beta", "high_beta", "low_beta"
        - channel_group: str "all", "ring", "segm_inter", "segm_intra", "segm"

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
    
    elif channel_group == "segm":
        channels = BSSU_SEGM_CHANNELS

    freq_band_data_all = frequency_band_mean_sd(filtered=filtered, sec_per_epoch=sec_per_epoch)

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

def get_statistics(data_info:str, data=None):
    """
    Caculates statistical information of the data
    Input:
        - data_info: str, information about the data
        - data: pd.Series, data to calculate statistics
    """

    # caculate Outliers
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    threshold = 1.5 

    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers_indices = outliers.index
    outliers_values = outliers.values

    # calculate shapiro wilk test
    stat, p = shapiro(np.array(data))

    if p > 0.05:
        print(f'Data is normally distributed (p={p})')
    else:
        print(f'Data is not normally distributed (p={p})')

    stats_dict = {
            "data_info": [data_info],
            "sample_size": [len(data)],
            "mean_cv": [np.mean(data)],
            "std_cv": [np.std(data)],
            "median_cv": [np.median(data)],
            "outliers_indices": [outliers_indices],
            "outliers_values": [outliers_values],
            "n_outliers": [len(outliers)],
            "min": [np.min(data)],
            "max": [np.max(data)],
            "25%": [q25],
            "50%": [np.percentile(data, 50)],
            "75%": [q75],
            "shapiro_wilk_stat": [stat],
            "shapiro_wilk_p": [p],
            "normal_distribution": [p > 0.05],
        }
    
    stats_df = pd.DataFrame(stats_dict)

    return stats_df





################### BETA VARIATION OF THE MAXIMAL BETA CHANNEL PER HEMISPHERE DURING 5X 20SEC #####################

def calculate_coefficient_of_variation(filtered:str, freq_band:str, channel_group:str, sec_per_epoch:int):
    """
    For a selected maximal beta (2min) channel per subject hemisphere:
    - Calculate the coefficient of variation of the 4 x 20 sec power averages 

    """
    cv_result = pd.DataFrame()

    # load the maximal power data for each sub, hem
    maximal_power_data = rank_power_2min(filtered=filtered, freq_band=freq_band, channel_group=channel_group, sec_per_epoch=sec_per_epoch)["maximal_power_data"]

    # merge together columns "subject", "hemisphere", "channel" to get unique patient ids
    maximal_power_data["patient_id"] = maximal_power_data["subject"] + "_" + maximal_power_data["hemisphere"] + "_" + maximal_power_data["channel"]
    patient_ids = maximal_power_data['patient_id'].unique()

    for patient_id in patient_ids:
        patient_data = maximal_power_data[maximal_power_data['patient_id'] == patient_id]

        measurements = [] # fill this list with the 4 x 20 sec power averages
        for i in range(1, 5):
            column_name = f'power_spectrum_20sec_{i}_mean' # power average within the freq band of interest
            measurements.append(patient_data[column_name].values[0])
        
        # calculate the coefficient of variation
        cv = (np.std(measurements) / np.mean(measurements)) * 100 # in percent
        cv_result_dict = {
            "patient_id": [patient_id],
            "cv": [cv],
            "length_of_power_spectra": [sec_per_epoch] # 20 seconds (list of 20 multiplied by the number of measurements 4)
        }

        cv_result_row = pd.DataFrame(cv_result_dict)
        cv_result = pd.concat([cv_result, cv_result_row], ignore_index=True)
    
    return cv_result

def plot_coefficient_of_variation(filtered:str, freq_band:str, channel_group:str, sec_per_epoch:int):
    """
    Plot the coefficient of variation of the 4 x 20 sec power averages of the maximal beta channel per hemisphere
    as a violin plot
    """

    coefficient_variation_data = calculate_coefficient_of_variation(filtered=filtered, freq_band=freq_band, channel_group=channel_group, sec_per_epoch=sec_per_epoch)

    # Plot the coefficient of variation
    #fig = plt.figure(figsize=(10, 10), layout='tight')
    #plt.subplot(1,1,1)

    fig, ax = plt.subplots(figsize=(10, 10))
    # add jitter to the x-coordinates
    power_spectrum_length = coefficient_variation_data["length_of_power_spectra"].values # e.g. 20 seconds (* number of patients )
    
    #jitter = np.random.normal(0, 0.3, len(power_spectrum_length))
    #x_jittered = power_spectrum_length + jitter

    # Plotting the box plot
    # plt.boxplot(coefficient_variation_data["cv"], 
    #             vert=True, 
    #             widths=0.2, 
    #             patch_artist=True, 
    #             boxprops=dict(facecolor='lightblue'),
    #             positions=[power_spectrum_length[0]])
    
    # Plot a violin plot showing the distribution of the coefficient of variation, with the patient ids in different colors
    plt.violinplot(coefficient_variation_data["cv"], 
                   showmeans=True, 
                   showextrema=True, 
                   showmedians=True,
                   positions=[power_spectrum_length[0]], # list of positions of the violins on the x axis
                   widths=0.1,)
    
    # plot each dot for each patient in the violin plot
    #colors = plt.cm.viridis(np.linspace(0, 2, len(coefficient_variation_data)))
    #for i in range(len(coefficient_variation_data)):
    plt.plot(power_spectrum_length, coefficient_variation_data["cv"], 'o', alpha=0.3, markersize=10, color="k") #color=colors[i]
    
    plt.xticks([sec_per_epoch], [f'{power_spectrum_length[0]} seconds']) # for violin plot
                
    # plt.bar(range(len(coefficient_variation_data["cv"])), coefficient_variation_data["cv"], color='skyblue')
    plt.ylabel('Coefficient of Variation (%)')
    plt.xlabel('Length of Power Spectra (seconds)')
    plt.title(f'Coefficient of Variation of 4x {freq_band} power in the maximal {freq_band} {channel_group} channels')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    io_externalized.save_fig_png_and_svg(figure=fig,
                                         filename=f"Coefficient_of_Variation_4x_{sec_per_epoch}sec_spectra_externalized_BSSU_maximal_{freq_band}_{channel_group}_{filtered}",
                                         path=GROUP_FIGURES_PATH
    )

def plot_coefficient_of_variation_multiple_durations(filtered:str, freq_band:str, channel_group:str):
    """
    Plot the coefficient of variation of the 4 x short power averages of the maximal beta channel per hemisphere
    as a violin plot
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    sec_per_epochs_list = [5, 10, 15, 20, 25]
    sample_size_and_infos = pd.DataFrame()
    
    for epoch in sec_per_epochs_list:

        coefficient_variation_data = calculate_coefficient_of_variation(filtered=filtered, freq_band=freq_band, channel_group=channel_group, sec_per_epoch=epoch)

        power_spectrum_length = coefficient_variation_data["length_of_power_spectra"].values # e.g. 20 seconds (* number of patients )

        # Plotting the box plot
        plt.boxplot(coefficient_variation_data["cv"], 
                    vert=True, 
                    widths=1.0, 
                    patch_artist=True, 
                    showmeans=True,
                    meanline=True,
                    boxprops=dict(facecolor='white'),
                    positions=[power_spectrum_length[0]])
        # orange line for the median
        # green dotted line for the mean
        
        
        # statistics = coefficient_variation_data["cv"].describe()
        
    
        # Plot a violin plot showing the distribution of the coefficient of variation, with the patient ids in different colors
        # plt.violinplot(coefficient_variation_data["cv"], 
        #             showmeans=True, 
        #             showextrema=True, 
        #             showmedians=True,
        #             positions=[power_spectrum_length[0]], # list of positions of the violins on the x axis
        #             widths=0.7,)
    
        # plot each dot for each patient in the violin plot
        plt.plot(power_spectrum_length, coefficient_variation_data["cv"], 'o', alpha=0.3, markersize=5, color="k") #color=colors[i]

        # epoch_info = {
        #     "epoch": [epoch],
        #     "sample_size": [len(coefficient_variation_data)],
        #     "mean_cv": [np.mean(coefficient_variation_data["cv"])],
        #     "std_cv": [np.std(coefficient_variation_data["cv"])],
        #     "median_cv": [np.median(coefficient_variation_data["cv"])],
            
        #     "statistics": [statistics],
        # }
        # epoch_info_df = pd.DataFrame(epoch_info)

        statistics_info = get_statistics(data_info=str(epoch), data=coefficient_variation_data["cv"])

        # get the outlier indices
        outlier_indices = statistics_info["outliers_indices"].values[0]
        outlier_patient_ids = coefficient_variation_data["patient_id"].iloc[outlier_indices].values
        outlier_values = coefficient_variation_data["cv"].iloc[outlier_indices].values
        statistics_info["outlier_patient_ids"] = [outlier_patient_ids]

        sample_size_and_infos = pd.concat([sample_size_and_infos, statistics_info], ignore_index=True)
        
    #plt.xticks(sec_per_epochs_list, [f'{power_spectrum_length[0]} seconds']) # for violin plot
    #plt.xticks(sec_per_epochs_list, [f'{sec_per_epochs_list} seconds']) # for violin plot
    ax.set_xticks(sec_per_epochs_list)
    ax.set_xticklabels(sec_per_epochs_list)
                
    # plt.bar(range(len(coefficient_variation_data["cv"])), coefficient_variation_data["cv"], color='skyblue')
    plt.ylabel('Coefficient of Variation (%)')
    plt.xlabel('Length of Power Spectra (seconds)')
    plt.title(f'Coefficient of Variation of 4x {freq_band} power in the maximal {freq_band} {channel_group} channels')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    io_externalized.save_fig_png_and_svg(figure=fig,
                                         filename=f"Coefficient_of_Variation_4x_multiple_sec_spectra_externalized_BSSU_maximal_{freq_band}_{channel_group}_{filtered}",
                                         path=GROUP_FIGURES_PATH
    )

    return sample_size_and_infos






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

def calculate_z_score(measurements:list):
    """
    This function calculates the z-score of a list of measurements:
        z = (x - mean) / std
    
        A z-score describes how many standard deviations a measurement is from the population mean.

    Output:
        - z_score: list of z-scores
        - mean: mean of the measurements
        - std: standard deviation of the measurements

    """

    # check if all measurements are the same, in this case std becomes zero and division by zero is not possible
    if np.std(measurements) != 0:
        z_score = [(x - np.mean(measurements)) / np.std(measurements) for x in measurements]
    
    else: 
        z_score = [0] * len(measurements)

    return {
        "z_score": z_score,
        "mean": np.mean(measurements),
        "std": np.std(measurements)}



def tukey_mean_difference_plot(filtered:str, freq_band:str, channel_group:str, z_score:str):
    """
    For each patient hemisphere, one maximal power channel:
    Calculate the MEAN and DIFFERENCE TO MEAN of 5 x 20 sec power spectra within each patient hemisphere

    Input:
        - filtered: str "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
        - freq_band: str "beta", "high_beta", "low_beta"
        - channel_group: str "all", "ring", "segm_inter", "segm_intra", "segm"
        - z_score: str "yes", "no"

    """
    all_differences_to_group_mean = []

    fig = plt.figure(figsize=(10, 10), layout='tight')
    plt.subplot(1,1,1)

    # load the maximal power data for each sub, hem
    maximal_power_data = rank_power_2min(filtered=filtered, freq_band=freq_band, channel_group=channel_group)["maximal_power_data"]

    # merge together columns "subject", "hemisphere", "channel" to get unique patient ids
    maximal_power_data["patient_id"] = maximal_power_data["subject"] + "_" + maximal_power_data["hemisphere"] + "_" + maximal_power_data["channel"]
    patient_ids = maximal_power_data['patient_id'].unique()

    for patient_id in patient_ids:
        patient_data = maximal_power_data[maximal_power_data['patient_id'] == patient_id]
        
        measurements = []
        for i in range(1, 6): # 5 x 20 sec power averages
            column_name = f'power_spectrum_20sec_{i}_mean'
            measurements.append(patient_data[column_name].values[0])

        if z_score == "yes":
            z_score_data = calculate_z_score(measurements=measurements)
            measurements = z_score_data["z_score"] # list of z-scores for each measurement
        
        # calculate mean and difference to mean
        mean_of_measurements = np.mean(measurements)
        difference_to_mean = [x - mean_of_measurements for x in measurements]

        all_differences_to_group_mean.extend(difference_to_mean)        

        # plot the group mean (x-axis) and the difference to the group mean (y-axis) of the same patient with the same color
        plt.plot([mean_of_measurements]*5, difference_to_mean, 'o', linestyle='-', label=patient_id, alpha=0.5)

        # plt.plot([0, len(difference_to_mean)-1], [np.mean(difference_to_mean), np.mean(difference_to_mean)], linestyle='--', color='grey', alpha=0.5)
    
    # calculate the mean of all differences to group mean
    mean_all_differences = np.mean(all_differences_to_group_mean)
    ci_upper = mean_all_differences + 1.96 * np.std(all_differences_to_group_mean)
    ci_lower = mean_all_differences - 1.96 * np.std(all_differences_to_group_mean)

    sample_size = len(patient_ids)

    # Plot the Tukey mean-difference plot
    if z_score == "yes":
        plt.xlabel('Group Mean (z-score)')
        plt.ylabel('Difference to Group Mean (z-score)')
    else:
        plt.xlabel('Group Mean')
        plt.ylabel('Difference to Group Mean')
    
    plt.grid(True)
    plt.axhline(y=mean_all_differences, color='r', linestyle='--', linewidth=2)  # Add a horizontal line at the mean of all differences
    # Add mean text
    # plt.text(mean_all_differences + 0.1, 0, f"Mean of all differences: {mean_all_differences}", verticalalignment='right')

    plt.axhline(y=ci_upper, color='k', linestyle='--', linewidth=1)  # Add a horizontal line at the upper confidence interval
    plt.axhline(y=ci_lower, color='k', linestyle='--', linewidth=1)  # Add a horizontal line at the lower confidence interval

    plt.title(f'Tukey Mean-Difference Plot of {freq_band} power in the maximal {channel_group} channels')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    io_externalized.save_fig_png_and_svg(figure=fig, 
                                         filename=f"Tukey_Mean_Difference_Plot_20sec_externalized_BSSU_maximal_{freq_band}_{channel_group}_{filtered}_z_score_{z_score}",
                                         path=GROUP_FIGURES_PATH)

    return mean_all_differences, ci_upper, ci_lower, sample_size, patient_ids


######### SPATIAL DISTRIBUTION OF POWER #########

def rank_20sec_power_channels(filtered:str, freq_band:str, channel_group:str, rank_of_interest:int):
    """
    1) This function ranks the 20 sec power spectra for each subject and hemisphere within a selected group of channels
    2) This function selects the channels wiht a specific power rank in the 2min power spectrum
    
    """

    ranked_power_20sec = pd.DataFrame()
    only_one_2min_rank_data = pd.DataFrame()

    # load data
    all_power_data = rank_power_2min(filtered=filtered, freq_band=freq_band, channel_group=channel_group)["ranked_power_2min"]

    # rank average power of each 20 sec recording within the group of channels
    sub_list = all_power_data["subject"].unique() # list of subjects

    for sub in sub_list:

        sub_data = all_power_data[all_power_data["subject"] == sub]

        for hem in HEMISPHERES:

            hem_data = sub_data[sub_data["hemisphere"] == hem]
            hem_data_copy = hem_data.copy()

            # rank power of each 20 sec recording
            for i in range(1, 6):
                hem_data_copy[f"rank_20sec_{i}"] = hem_data_copy[f"power_spectrum_20sec_{i}_mean"].rank(ascending=False)
            
            ranked_power_20sec = pd.concat([ranked_power_20sec, hem_data_copy], ignore_index=True)

            # only keep maximal power channel
            rank_2min_power_channel = ranked_power_20sec[ranked_power_20sec["rank_2min"] == rank_of_interest]
            only_one_2min_rank_data = pd.concat([only_one_2min_rank_data, rank_2min_power_channel], ignore_index=True)
    

    return {
        "all_ranked_power_20sec": ranked_power_20sec,
        "only_one_2min_rank_data": only_one_2min_rank_data,}



def tukey_mean_difference_plot_20sec_ranks(filtered:str, freq_band:str, channel_group:str, z_score:str, rank_of_interest:int):
    """
    For each patient hemisphere, one power channel with the specific 2min power rank:
    Calculate the MEAN and DIFFERENCE TO MEAN of 5 x 20 sec power spectra within each patient hemisphere

    Input:
        - filtered: str "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
        - freq_band: str "beta", "high_beta", "low_beta"
        - channel_group: str "all", "ring", "segm_inter", "segm_intra", "segm"
        - z_score: str "yes", "no"
        - rank_of_interest: int 1-12 (depending on the number of channels in the group)

    """
    all_differences_to_group_mean = []

    fig = plt.figure(figsize=(10, 10), layout='tight')
    plt.subplot(1,1,1)

    # load the maximal power data for each sub, hem
    ranks_of_20sec_channels = rank_20sec_power_channels(filtered=filtered, 
                                                        freq_band=freq_band, 
                                                        channel_group=channel_group, 
                                                        rank_of_interest=rank_of_interest)["only_one_2min_rank_data"]

    # merge together columns "subject", "hemisphere", "channel" to get unique patient ids
    ranks_of_20sec_channels["patient_id"] = ranks_of_20sec_channels["subject"] + "_" + ranks_of_20sec_channels["hemisphere"] + "_" + ranks_of_20sec_channels["channel"]
    patient_ids = ranks_of_20sec_channels['patient_id'].unique()

    for patient_id in patient_ids:
        patient_data = ranks_of_20sec_channels[ranks_of_20sec_channels['patient_id'] == patient_id]
        
        measurements = []
        for i in range(1, 6): # 5 x 20 sec power averages
            column_name = f'rank_20sec_{i}'
            measurements.append(patient_data[column_name].values[0])

        if z_score == "yes":
            z_score_data = calculate_z_score(measurements=measurements)
            measurements = z_score_data["z_score"] # list of z-scores for each measurement
        
        # calculate mean and difference to mean
        mean_of_measurements = np.mean(measurements)
        difference_to_mean = [x - mean_of_measurements for x in measurements]

        all_differences_to_group_mean.extend(difference_to_mean) 

        # add a little bit of jitter, so that the points do not overlap
        jitter_amount = 0.15
        x_jittered = [mean_of_measurements + np.random.uniform(-jitter_amount, jitter_amount)] # mean here is often the same rank e.g. 1, so adding jitter helps to visualize the data      

        # plot the group mean (x-axis) and the difference to the group mean (y-axis) of the same patient with the same color
        plt.plot([x_jittered]*5, difference_to_mean, 'o', linestyle='-', label=patient_id, alpha=0.5)

        # plt.plot([0, len(difference_to_mean)-1], [np.mean(difference_to_mean), np.mean(difference_to_mean)], linestyle='--', color='grey', alpha=0.5)
    
    # calculate the mean of all differences to group mean
    mean_all_differences = np.mean(all_differences_to_group_mean)
    ci_upper = mean_all_differences + 1.96 * np.std(all_differences_to_group_mean)
    ci_lower = mean_all_differences - 1.96 * np.std(all_differences_to_group_mean)

    sample_size = len(patient_ids)

    # Plot the Tukey mean-difference plot
    if z_score == "yes":
        plt.xlabel(f'Group Mean (z-score), (rank within {channel_group} group)')
        plt.ylabel('Difference to Group Mean (z-score)')
    else:
        plt.xlabel(f'Group Mean (rank within {channel_group} group)')
        plt.ylabel(f'Difference to Group Mean')
    
    plt.grid(True)
    plt.axhline(y=mean_all_differences, color='r', linestyle='--', linewidth=2)  # Add a horizontal line at the mean of all differences
    # Add mean text
    # plt.text(mean_all_differences + 0.1, 0, f"Mean of all differences: {mean_all_differences}", verticalalignment='right')

    plt.axhline(y=ci_upper, color='k', linestyle='--', linewidth=1)  # Add a horizontal line at the upper confidence interval
    plt.axhline(y=ci_lower, color='k', linestyle='--', linewidth=1)  # Add a horizontal line at the lower confidence interval

    plt.title(f'Tukey Mean-Difference Plot of 20 sec {freq_band} power ranks within the {channel_group} group\n'
              f'of one channel per hemisphere with 2min {freq_band} rank: {rank_of_interest}')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    io_externalized.save_fig_png_and_svg(figure=fig, 
                                         filename=f"Tukey_Mean_Difference_Plot_20sec_externalized_BSSU_ranks_{freq_band}_{channel_group}_{filtered}_z_score_{z_score}_of_2min_channel_rank_{rank_of_interest}",
                                         path=GROUP_FIGURES_PATH)

    return mean_all_differences, ci_upper, ci_lower, sample_size, patient_ids





