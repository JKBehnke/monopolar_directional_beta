""" Load result files from results folder"""


import os
import pandas as pd
import pickle
import json

from ..utils import find_folders as find_folders


def load_PSDjson(sub: str, result: str, hemisphere: str, filter: str):
    """
    Reads result CSV file of the initial SPECTROGRAM method

    Input:
        - subject = str, e.g. "024"
        - result = str "PowerSpectrum", "PSDaverageFrequencyBands", "PeakParameters"
        - hemisphere = str, "Right" or "Left"
        - filter = str "band-pass" or "unfiltered"


    Returns:
        - data: loaded CSV file as a Dataframe

    """

    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="results", sub=sub)

    # create Filename out of input
    filename = ""
    filebase = ""

    if result == "PowerSpectrum":
        filebase = "SPECTROGRAMPSD"

    elif result == "PSDaverageFrequencyBands":
        filebase = f"SPECTROGRAMpsdAverageFrequencyBands"

    elif result == "PeakParameters":
        filebase = f"SPECTROGRAM_highestPEAK_FrequencyBands"

    hem = f"_{hemisphere}"
    filt = f"_{filter}.json"

    string_list = [filebase, hem, filt]
    filename = "".join(string_list)

    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )

    with open(os.path.join(local_results_path, filename)) as file:
        data = json.load(file)

    return data


def load_BIPChannelGroups_ALL(freqBand: str, normalization: str, signalFilter: str):
    """
    Loads pickle file from Group Results folder
    filename example: "BIPChannelGroups_ALL_{freqBand}_{normalization}_{signalFilter}.pickle"

    """
    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert freqBand in ["beta", "highBeta", "lowBeta"], f'Result ({freqBand}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    norm = f"_{normalization}"
    filt = f"_{signalFilter}.pickle"

    string_list = ["BIPChannelGroups_ALL_", freqBand, norm, filt]
    filename = "".join(string_list)
    print("pickle file loaded: ", filename, "\nloaded from: ", local_results_path)

    filepath = os.path.join(local_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_BIPChannelGroups_psdRanks_relToRank1(freqBand: str, normalization: str, signalFilter: str):
    """
    Loads pickle file from Group Results folder
    filename: "BIPChannelGroups_psdRanks_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle"

    Input:
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: list e.g. ["beta", "highBeta", "lowBeta"]


    """
    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert freqBand in ["beta", "highBeta", "lowBeta"], f'Result ({freqBand}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    norm = f"_{normalization}"
    filt = f"_{signalFilter}.pickle"

    string_list = ["BIPChannelGroups_psdRanks_relToRank1_", freqBand, norm, filt]
    filename = "".join(string_list)
    print("pickle file loaded: ", filename, "\nloaded from: ", local_results_path)

    filepath = os.path.join(local_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_PSDresultCSV(sub: str, psdMethod: str, normalization: str, hemisphere: str, filter: str):
    """
    Reads result CSV file of the initial SPECTROGRAM method

    Input:
        - subject = str, e.g. "024"
        - psdMethod = str "Welch" or "Spectrogram"
        - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - hemisphere = str, "Right" or "Left"
        - filter = str "band-pass" or "unfiltered"


    Returns:
        - data: loaded CSV file as a Dataframe

    """

    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="results", sub=sub)

    # create Filename out of input
    filename = ""

    if psdMethod == "Spectrogram":
        method = "SPECTROGRAM"

    elif psdMethod == "Welch":
        method = ""

    norm = normalization
    hem = f"_{hemisphere}"
    filt = f"_{filter}"

    string_list = [method, norm, hem, filt]
    filename = "".join(string_list)

    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )

    df = pd.read_csv(os.path.join(local_results_path, filename), sep=",")

    return df


def load_freqBandsCSV(sub: str, parameters: str, normalization: str, hemisphere: str, filter: str):
    """
    Reads result CSV file of the initial SPECTROGRAM method

    Input:
        - subject = str, e.g. "024"
        - parameters = str, "Peak" or "PSDaverage"
        - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - hemisphere = str, "Right" or "Left"
        - filter = str "band-pass" or "unfiltered"


    Returns:
        - data: loaded CSV file as a Dataframe

    """

    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="results", sub=sub)

    # create Filename out of input
    filename = ""

    if parameters == "Peak":
        values = "_highestPEAK_FrequencyBands_"

    elif parameters == "PSDaverage":
        values = "psdAverageFrequencyBands_"

    else:
        print("define parameters correctly: as 'Peak' or 'PSDaverage'.")

    norm = normalization
    hem = f"_{hemisphere}"
    filt = f"_{filter}"

    string_list = ["SPECTROGRAM", values, norm, hem, filt]
    filename = "".join(string_list)

    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )

    df = pd.read_csv(os.path.join(local_results_path, filename))

    return df


def load_BIPchannelGroupsPickle(result: str, channelGroup: list, normalization: str, filterSignal: str):
    """
    Reads pickle file written with functions in BIP_channelGroups.py -> filename e.g. BIPpsdAverage_Ring_{normalization}_{signalFilter}.pickle

    Input:
        - result = str "psdAverage", "peak"
        - channelGroup = list, ["Ring", "SegmInter", "SegmIntra"]
        - normalization =  str "rawPsd" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - filterSignal = str "band-pass" or "unfiltered"


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert result in ["psdAverage", "peak"], f'Result ({result}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")
    data = {}

    # create Filename out of input for each channel Group
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    for g in channelGroup:
        group = f"_{g}"

        string_list = ["BIP", result, group, norm, filt]
        filename = "".join(string_list)
        print("pickle file: ", filename, "\nloaded from: ", local_results_path)

        filepath = os.path.join(local_results_path, filename)

        # Error if filename doesn´t end with .mat
        # assert filename[-4:] == '.csv', (
        #     f'filename no .csv INCORRECT extension: {filename}'
        # )

        with open(filepath, "rb") as file:
            data[g] = pickle.load(file)

    return data


def load_BIPchannelGroup_sessionPickle(result: str, freqBand: str, normalization: str, filterSignal: str):
    """
    Reads pickle file written with function Rank_BIPRingSegmGroups() in BIPchannelGroups_ranks.py
    -> filename: BIPranksChannelGroup_session_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle

    Input:
        - result = str "psdAverage", "peak"
        - freqBand = str, e.g. "beta"
        - normalization =  str "rawPsd" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - filterSignal = str "band-pass" or "unfiltered"


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert result in ["psdAverage", "peak"], f'Result ({result}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    freq = f"_{freqBand}"
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["BIPranksChannelGroup_session_dict_", result, freq, norm, filt]
    filename = "".join(string_list)
    print("pickle file loaded: ", filename, "\nloaded from: ", local_results_path)

    filepath = os.path.join(local_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_BIPpermutationComparisonsPickle(result: str, freqBand: str, normalization: str, filterSignal: str):
    """
    Reads pickle file written with function Rank_BIPRingSegmGroups() in BIPchannelGroups_ranks.py

    Input:
        - result = str "psdAverage", "peak"
        - freqBand = str, e.g. "beta"
        - normalization =  str "rawPsd" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - filterSignal = str "band-pass" or "unfiltered"


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # Error check:
    # Error if sub str is not exactly 3 letters e.g. 024
    assert result in ["psdAverage", "peak"], f'Result ({result}) INCORRECT'

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    comparison = [
        "Postop_Postop",
        "Postop_Fu3m",
        "Postop_Fu12m",
        "Postop_Fu18m",
        "Fu3m_Postop",
        "Fu3m_Fu3m",
        "Fu3m_Fu12m",
        "Fu3m_Fu18m",
        "Fu12m_Postop",
        "Fu12m_Fu3m",
        "Fu12m_Fu12m",
        "Fu12m_Fu18m",
        "Fu18m_Postop",
        "Fu18m_Fu3m",
        "Fu18m_Fu12m",
        "Fu18m_Fu18m",
    ]

    res = f"_{result}"
    freq = f"_{freqBand}"
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    data = {}

    for c in comparison:
        string_list = ["BIPpermutationDF_", c, res, freq, norm, filt]
        filename = "".join(string_list)

        filepath = os.path.join(local_results_path, filename)

        with open(filepath, "rb") as file:
            data[c] = pickle.load(file)

        print("pickle file loaded: ", filename, "\nloaded from: ", local_results_path)

    return data


def load_BIPpermutation_ranks_result(
    data2permute: str,
    filterSignal: str,
    normalization: str,
    freqBand: str,
):
    """
    Reads pickle file written with function PermutationTest_BIPchannelGroups() in Permutation_rankings.py
        filename: "Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"

    Input:
        - data2permute: str e.g. "psdAverage",  "peak"
        - filterSignal: str e.g. "band-pass"
        - normalization: str e.g. "rawPsd"
        - freqBand: str e.g. "beta"


    Returns:
        - data: loaded pickle file as a Dataframe


    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_BestClinicalStimulation_excel():
    """
    Reads Excel file from the results folder: BestClinicalStimulation.xlsx
    loaded file = dictionary
        - all sheets are loaded as different keys

    Input:


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    data_path = find_folders.get_local_path(folder="data")

    filename = "BestClinicalStimulation.xlsx"
    filepath = os.path.join(data_path, filename)

    data = pd.read_excel(filepath, keep_default_na=True, sheet_name=None)  # all sheets are loaded
    print("Excel file loaded: ", filename, "\nloaded from: ", data_path)

    return data


def load_monoRef_weightedPsdCoordinateDistance_pickle(
    sub: str,
    hemisphere: str,
    freqBand: str,
    normalization: str,
    filterSignal: str,
):
    """
    Reads Pickle file from the subjects results folder:
        - sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle


    loaded file is a dictionary with keys:
        - f"{session}_bipolar_Dataframe": bipolar channels, averaged PSD in freq band, coordinates z- and xy-axis of mean point between two contacts
        - f"{session}_monopolar_Dataframe": 10 monopolar contacts, averaged monopolar PSD in freq band, rank of averagedPsd values



    Input:


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    sub_results_path = find_folders.get_local_path(folder="results", sub=sub)

    hem = f"_{hemisphere}"
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["sub", sub, hem, "_monoRef_weightedPsdByCoordinateDistance_", freqBand, norm, filt]
    filename = "".join(string_list)

    filepath = os.path.join(sub_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", sub_results_path)

    return data


def load_monoRef_only_segmental_weight_psd_by_distance(
    sub: str,
    hemisphere: str,
    freqBand: str,
    normalization: str,
    filterSignal: str,
):
    """
    Reads Pickle file from the subjects results folder:
        - sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle


    loaded file is a dictionary with keys:
        - f"{session}_bipolar_Dataframe": bipolar channels, averaged PSD in freq band, coordinates z- and xy-axis of mean point between two contacts
        - f"{session}_monopolar_Dataframe": 10 monopolar contacts, averaged monopolar PSD in freq band, rank of averagedPsd values

    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    sub_results_path = find_folders.get_local_path(folder="results", sub=sub)

    filename = f"sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

    filepath = os.path.join(sub_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", sub_results_path)

    return data


def load_Group_monoRef_only_segmental_weight_psd_by_distance(
    freqBand: str,
    normalization: str,
    filterSignal: str,
):
    """
    Reads Pickle file from the results folder:
        - "group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{signalFilter}.pickle"


    loaded file is a dictionary with keys:
        - f"{session}_bipolar_Dataframe": bipolar channels, averaged PSD in freq band, coordinates z- and xy-axis of mean point between two contacts
        - f"{session}_monopolar_Dataframe": 10 monopolar contacts, averaged monopolar PSD in freq band, rank of averagedPsd values

    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    filename = f"group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{filterSignal}.pickle"

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", results_path)

    return data


def load_monopol_rel_psd_from0To8_pickle(
    freqBand: str,
    normalization: str,
    filterSignal: str,
):
    """
    Reads Pickle file from the subjects results folder:
        - "monopol_rel_psd_from0To8_{freqBand}_{normalization}_{signalFilter}.pickle"


    Input:
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["monopol_rel_psd_from0To8_", freqBand, norm, filt]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", results_path)

    return data


def load_GroupMonoRef_weightedPsdCoordinateDistance_pickle(
    freqBand: str,
    normalization: str,
    filterSignal: str,
):
    """
    Reads Pickle file from the subjects results folder:
        - "GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle"


    Input:
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_", freqBand, norm, filt]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", results_path)

    return data


def load_monoRef_JLB_pickle(
    sub: str,
    hemisphere: str,
    normalization: str,
    filterSignal: str,
):
    """
    Reads Pickle file from the subjects results folder:
        - sub{incl_sub}_{hemisphere}_MonoRef_JLB_result_{normalization}_band-pass.pickle"


    loaded file is a dictionary with keys:
        - "BIP_psdAverage"
        - "BIP_directionalPercentage"
        - "monopolar_psdAverage"
        - "monopolar_psdRank"


    Input:


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder of a subject
    sub_results_path = find_folders.get_local_path(folder="results", sub=sub)

    hem = f"_{hemisphere}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["sub", sub, hem, "_MonoRef_JLB_result_", normalization, filt]
    filename = "".join(string_list)

    filepath = os.path.join(sub_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", sub_results_path)

    return data


def load_ClinicalActiveVsInactive(freqBand: str, attribute: str, singleContacts_or_average: str):
    """
    Loading Dataframe with clinically active and inactive contacts

    file: "ClinicalActiveVsNonactiveContacts_{attribute}_{freqBand}_{singleContacts_or_average}.pickle

    Input:
        - freqBand: str e.g. "beta"
        - attribute: str e.g. "rank", "relativeToRank1_psd"
        - singleContacts_or_average: str e.g. "singleContacts", "averageContacts"

    """

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    r_or_psd = f"_{attribute}"
    freq = f"_{freqBand}"
    single_or_average = f"_{singleContacts_or_average}.pickle"

    if attribute == "rank":
        r_or_psd = ""

    string_list = ["ClinicalActiveVsNonactiveContacts", r_or_psd, freq, single_or_average]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", results_path)

    return data


def load_SSD_results_pickle(f_band: str):
    """
    Reads Pickle file from the group results folder:
        - "SSD_results_Dataframe_{f_band}.pickle"


    loaded file is a Dataframe with columns:

        - subject
        - hemisphere
        - session
        - recording_group (e.g. RingR)
        - bipolarChannel
        - ssd_filtered_timedomain (array)
        - ssd_pattern (weights of a channel contributing to the first component)
        - ssd_eigvals


    Input:
        - f_band = str, e.g. "beta", "highBeta", "lowBeta"


    Returns:
        - data: loaded pickle file as a Dataframe

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    freq = f"_{f_band}.pickle"

    string_list = ["SSD_results_Dataframe", freq]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ", filename, "\nloaded from: ", results_path)

    return data


def load_fooof_json(subject: str, fooof_version: str):
    """
    Load the file: "fooof_model_sub{subject}.json"
    from each subject result folder
    - fooof_version: "v1" or "v2"

    """

    # find the path to the results folder
    results_path_sub = find_folders.get_local_path(folder="results", sub=subject)

    # create filename
    filename = f"fooof_model_sub{subject}_{fooof_version}.json"

    # load the json file
    with open(os.path.join(results_path_sub, filename)) as file:
        json_data = json.load(file)

    fooof_result_df = pd.DataFrame(json_data)

    return fooof_result_df


def load_group_fooof_result(fooof_version: str):
    """
    Load the file: "fooof_model_group_data.json"
    from the group result folder

    - fooof_version: "v1" or "v2"

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_model_group_data_{fooof_version}.json"

    # load the json file
    with open(os.path.join(results_path, filename)) as file:
        json_data = json.load(file)

    fooof_result_df = pd.DataFrame(json_data)

    return fooof_result_df


def load_fooof_peaks_per_session():
    """
    Load the file: "fooof_peaks_per_session.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_peaks_per_session.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_rank_beta_peak_power():
    """
    Load the file: "fooof_rank_beta_power_dataframe.pickle"
    from the group result folder
    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_rank_beta_power_dataframe.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_beta_ranks(fooof_spectrum: str, fooof_version: str, all_or_one_chan: str, all_or_one_longterm_ses: str):
    """
    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - fooof_version: str "v1" or "v2"

        - all_or_one_chan: str "highest_beta" or "beta_ranks_all" (12 LFP channels, with ranks) or "beta_all" (all 15 channels, no ranks)


        - all_or_one_longterm_ses: str "all_sessions", "one_longterm_session"

        if all_or_one_longterm_ses == "one_longterm_session" -> The loaded dataframe only consists of one longterm session (fu18m or fu24m).
        So if a STN was recorded at the fu18m and fu24m, the fu24m session was deleted


    Load the file: f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"
    from the group result folder

    """
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}_{fooof_version}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    ############## only keep one longterm session ##############
    if all_or_one_longterm_ses == "one_longterm_session":
        # per stn and session: if fu18m and fu24m exist -> delete fu24m
        stn_unique = list(data.subject_hemisphere.unique())
        longterm_sessions = ["fu18m", "fu24m"]
        dataframe_longterm = pd.DataFrame()

        for stn in stn_unique:
            stn_data = data.loc[data.subject_hemisphere == stn]

            # check if both fu18m and fu24m exist, if yes: delete fu24m
            if all(l_ses in stn_data["session"].values for l_ses in longterm_sessions):
                # exclude all rows including "fu24m"
                stn_data = stn_data[stn_data["session"] != "fu24m"]

            # append stn dataframe to the longterm dataframe
            dataframe_longterm = pd.concat([dataframe_longterm, stn_data])

        # replace all "fu18m" and "fu24m" by "longterm"
        dataframe_longterm["session"] = dataframe_longterm["session"].replace(longterm_sessions, "fu18or24m")

    ############## or keep all sessions ##############
    elif all_or_one_longterm_ses == "all_sessions":
        dataframe_longterm = data

    return dataframe_longterm


def load_power_spectra_session_comparison(incl_channels: str, signalFilter: str, normalization: str):
    """
    Load the file: f"power_spectra_{signalFilter}_{incl_channels}_{normalization}_session_comparisons.pickle"
    from the group result folder

    Input:
        - incl_channels: str, e.g. "SegmInter", "SegmIntra", "Ring"
        - signalFilter: str, e.g. "band-pass" or "unfiltered"
        - normalization: str, e.g. "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"

    Return:
        - dictionary with keys:
        ['postop_fu3m_df', 'postop_fu12m_df', 'postop_fu18m_df', 'fu3m_fu12m_df', 'fu3m_fu18m_df', 'fu12m_fu18m_df']

        - each key value is a dataframe with the power spectra of STNs with recordings at both sessions

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"power_spectra_{signalFilter}_{incl_channels}_{normalization}_session_comparisons.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_monopolar_weighted_psd(
    fooof_spectrum: str, fooof_version: str, segmental: str, similarity_calculation: str
):
    """
    The loaded dataframe only consists of one longterm session (fu18m or fu24m).
    So if a STN was recorded at the fu18m and fu24m, the fu24m session was deleted

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - fooof_version: "v1", "v2"
        - segmental: "yes"              -> only using segmental channels to weight monopolar psd

        - similarity_calculation: "inverse_distance", "exp_neg_distance"



    Load the file: f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    if segmental == "yes":
        bipolar_chans = "only_segmental_"

    else:
        bipolar_chans = "segments_and_rings_"

    # create filename
    filename = f"fooof_monoRef_{bipolar_chans}weight_beta_psd_by_{similarity_calculation}_{fooof_spectrum}_{fooof_version}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_monopolar_JLB():
    """
    The loaded dataframe consists of monopolar beta estimates with ranks
    for all segmented contacts, for all STNs at each session ("postop", "fu3m", "fu12m", "fu18or24m")

    The pickle file: "MonoRef_JLB_fooof_beta.pickle"
    in the results path /Users/jenniferbehnke/Dropbox/work/ResearchProjects/BetaSenSightLongterm/results
    is written by the function MonoRefJLB.fooof_monoRef_JLB()


    """
    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = "MonoRef_JLB_fooof_beta.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_permutation_bip_beta_ranks():
    """
    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - segmental: "yes"              -> only using segmental channels to weight monopolar psd



    Load the file: f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"permutation_beta_ranks_fooof_spectra.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_lme_model_result(highest_beta_session: str, data_to_fit: str, incl_sessions: list, shape_of_model: str):
    """
    Input:

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency"

        - incl_sessions: [0,3] or [3,12,18] o [0,3,12,18]

        - shape_of_model: e.g. "straight", "curved", "asymptotic"


    Load the file: f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = (
        f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle"
    )

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_fooof_monoRef_all_contacts_weight_beta(similarity_calculation: str):
    """
    Input:
        - similarity_calculation: "inverse_distance", "exp_neg_distance"

    Load the file: fooof_monoRef_all_contacts_weight_beta_psd_by_distance.pickle
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_preprocessing_files(table: str):
    """
    Input:
        - table: "movement_artifact_coord", "cleaned_power_spectra"

    Load the file: movement_artifacts_from_raw_time_series_band-pass.pickle  # always band-pass because filtered signal is easier to find movement artifacts
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    if table == "movement_artifact_coord":
        filename = f"movement_artifacts_from_raw_time_series_band-pass.pickle"

    elif table == "cleaned_power_spectra":
        filename = f"clean_power_spectra.json"

    filepath = os.path.join(results_path, filename)

    # load the file
    if table == "movement_artifact_coord":
        with open(filepath, "rb") as file:
            data = pickle.load(file)

    elif table == "cleaned_power_spectra":
        with open(filepath) as file:
            data = json.load(file)
            data = pd.DataFrame(data)

    return data


def load_pickle_group_result(filename: str, fooof_version: str):
    """
    Loading pickle file. Filename must be in:
    allowed_filenames:
        "fooof_rank_beta_power_dataframe",
        "fooof_peaks_per_session",
        "MonoRef_JLB_fooof_beta",
        "permutation_beta_ranks_fooof_spectra",
        "best_2_contacts_from_directional_bssu",
        "fooof_group_data_percept" -> {fooof_version},
        "permutation_beta_ranks_fooof_spectra"
        "fooof_monoRef_all_contacts_weight_beta_psd_by_inverse_distance",
        "fooof_detec_beta_all_directional_ranks",
        "fooof_detec_beta_levels_and_directions_ranks",
        "fooof_externalized_BSSU_monoRef_only_segmental_weight_beta_psd_by_inverse_distance"
        "fooof_externalized_BSSU_euclidean_weighted_power_spectra_only_segmental_inverse_distance",
        "MonoRef_JLB_fooof_externalized_BSSU_beta",
        "best_2_contacts_from_directional_externalized_bssu"
        "fooof_detec_externalized_bssu_beta_all_directional_ranks"
        "fooof_detec_externalized_bssu_beta_levels_and_directions_ranks"
        "notch_and_band_pass_filtered_externalized_BSSU_monoRef_only_segmental_weight_beta_psd_by_inverse_distance_v2" # externalized bssu euclidean method NO FOOOOF
        "notch_and_band_pass_filtered_externalized_BSSU_euclidean_weighted_power_spectra_only_segmental_inverse_distance_v2" # only weighted power spectra
        "MonoRef_JLB_notch_and_band_pass_filtered_externalized_BSSU_beta_v2"
        "MonoRef_JLB_notch_and_band_pass_filtered_externalized_BSSU_weighted_power_spectra_v2"
        "detec_strelow_notch_and_band_pass_filtered_externalized_BSSU_weighted_power_spectra_v2"
        "notch_and_band_pass_filtered_detec_externalized_bssu_beta_all_directional_ranks_v2"
        "notch_and_band_pass_filtered_detec_externalized_bssu_beta_levels_and_directions_ranks_v2"


    The pickle file: "{filename}.pickle"
    in the results path /Users/jenniferbehnke/Dropbox/work/ResearchProjects/BetaSenSightLongterm/results


    """

    filenames_with_fooof_version = [
        "fooof_group_data_percept",
        "MonoRef_JLB_fooof_beta",
        "permutation_beta_ranks_fooof_spectra",
        "fooof_monoRef_all_contacts_weight_beta_psd_by_inverse_distance",
        "fooof_detec_beta_all_directional_ranks",
        "fooof_detec_beta_levels_and_directions_ranks",
        "fooof_externalized_BSSU_monoRef_only_segmental_weight_beta_psd_by_inverse_distance",
        "fooof_externalized_BSSU_euclidean_weighted_power_spectra_only_segmental_inverse_distance",
        "MonoRef_JLB_fooof_externalized_BSSU_beta",
        "best_2_contacts_from_directional_externalized_bssu",
        "fooof_detec_externalized_bssu_beta_all_directional_ranks",
        "fooof_detec_externalized_bssu_beta_levels_and_directions_ranks",
    ]

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    if filename in filenames_with_fooof_version:
        filename = f"{filename}_{fooof_version}.pickle"

    else:  # create filename without fooof_version
        filename = f"{filename}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


# def load_MonoRef_GroupCSV(normalization: str, hemisphere: str):

#     """
#     Reads monopolar reference result of all subjects in one CSV from Johannes' method

#     Input:
#         - subject = str, e.g. "024"
#         - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
#         - hemisphere = str, "Right" or "Left"


#     Returns: loading csv files into a dictionary
#         - psd average (columns: session, frequency_band, channel, averagedPSD)
#         - percentage of psd per directions (columns: session, frequency_band, direction, percentagePSD_perDirection)
#         - monopolar Reference Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)
#         - monopolar Ranks Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)

#     """

#     # find the path to the results folder of a subject
#     project_path = os.getcwd()

#     while project_path[-21:] != 'Longterm_beta_project':
#         project_path = os.path.dirname(project_path)

#     results_path = os.path.join(project_path, 'results')
#     sys.path.append(results_path)

#     # change directory to code path
#     os.chdir(results_path)
#     local_results_path = find_folder.get_local_path(folder="results", sub=sub)


#     psdAverageDataframe = pd.read_csv(os.path.join(local_results_path, f"averagedPSD_{normalization}_{hemisphere}"))
#     psdPercentagePerDirection = pd.read_csv(os.path.join(local_results_path, f"percentagePsdDirection_{normalization}_{hemisphere}"))
#     monopolRefDF = pd.read_csv(os.path.join(local_results_path, f"monopolarReference_{normalization}_{hemisphere}"))
#     monopolRefDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     monopolRankDF =  pd.read_csv(os.path.join(local_results_path, f"monopolarRanks_{normalization}_{hemisphere}"))
#     monopolRankDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     # FirstRankChannel_PSD_DF = pd.read_csv(os.path.join(local_results_path,f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_beta = pd.read_csv(os.path.join(local_results_path,f"postopBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_beta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_highBeta{normalization}_{hemisphere}"), sep=",")


#     return {
#         "psdAverageDataframe": psdAverageDataframe,
#         "psdPercentagePerDirection": psdPercentagePerDirection,
#         "monopolRefDF": monopolRefDF,
#         "monopolRankDF": monopolRankDF,
#         # "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
#         # "PostopBaseline_beta": postopBaseline_beta,
#         # "PostopBaseline_lowBeta": postopBaseline_lowBeta,
#         # "PostopBaseline_highBeta": postopBaseline_highBeta,
#         # "Fu3mBaseline_beta": fu3mBaseline_beta,
#         # "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
#         # "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,

#         }


# def load_MonoRef_JLBresultCSV(sub: str, normalization: str, hemisphere: str):

#     """
#     Reads monopolar reference result from Johannes' method CSV file

#     Input:
#         - subject = str, e.g. "024"
#         - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
#         - hemisphere = str, "Right" or "Left"


#     Returns: loading csv files into a dictionary
#         - psd average (columns: session, frequency_band, channel, averagedPSD)
#         - percentage of psd per directions (columns: session, frequency_band, direction, percentagePSD_perDirection)
#         - monopolar Reference Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)
#         - monopolar Ranks Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)

#     """


#     # Error check:
#     # Error if sub str is not exactly 3 letters e.g. 024
#     assert len(sub) == 3, f'Subject string ({sub}) INCORRECT'


#     # find the path to the results folder of a subject
#     local_results_path = find_folder.get_local_path(folder="results", sub=sub)


#     psdAverageDataframe = pd.read_csv(os.path.join(local_results_path, f"averagedPSD_{normalization}_{hemisphere}"))
#     psdPercentagePerDirection = pd.read_csv(os.path.join(local_results_path, f"percentagePsdDirection_{normalization}_{hemisphere}"))
#     monopolRefDF = pd.read_csv(os.path.join(local_results_path, f"monopolarReference_{normalization}_{hemisphere}"))
#     monopolRefDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     monopolRankDF =  pd.read_csv(os.path.join(local_results_path, f"monopolarRanks_{normalization}_{hemisphere}"))
#     monopolRankDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     # FirstRankChannel_PSD_DF = pd.read_csv(os.path.join(local_results_path,f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_beta = pd.read_csv(os.path.join(local_results_path,f"postopBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_beta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_highBeta{normalization}_{hemisphere}"), sep=",")


#     return {
#         "psdAverageDataframe": psdAverageDataframe,
#         "psdPercentagePerDirection": psdPercentagePerDirection,
#         "monopolRefDF": monopolRefDF,
#         "monopolRankDF": monopolRankDF,
#         # "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
#         # "PostopBaseline_beta": postopBaseline_beta,
#         # "PostopBaseline_lowBeta": postopBaseline_lowBeta,
#         # "PostopBaseline_highBeta": postopBaseline_highBeta,
#         # "Fu3mBaseline_beta": fu3mBaseline_beta,
#         # "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
#         # "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,

#         }
