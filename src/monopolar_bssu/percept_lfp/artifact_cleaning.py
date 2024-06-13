""" artifact cleaning before computing power spectra """


import os

import matplotlib.pyplot as plt
import mne
# from mne.preprocessing import ICA, create_ecg_epochs
import numpy as np
import pandas as pd
import scipy
from cycler import cycler
from scipy.signal import hann
import pickle
from sklearn.decomposition import FastICA


# PyPerceive Imports
# import py_perceive
from PerceiveImport.classes import main_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults
from ..utils import sub_session_dict as sub_session_dict
from ..tfr import bssu_from_source_JSON as bssu_json


HEMISPHERES = ["Right", "Left"]
CHANNEL_GROUPS = ["RingL", "SegmIntraL", "SegmInterL", "RingR", "SegmIntraR", "SegmInterR"]




def load_mne_object_pyPerceive(sub: str, session: str, channel_group: str):
    """
    
    """

    mainclass_sub = main_class.PerceiveData(
        sub = sub, 
        incl_modalities= ["survey"],
        incl_session = [session],
        incl_condition = ["m0s0"],
        incl_task = ["rest"],
        incl_contact=[channel_group]
        )

    temp_data = getattr(mainclass_sub.survey, session) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
    temp_data = getattr(temp_data, "m0s0") # gets attribute e.g. "m0s0"
    temp_data = getattr(temp_data.rest, channel_group)
    temp_data = temp_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"

    return temp_data


def plot_ieeg_data(sub: str, session: str, channel_group: str, ieeg_data, fig_title: str):
    """
    Function to plot the iEEG data. This function can be used when you have already extracted the data into a 2D array.

    Input:
        - ieeg_data: np.array -> 2D array shape: (n_channels, n_samples)
    """

    plt.style.use('seaborn-whitegrid')  
    figures_path = findfolders.get_local_path(folder="figures", sub=sub)

    # if channel_group ends with R, it is Right hemisphere
    if channel_group.endswith("R"):
        hem = "Right"
    
    elif channel_group.endswith("L"):
        hem = "Left"

    # depending on which channel group, choose figure size and channels
    if channel_group == "RingR" or channel_group == "RingL":
        channels = ['03', '13', '02', '12', '01', '23']
        fig_size = (30, 10)
        group_name = "ring"
    
    elif channel_group == "SegmIntraR" or channel_group == "SegmIntraL":
        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
        fig_size = (30, 10)
        group_name = "segm_intra"
    
    elif channel_group == "SegmInterR" or channel_group == "SegmInterL":
        channels = ['1A2A', '1B2B', '1C2C']
        fig_size = (30, 5)
        group_name = "segm_inter"
    

    fig, axes = plt.subplots(len(channels), 1, figsize=fig_size) # subplot(rows, columns, panel number), figsize(width,height)

    for i, ch in enumerate(channels):
     
        signal = ieeg_data[i, :]

        #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

        axes[i].set_title(f"{session}, {channel_group}, {channels[i]}", fontsize=15) 
        axes[i].plot(signal, label=f"{channels[i]}_m0s0", color="k", linewidth=0.3)  

    for ax in axes:
        ax.set_xlabel("timestamp", fontsize=15)
        ax.set_ylabel("amplitude", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    for ax in axes.flat[:-1]:
        ax.set(xlabel='')

    fig.suptitle(f"{fig_title} sub-{sub}, {hem} hemisphere", ha="center", fontsize=15)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show(block=False)

    # save figure
    fig.savefig(os.path.join(figures_path, f"{fig_title}_sub-{sub}_{hem}_{session}_m0s0_{group_name}.svg"), bbox_inches="tight", format="svg")
    fig.savefig(os.path.join(figures_path, f"{fig_title}_sub-{sub}_{hem}_{session}_m0s0_{group_name}.png"), bbox_inches="tight")

    return {
        "data": ieeg_data, 
        "channels": channels,
        "hemisphere": hem
        }


def plot_raw_time_series_before_MNE(sub: str, session: str, channel_group: str):
    """
    Function to plot raw time series of all channels for each subject, session, condition and channel group.
    This function can be used before using MNE to extract the data
    
    """
    plt.style.use('seaborn-whitegrid')  
    figures_path = findfolders.get_local_path(folder="figures", sub=sub)

    # if channel_group ends with R, it is Right hemisphere
    if channel_group.endswith("R"):
        hem = "Right"
    
    elif channel_group.endswith("L"):
        hem = "Left"

    # depending on which channel group, choose figure size and channels
    if channel_group == "RingR" or channel_group == "RingL":
        channels = ['03', '13', '02', '12', '01', '23']
        fig_size = (30, 10)
        group_name = "ring"
    
    elif channel_group == "SegmIntraR" or channel_group == "SegmIntraL":
        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
        fig_size = (30, 10)
        group_name = "segm_intra"
    
    elif channel_group == "SegmInterR" or channel_group == "SegmInterL":
        channels = ['1A2A', '1B2B', '1C2C']
        fig_size = (30, 5)
        group_name = "segm_inter"
    
    temp_data = load_mne_object_pyPerceive(sub, session, channel_group)

    # get new channel names
    ch_names = temp_data.info.ch_names


    #################### PICK CHANNELS ####################
    include_channelList = [] # this will be a list with all channel names selected

    for names in ch_names:
        
        # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
        for picked in channels:
            if picked in names:
                include_channelList.append(names)

        
    # Error Checking: 
    if len(include_channelList) == 0:
        print("No channels found for the given channel group.")

    # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
    ch_names_indices = mne.pick_channels(ch_names, include=include_channelList)

    fig, axes = plt.subplots(len(channels), 1, figsize=fig_size) # subplot(rows, columns, panel number), figsize(width,height)

    for i, ch in enumerate(ch_names):
        
        # only get picked channels
        if i not in ch_names_indices:
            continue

        signal = temp_data.get_data()[i, :]

        #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

        axes[i].set_title(f"{session}, {channel_group}, {channels[i]}", fontsize=15) 
        axes[i].plot(signal, label=f"{channels[i]}_m0s0", color="k", linewidth=0.3)  

    for ax in axes:
        ax.set_xlabel("timestamp", fontsize=15)
        ax.set_ylabel("amplitude", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    for ax in axes.flat[:-1]:
        ax.set(xlabel='')

    fig.suptitle(f"raw time series sub-{sub}, {hem} hemisphere", ha="center", fontsize=15)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show(block=False)

    # save figure
    fig.savefig(os.path.join(figures_path, f"raw_time_series_sub-{sub}_{hem}_{session}_m0s0_{group_name}.svg"), bbox_inches="tight", format="svg")
    fig.savefig(os.path.join(figures_path, f"raw_time_series_sub-{sub}_{hem}_{session}_m0s0_{group_name}.png"), bbox_inches="tight")

    return {
        "data": temp_data.get_data(), 
        "channels": channels,
        "hemisphere": hem
        }


def fit_ica_on_channel_group(data):
    """
    Function to fit ICA on the raw data of one channel group.

    Input: mne_object.get_data() -> 2D array shape: (n_channels, n_samples)

    """

    # first transpose the data to fit the input of FastICA
    data_to_fit = data.T

    # initialize ICA
    n_components = data.shape[0] # number of channels (only so many components can be estimated)

    ica = FastICA(n_components=n_components, random_state=97) # why 97? 

    # fit ICA model (fit) and transform the data into the independent components (transform)
    ica_components = ica.fit_transform(data_to_fit)

    return ica_components, n_components, ica

def plot_ica_components(data):
    """
    Function to plot the independent components of the ICA.

    Input: mne_object.get_data() -> 2D array shape: (n_channels, n_samples)
    
    """
    plt.style.use('seaborn-whitegrid')  

    # fit the ICA model and get components
    ica_components, n_components, ica = fit_ica_on_channel_group(data)

    if n_components == 6:
        fig_size = (30, 10)
    
    elif n_components == 3:
        fig_size = (30, 5)

    time_points = np.arange(data.shape[1])  # Assuming your data is sampled uniformly
    fig, axes = plt.subplots(n_components, 1, figsize=fig_size, sharex=True, sharey=True)

    for i in range(n_components):
        axes[i].plot(time_points, ica_components[:, i], label=f'Component {i + 1}', linewidth=0.3)
        axes[i].set_title(f"Component {i + 1}", fontsize=15)
        axes[i].set_ylabel(f'Amplitude')

    axes[n_components - 1].set_xlabel('Time points')
    plt.tight_layout()
    plt.show()

    return {
        "fig_ecg_components": fig,
        "ica_components": ica_components,
        "n_components": n_components,
        "ica": ica
    }

def remove_ecg_artifact_from_signals(data: np.array, ecg_component_index: int, ica_components: np.array, ica):
    """
    Function to remove ECG artifact from the signals.

    Input:
        - ecg_component: int -> number of the independent component that represents the ECG artifact
        - data: np.array -> 2D array shape: (n_channels, n_samples)
        - ica_components: np.array -> 2D array shape: (n_samples, n_components)
        - ica: FastICA object
    
    Output:
        - cleaned_ieeg_data: np.array -> 2D array shape: (n_channels, n_samples)
    """

    # get the ecg component
    ecg_component = ica_components[:, ecg_component_index]

    # remove the ecg component from the data
    cleaned_ieeg_data = data.T - np.outer(ecg_component, ica.mixing_[:, ecg_component_index])  # both shape (times x channels)

    # explained: ica.mixing_[:, ecg_component_index]: Retrieves the mixing coefficients corresponding to the ECG component. The ica.mixing_ matrix describes how the independent components are combined to form the observed data.
    # np.outer(ecg_component, ica.mixing_[:, ecg_component_index]): This is the outer product of the ECG component and the mixing coefficients. This gives the contribution of the ECG component to each channel at each time point.
    # This results in a matrix where each element is the product of the corresponding elements of ecg_component and the mixing coefficients.
    # - n.outer() subtracts the ECGF artifact contribution from the transposed data

    # Now, cleaned_ieeg_data contains the EEG data with the ECG artifact removed
    # Note: Depending on the scaling of the components, you might need to adjust the amplitude of the subtracted signal to achieve the desired artifact removal. 
    # You may also need to experiment with the sign of the subtracted signal. 
    # If the artifact is not completely removed, you can try multiplying the ECG component by a scaling factor before subtraction.

    return cleaned_ieeg_data.T

def get_input_y_n(message: str) -> str:
    """Get `y` or `n` user input."""
    while True:
        user_input = input(f"{message} (y/n)? ")
        if user_input.lower() in ["y", "n"]:
            break
        print(
            f"Input must be `y` or `n`. Got: {user_input}."
            " Please provide a valid input."
        )
    return user_input

def get_input_ecg_component(message: str) -> str:
    """Get integer of ecg component user input."""
    while True:
        user_input = input(f"{message} (integer)? 0 if none.")
        if int(user_input) in {0, 1, 2, 3, 4, 5, 6}:
            break
        print(
            f"Input must be an integer from 0 to 6. Got: {user_input}."
            " Please provide a valid input."
        )
    return int(user_input)

def get_ecg_artifact_excel():
    """
    Load the excel file "ecg_artifacts.xlsx" from the data folder > ecg_cleaning folder
    """

    # find the path to the results folder
    path = findfolders.get_local_path(folder="data")

    # go into ecg cleaning folder
    path = os.path.join(path, "ecg_cleaning")

    # load the file
    filename = "ecg_artifacts.xlsx"
    filepath = os.path.join(path, filename)

    # load the file
    ecg_artifact_df = pd.read_excel(filepath, keep_default_na=True, sheet_name="ecg_artifacts")
    print("Excel file loaded: ", filename, "\nloaded from: ", path)

    return ecg_artifact_df

def save_updated_excel(updated_df):
    """
    Save the updated excel file with the new row added.

    Input:
        - ecg_artifact_df: pd.DataFrame -> dataframe with the ecg artifacts
        - new_row: dict -> {"subject": str, "session": str, "channel_group": str, "ecg_component": int, "cleaned": str, "comment": str}
    """

    # find the path to the results folder
    path = findfolders.get_local_path(folder="data")

    # go into ecg cleaning folder
    path = os.path.join(path, "ecg_cleaning")

    # load the file
    filename = "ecg_artifacts.xlsx"
    filepath = os.path.join(path, filename)

    updated_df.to_excel(filepath, index=False, sheet_name="ecg_artifacts")
    print("Excel file updated: ", filename, "\nloaded from: ", path)

    return updated_df





############################ SCRIPT TO CLEAN ALL ECG ARTIFACTS FROM THE DATA ############################


def ecg_cleaning(sub: str):
    """
    Function to clean the ECG artifacts from the iEEG data of a subject.

    This function will iterate over all sessions and channel groups of a subject and clean the ECG artifacts from the iEEG data.
    It will check which session exist for a subject
    It will also check if the subject has a perceive error in any of the sessions. In case of a perceive error, the data will be loaded from the JSON file.
    Make sure to have the Report json in the data folder > source_json

    - Step 1: load and plot the raw time series
    - Step 2: fit ICA on the raw data and plot the components
    - Step 3: remove the ECG artifact from the signals
    - Step 4: plot the cleaned signals
    - Step 5: save the results  
        in an Excel file ("ecg_artifacts.xlsx") in the data folder > ecg_cleaning folder
        and the cleaned signals in a pickle file ("cleaned_time_series.pickle") in the subjects results folder

    
    """

    # load the excel file with the ecg artifacts
    ecg_artifact_df = get_ecg_artifact_excel()

    # get sessions of the subject
    sessions = sub_session_dict.get_sessions(sub) # list of existing sessions for the subject

    figures_path = findfolders.get_local_path(folder="figures", sub=sub)
    results_path = findfolders.get_local_path(folder="results", sub=sub)

    # dictionary to store the results
    clean_time_series_dict = {}
    

    # check if subject has perceive error
    perceive_error = sub_session_dict.check_if_perceive_error(sub)
    if perceive_error != "No":
        print(f"Perceive error for {sub} in sessions: {perceive_error}.") # alternative code will be used

    # iterate over sessions
    for ses in sessions:

        if ses in perceive_error:
            data_from_json = bssu_json.load_json_data_if_perceive_error(sub=sub, session=ses, condition="m0s0")

        # iterate over channel groups
        for group in CHANNEL_GROUPS:

            # Step 1: load and plot the raw time series ############################################
            if ses in perceive_error:
                plot_raw = plot_ieeg_data(sub, ses, group, data_from_json[group], fig_title="raw_time_series")

            else:
                plot_raw = plot_raw_time_series_before_MNE(sub, ses, group)

            # extract the data and channels and hemisphere
            data = plot_raw["data"] # in 2D array shape: (n_channels, n_samples)
            channels = plot_raw["channels"]
            hem = plot_raw["hemisphere"]

            # check if data contains ECG artifact
            input_y_or_n_artifact = get_input_y_n("ECG artifacts found") # interrups run and asks for input
            
            if input_y_or_n_artifact == "n":

                # save new row in the excel file
                new_row = {"subject": sub, "session": ses, "channel_group": group, "ecg_artifact": input_y_or_n_artifact, 
                           "ecg_component_first_run": 0, "ecg_component_second_run": 0,"cleaned": "n", "ieeg_saved": "original"}
                new_row = pd.DataFrame([new_row], index=[0])
                ecg_artifact_df = pd.concat([ecg_artifact_df, new_row], ignore_index=True)

                # keep the original data and continue
                clean_time_series_dict[f"{sub}_{hem}_{ses}_{group}"] = [
                    sub,
                    hem, 
                    ses, 
                    group,
                    "no", 
                    data]
                
                continue

            # Step 2: fit ICA on the raw data and plot the components ############################################
            plot_ica = plot_ica_components(data)

            # save figure: 
            plot_ica["fig_ecg_components"].savefig(os.path.join(figures_path, f"ica_components_sub-{sub}_{ses}_{group}.svg"), bbox_inches="tight", format="svg")
            plot_ica["fig_ecg_components"].savefig(os.path.join(figures_path, f"ica_components_sub-{sub}_{ses}_{group}.png"), bbox_inches="tight")

            # extract features from plot_ica
            ica_components = plot_ica["ica_components"]
            ica = plot_ica["ica"]

            # select the ECG component
            first_input_ecg_component = get_input_ecg_component("Which component is the ECG artifact? (if none is obvious type 0 and save original)") # interrups run and asks for input
            if first_input_ecg_component in {1, 2, 3, 4, 5, 6}:
                first_ecg_component = first_input_ecg_component - 1 # index starts at 0
            
            elif first_input_ecg_component == 0:

                # save new row in the excel file
                new_row = {"subject": sub, "session": ses, "channel_group": group, "ecg_artifact": "n", 
                           "ecg_component_first_run": first_input_ecg_component, "ecg_component_second_run": 0,"cleaned": "n", "ieeg_saved": "original"}
                new_row = pd.DataFrame([new_row], index=[0])
                ecg_artifact_df = pd.concat([ecg_artifact_df, new_row], ignore_index=True)

                # keep the original data and continue
                clean_time_series_dict[f"{sub}_{hem}_{ses}_{group}"] = [
                    sub,
                    hem, 
                    ses, 
                    group,
                    "no", 
                    data]
                
                continue


            else: 
                print("Invalid input. Please provide a valid input.")
                break

            # Step 3: remove the ECG artifact from the signals ############################################
            cleaned_ieeg_data = remove_ecg_artifact_from_signals(data, first_ecg_component, ica_components, ica)

            # Step 4: plot the cleaned signals ############################################
            plot_cleaned = plot_ieeg_data(sub, ses, group, cleaned_ieeg_data, fig_title="cleaned_time_series")

            # check if data now clean
            input_y_or_n_clean = get_input_y_n("Data cleaned?") # interrups run and asks for input
            ieeg_saved = "cleaned"
            second_input_ecg_component = 0

            if input_y_or_n_clean == "n":
                second_input_y_or_n = get_input_y_n("Try again?") # interrups run and asks for input

                # try a second time #############################################
                if second_input_y_or_n == "y":
                    second_plot_ica = plot_ica_components(cleaned_ieeg_data)

                    # save figure: 
                    second_plot_ica["fig_ecg_components"].savefig(os.path.join(figures_path, f"second_run_ica_components_sub-{sub}_{ses}_{group}.png"), bbox_inches="tight")

                    # extract features from plot_ica
                    second_ica_components = second_plot_ica["ica_components"]
                    second_ica = second_plot_ica["ica"]

                    second_input_ecg_component = get_input_ecg_component("Which component is the ECG artifact?") # interrups run and asks for input
                    if second_input_ecg_component in {1, 2, 3, 4, 5, 6}:
                        second_ecg_component = second_input_ecg_component - 1 # index starts at 0
                        # remove another component from the cleaned signals ############################################
                        cleaned_ieeg_data = remove_ecg_artifact_from_signals(cleaned_ieeg_data, second_ecg_component, second_ica_components, second_ica)
                        
                        # Step 4: plot the cleaned signals ############################################
                        plot_cleaned = plot_ieeg_data(sub, ses, group, cleaned_ieeg_data, fig_title="second_run_cleaned_time_series")
                
                elif second_input_y_or_n == "n":
                    second_input_ecg_component = 0
            
            # check if data now clean
            input_keep_original = get_input_y_n("Do you want to keep the original? (type y if you want to keep original)") # interrups run and asks for input

            # if still not clean ask to keep original
            if input_keep_original == "y":
                cleaned_ieeg_data = data
                first_input_ecg_component = 0
                second_input_ecg_component = 0
                ieeg_saved = "original"
            
            elif input_keep_original == "n":
                ieeg_saved = "cleaned"
            
            # save new row in the excel file
            new_row = {"subject": sub, "session": ses, "channel_group": group, "ecg_artifact": input_y_or_n_artifact, 
                       "ecg_component_first_run": first_input_ecg_component, "ecg_component_second_run": second_input_ecg_component, "cleaned": input_y_or_n_clean, "ieeg_saved": ieeg_saved}
            new_row = pd.DataFrame([new_row], index=[0])
            ecg_artifact_df = pd.concat([ecg_artifact_df, new_row], ignore_index=True)

            clean_time_series_dict[f"{sub}_{hem}_{ses}_{group}"] = [
                sub,
                hem, 
                ses, 
                group,
                "yes", 
                cleaned_ieeg_data]

    # save the updated excel file
    save_updated_excel(ecg_artifact_df)

    # dataframe to store the results
    clean_time_series_df = pd.DataFrame(clean_time_series_dict)
    clean_time_series_df.rename(index={
        0: "subject",
        1: "hemisphere",
        2: "session",
        3: "channel_group",
        4: "ecg_artifact",
        5: "cleaned",
    }, inplace=True)
    clean_time_series_df = clean_time_series_df.transpose()

    # save the results as pickle
    results_filepath = os.path.join(results_path, "cleaned_time_series.pickle")
    with open(results_filepath, "wb") as file:
        pickle.dump(clean_time_series_df, file)



    return ecg_artifact_df, clean_time_series_df






            



































def plot_raw_time_series(incl_sub: list, incl_session: list, incl_condition: list):
    """
    Function to plot raw time series of all channels for each subject, session, condition and channel group.
    """

    results_path = findfolders.get_local_path(folder="GroupResults")

    # sns.set()
    plt.style.use('seaborn-whitegrid')  

    hemispheres = ["Right", "Left"]

    artifact_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject

    for sub in incl_sub:

        for hem in hemispheres:

            if hem == "Right":
                channel_groups = ["RingR", "SegmIntraR", "SegmInterR"]

            elif hem == "Left":
                channel_groups = ["RingL", "SegmIntraL", "SegmInterL"]

            mainclass_sub = main_class.PerceiveData(
                sub = sub, 
                incl_modalities= ["survey"],
                incl_session = incl_session,
                incl_condition = incl_condition,
                incl_task = ["rest"],
                incl_contact=channel_groups
                )

            
            figures_path = findfolders.get_local_path(folder="figures", sub=sub)

            # add error correction for sub and task??
            
            # one figure for each STN, session, channel group
            for t, tp in enumerate(incl_session):

                for g, group in enumerate(channel_groups):

                    if g == 0:
                        channels = ['03', '13', '02', '12', '01', '23']
                        group_name = "ring"
                        fig_size = (30, 10) 
                    
                    elif g == 1:
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        group_name = "segm_intra"
                        fig_size = (30, 10)
                    
                    elif g == 2:
                        channels = ['1A2A', '1B2B', '1C2C']
                        group_name = "segm_inter"
                        fig_size = (30, 5)

                    for c, cond in enumerate(incl_condition):

                        # set layout for figures: using the object-oriented interface
                        # fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)


                        # avoid Attribute Error, continue if session attribute doesn´t exist
                        try:
                            getattr(mainclass_sub.survey, tp)
                        
                        except AttributeError:
                            continue

                        # if getattr(mainclass_sub.survey, tp) is None:
                        #     continue
        
                        # apply loop over channels
                        temp_data = getattr(mainclass_sub.survey, tp) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                        
                        # avoid Attribute Error, continue if attribute doesn´t exist
                        # if getattr(temp_data, cond) is None:
                        #     continue
                    
                        try:
                            getattr(temp_data, cond)
                            #temp_data = temp_data.rest.data[tasks[tk]]
                        
                        except AttributeError:
                            continue

                        temp_data = getattr(temp_data, cond) # gets attribute e.g. "m0s0"
                        temp_data = getattr(temp_data.rest, group)
                        temp_data = temp_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"
                        
                        # get new channel names
                        ch_names = temp_data.info.ch_names


                        #################### PICK CHANNELS ####################
                        include_channelList = [] # this will be a list with all channel names selected

                        for n, names in enumerate(ch_names):
                            
                            # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
                            for picked in channels:
                                if picked in names:
                                    include_channelList.append(names)

                            
                        # Error Checking: 
                        if len(include_channelList) == 0:
                            continue

                        # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
                        ch_names_indices = mne.pick_channels(ch_names, include=include_channelList)

                        fig, axes = plt.subplots(len(channels), 1, figsize=fig_size) # subplot(rows, columns, panel number), figsize(width,height)

                        for i, ch in enumerate(ch_names):
                            
                            # only get picked channels
                            if i not in ch_names_indices:
                                continue

                            signal = temp_data.get_data()[i, :]

                            # save signals in dictionary
                            artifact_dict[f"{sub}_{hem}_{tp}_{group_name}_{cond}_{channels[i]}"] = [sub, hem, tp, group_name, cond, channels[i], signal]
                            

                            #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################
                            # x = signal[f"{filter}"]
                            # y = np.arange(1, len(signal[f"{filter}"])+1)

                            axes[i].set_title(f"{tp}, {group}, {channels[i]}", fontsize=15) 
                            axes[i].plot(signal, label=f"{channels[i]}_{cond}", color="k", linewidth=0.3)  

                        for ax in axes:
                            ax.set_xlabel("timestamp", fontsize=15)
                            ax.set_ylabel("amplitude", fontsize=15)
                            ax.tick_params(axis='both', which='major', labelsize=10)
                        
                        for ax in axes.flat[:-1]:
                            ax.set(xlabel='')

                        # # interaction: when a movement artifact is found first click = x1, second click = x2
                        # pos = [] # collecting the clicked x and y values for one channel group of stn at one session
                        # def onclick(event):
                        #     pos.append([event.xdata,event.ydata])
                                    
                        # fig.canvas.mpl_connect('button_press_event', onclick)

                        fig.suptitle(f"raw time series sub-{sub}, {hem} hemisphere", ha="center", fontsize=15)
                        fig.tight_layout()
                        plt.subplots_adjust(wspace=0, hspace=0)

                        plt.show(block=False)
                        #plt.gcf().canvas.draw()

                        #input_y_or_n = get_input_y_n("Artifacts found?") # interrups run and asks for input

                        # if input_y_or_n == "y":

                        #     # save figure
                        #     fig.savefig(os.path.join(figures_path, f"raw_time_series_{filter}_sub-{sub}_{hem}_{tp}_{cond}_{group_name}_with_artifact.png"), bbox_inches="tight")

                        #     # store results
                        #     number_of_artifacts = len(pos) / 2

                        #     artifact_x = [x_list[0] for x_list in pos] # list of all clicked x values
                        #     artifact_y = [y_list[1] for y_list in pos] # list of all clicked y values

                        #     move_artifact_dict[f"{sub}_{hem}_{tp}_{group_name}_{cond}"] = [sub, hem, tp, group_name, cond,
                        #                                                             number_of_artifacts, artifact_x, artifact_y]
                        

                        # save figure
                        fig.savefig(os.path.join(figures_path, f"raw_time_series_sub-{sub}_{hem}_{tp}_{cond}_{group_name}.png"), bbox_inches="tight")


                        #number_of_artifacts = len(pos)

                        plt.close()


    move_artifact_result_df = pd.DataFrame(artifact_dict)
    move_artifact_result_df.rename(index={0: "subject", 
                                          1: "hemisphere",
                                          2: "session",
                                          3: "channel_group",
                                          4: "condition",
                                          5: "channel",
                                          6: "signal",
                                          }, inplace=True)
    move_artifact_result_df = move_artifact_result_df.transpose()

    # join two columns sub and hem to one -> subject_hemisphere
    move_artifact_result_df["subject_hemisphere"] = move_artifact_result_df["subject"] + "_" + move_artifact_result_df["hemisphere"]

    # save dataframe as pickle
    # results_filepath = os.path.join(results_path, f"raw_time_series.pickle")
    # with open(results_filepath, "wb") as file:
    #     pickle.dump(move_artifact_result_df, file)   

    return move_artifact_result_df, temp_data



