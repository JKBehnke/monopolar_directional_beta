""" ECG artifact cleaning """

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

# internal
from ..utils import io_externalized as io_externalized
from ..utils import sub_recordings_dict as sub_rec_dict
from ..utils import find_folders as find_folders



HEMISPHERES = ["Right", "Left"]

FILTER = {
    "notch_and_band_pass_filtered": "filtered_lfp_250Hz",
    "unfiltered": "lfp_resampled_250Hz",
    "only_high_pass_filtered": "only_high_pass_lfp_250Hz",
}

CHANNELS = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]


def externalized_rec_to_2D_array(filter:str, data = None):
    """
    Input:
        - filter: str 
                "notch_and_band_pass_filtered",
                "unfiltered",
                "only_high_pass_filtered",
        
        - data: dataframe of one recording -> one subject_hemisphere, 8 contacts

    Output: 2D array with shape (8, 30000)
    """

    filter_name = FILTER[filter]

    # convert the arrays in the lfp column to a list of arrays
    lfp_list = data[filter_name].to_list()

    # stack the arrays vertically to create a 2D array
    lfp_array = np.vstack(lfp_list)

    return lfp_array


def plot_ieeg_data(sub: str, hemisphere: str, ieeg_data, fig_title: str, figures_path: str, filtered: str):
    """
    Function to plot the iEEG data. This function can be used when you have already extracted the data into a 2D array.

    Input:
        - ieeg_data: np.array -> 2D array shape: (n_channels, n_samples)
    """

    plt.style.use('seaborn-whitegrid')  
   
    fig_size = (40, 30)

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=fig_size) # subplot(rows, columns, panel number), figsize(width,height)

    for i, ch in enumerate(CHANNELS):
     
        signal = ieeg_data[i, :]

        #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

        axes[i].set_title(f"Externalized LFP, {CHANNELS[i]}", fontsize=20) 
        axes[i].plot(signal, label=f"{CHANNELS[i]}_m0s0", color="k", linewidth=0.2)  

    for ax in axes:
        ax.set_xlabel("timestamp", fontsize=20)
        ax.set_ylabel("amplitude", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
    
    for ax in axes.flat[:-1]:
        ax.set(xlabel='')

    fig.suptitle(f"{fig_title} sub-{sub}, {hemisphere} hemisphere", ha="center", fontsize=20)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show(block=False)

    # save figure
    filename = f"{fig_title}_sub-{sub}_{hemisphere}_m0s0_externalized_{filtered}"
    io_externalized.save_fig_png_and_svg(path=figures_path, filename=filename, figure=fig)

    return {
        "data": ieeg_data, 
        "channels": CHANNELS,
        "hemisphere": hemisphere

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

    fig_size = (40, 30)
    

    time_points = np.arange(data.shape[1])  # Assuming your data is sampled uniformly
    fig, axes = plt.subplots(n_components, 1, figsize=fig_size, sharex=True, sharey=True)

    for i in range(n_components):
        axes[i].plot(time_points, ica_components[:, i], label=f'Component {i + 1}', linewidth=0.3)
        axes[i].set_title(f"Component {i + 1}", fontsize=20)
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

    # load the ecg_artifacts.xlsx file
    ecg_artifact_df = io_externalized.load_excel_data(filename="ecg_artifacts")

    return ecg_artifact_df

def save_updated_excel(updated_df):
    """
    Save the updated excel file with the new row added.

    Input:
        - ecg_artifact_df: pd.DataFrame -> dataframe with the ecg artifacts
        - new_row: dict -> {"subject": str, "session": str, "channel_group": str, "ecg_component": int, "cleaned": str, "comment": str}
    """

    # find the path to the results folder
    path = find_folders.get_monopolar_project_path(folder="data")

    # load the file
    filename = "ecg_artifacts.xlsx"
    filepath = os.path.join(path, filename)

    updated_df.to_excel(filepath, index=False, sheet_name="ecg_artifacts")
    print("Excel file updated: ", filename, "\nloaded from: ", path)

    return updated_df


############################ SCRIPT TO CLEAN ALL ECG ARTIFACTS FROM THE DATA ############################


def ecg_cleaning(sub: str, filtered: str):
    """
    Function to clean the ECG artifacts from the iEEG data of a subject.

    Input:
        - sub: str -> subject number "024"
        - filtered: str -> "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"

    This function will iterate over all subject hemispheres and clean the ECG artifacts from the iEEG data.
    
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
    bids_id = sub_rec_dict.get_bids_id_from_sub(sub)

    figures_path = find_folders.get_monopolar_project_path(folder="figures", sub=bids_id)
    results_path = find_folders.get_monopolar_project_path(folder="results", sub=bids_id)

    # dictionary to store the results
    clean_time_series_dict = {}

    # get the data
    sub_ieeg = io_externalized.load_externalized_pickle(filename="externalized_preprocessed_data_artefact_free", 
                                                        reference="bipolar_to_lowermost")
    
    sub_ieeg = sub_ieeg.loc[sub_ieeg.subject == sub]

    for hem in HEMISPHERES:

        hem_ieeg = sub_ieeg.loc[sub_ieeg.hemisphere == hem]

        # transfer data to 2D array
        hem_ieeg_array = externalized_rec_to_2D_array(filter=filtered, data=hem_ieeg)

        # Step 1: load and plot the raw time series ############################################
        
        plot_raw = plot_ieeg_data(sub=sub, hemisphere=hem, ieeg_data=hem_ieeg_array, fig_title="raw_time_series", figures_path=figures_path, filtered=filtered)

        # extract the data and channels and hemisphere
        data = plot_raw["data"] # in 2D array shape: (n_channels, n_samples)
        channels = plot_raw["channels"]
        hem = plot_raw["hemisphere"]

        # check if data contains ECG artifact
        input_y_or_n_artifact = get_input_y_n("ECG artifacts found") # interrups run and asks for input
        
        if input_y_or_n_artifact == "n":

            # save new row in the excel file
            new_row = {"subject": sub, "hemisphere": hem, "ecg_artifact": input_y_or_n_artifact, 
                        "ecg_component_first_run": 0, "ecg_component_second_run": 0,"cleaned": "n", "ieeg_saved": "original"}
            new_row = pd.DataFrame([new_row], index=[0])
            ecg_artifact_df = pd.concat([ecg_artifact_df, new_row], ignore_index=True)

            # keep the original data and continue
            clean_time_series_dict[f"{sub}_{hem}"] = [
                sub,
                hem, 
                "no", 
                data]
            
            continue

        # Step 2: fit ICA on the raw data and plot the components ############################################
        plot_ica = plot_ica_components(data)

        # save figure: 
        io_externalized.save_fig_png_and_svg(path=figures_path, filename=f"ica_components_sub-{sub}_{hem}_{filtered}", figure=plot_ica["fig_ecg_components"])

        # extract features from plot_ica
        ica_components = plot_ica["ica_components"]
        ica = plot_ica["ica"]

        # select the ECG component
        first_input_ecg_component = get_input_ecg_component("Which component is the ECG artifact? (if none is obvious type 0 and save original)") # interrups run and asks for input
        if first_input_ecg_component in {1, 2, 3, 4, 5, 6}:
            first_ecg_component = first_input_ecg_component - 1 # index starts at 0
        
        elif first_input_ecg_component == 0:

            # save new row in the excel file
            new_row = {"subject": sub, "hemisphere": hem, "ecg_artifact": "n", 
                        "ecg_component_first_run": first_input_ecg_component, "ecg_component_second_run": 0,"cleaned": "n", "ieeg_saved": "original"}
            new_row = pd.DataFrame([new_row], index=[0])
            ecg_artifact_df = pd.concat([ecg_artifact_df, new_row], ignore_index=True)

            # keep the original data and continue
            clean_time_series_dict[f"{sub}_{hem}"] = [
                sub,
                hem, 
                "no", 
                data]
            
            continue


        else: 
            print("Invalid input. Please provide a valid input.")
            break

        # Step 3: remove the ECG artifact from the signals ############################################
        cleaned_ieeg_data = remove_ecg_artifact_from_signals(data, first_ecg_component, ica_components, ica)

        # Step 4: plot the cleaned signals ############################################
        plot_cleaned = plot_ieeg_data(sub, hemisphere=hem, ieeg_data=cleaned_ieeg_data, 
                                      fig_title="cleaned_time_series", figures_path=figures_path, filtered=filtered) 

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
                io_externalized.save_fig_png_and_svg(path=figures_path, filename=f"second_run_ica_components_sub-{sub}_{hem}_{filtered}", figure=second_plot_ica["fig_ecg_components"])

                # extract features from plot_ica
                second_ica_components = second_plot_ica["ica_components"]
                second_ica = second_plot_ica["ica"]

                second_input_ecg_component = get_input_ecg_component("Which component is the ECG artifact?") # interrups run and asks for input
                if second_input_ecg_component in {1, 2, 3, 4, 5, 6}:
                    second_ecg_component = second_input_ecg_component - 1 # index starts at 0
                    # remove another component from the cleaned signals ############################################
                    cleaned_ieeg_data = remove_ecg_artifact_from_signals(cleaned_ieeg_data, second_ecg_component, second_ica_components, second_ica)
                    
                    # Step 4: plot the cleaned signals ############################################
                    plot_cleaned = plot_ieeg_data(sub, hemisphere=hem, ieeg_data=cleaned_ieeg_data, 
                                                  fig_title="second_run_cleaned_time_series", figures_path=figures_path, filtered=filtered)
            
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
        new_row = {"subject": sub, "hemisphere": hem, "ecg_artifact": input_y_or_n_artifact, 
                    "ecg_component_first_run": first_input_ecg_component, "ecg_component_second_run": second_input_ecg_component, 
                    "cleaned": input_y_or_n_clean, "ieeg_saved": ieeg_saved}
        new_row = pd.DataFrame([new_row], index=[0])
        ecg_artifact_df = pd.concat([ecg_artifact_df, new_row], ignore_index=True)

        clean_time_series_dict[f"{sub}_{hem}"] = [
            sub,
            hem, 
            "yes", 
            cleaned_ieeg_data]

    # save the updated excel file
    save_updated_excel(ecg_artifact_df)

    # dataframe to store the results
    clean_time_series_df = pd.DataFrame(clean_time_series_dict)
    clean_time_series_df.rename(index={
        0: "subject",
        1: "hemisphere",
        2: "ecg_artifact",
        3: "cleaned",
    }, inplace=True)
    clean_time_series_df = clean_time_series_df.transpose()

    # save the results as pickle
    results_filepath = os.path.join(results_path, f"cleaned_time_series_{filtered}.pickle")
    with open(results_filepath, "wb") as file:
        pickle.dump(clean_time_series_df, file)



    return ecg_artifact_df, clean_time_series_df




