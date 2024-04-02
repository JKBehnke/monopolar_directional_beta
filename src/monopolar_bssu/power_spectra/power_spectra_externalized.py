""" Read and preprocess externalized LFPs"""


import os
import pickle

import fooof
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof.plts.spectra import plot_spectrum

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import loadResults as loadResults
from ..utils import sub_recordings_dict as sub_recordings_dict
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..externalized_lfp import feats_ssd as feats_ssd
from ..short_time_stability_power import externalized_short_chunks as externalized_short_chunks

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

# patient_metadata = load_data.load_patient_metadata_externalized()
PATIENT_METADATA = io_externalized.load_excel_data(filename="patient_metadata")
HEMISPHERES = ["Right", "Left"]
DIRECTIONAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]
BSSU_CHANNELS = [
    "01",
    "02",
    "03",
    "12",
    "13",
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

BSSU_CHANNELS_12 = [
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

MONOPOLAR_DIRECTIONAL = ["1A", "1B", "1C", "2A", "2B", "2C"]

MONOPOLAR_ALL = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]


FILENAME_DICT = {
    "euclidean_directional_externalized_bssu": "notch_and_band_pass_filtered_externalized_BSSU_euclidean_weighted_power_spectra_only_segmental_inverse_distance_v2",
    "JLB_directional_externalized_bssu": "MonoRef_JLB_notch_and_band_pass_filtered_externalized_BSSU_weighted_power_spectra_v2",
    "detec_strelow_contacts_externalized_bssu": "detec_strelow_notch_and_band_pass_filtered_externalized_BSSU_weighted_power_spectra_v2",
    "euclidean_directional_externalized_fooof": "fooof_externalized_BSSU_euclidean_weighted_power_spectra_only_segmental_inverse_distance_v2",
    "JLB_directional_externalized_fooof": "MonoRef_JLB_fooof_externalized_BSSU_weighted_power_spectra_v2",
    "detec_strelow_contacts_externalized_fooof": "detec_strelow_fooof_externalized_BSSU_weighted_power_spectra_v2",
    "externalized_bssu_monopolar": "",
}

FILTER = {
    "notch_and_band_pass_filtered": "filtered_lfp_250Hz",
    "unfiltered": "lfp_resampled_250Hz",
    "only_high_pass_filtered": "only_high_pass_lfp_250Hz",
}


def get_bids_id_from_sub_hem(subject_hemisphere: str):
    # get BIDS_IDs

    hemisphere = subject_hemisphere.split("_")[1]

    sub = subject_hemisphere.split("_")[0]
    sub_wo_zero = sub[1:]

    # get bids from patient metadata
    bids_id = PATIENT_METADATA.loc[
        PATIENT_METADATA["patient_ID"] == int(sub_wo_zero), "BIDS_key"
    ].values[0]

    return {
        "bids_id": bids_id,
        "hemisphere": hemisphere,
    }


def plot_power_spectra_monopolar(method: str, fooof: str, only_directional: str):
    """
    Input:
        - method: str
            "euclidean_directional_externalized_bssu"
            "JLB_directional_externalized_bssu"
            "detec_strelow_contacts_externalized_bssu"
            "euclidean_directional"
            "JLB_directional"
            "detec_strelow_contacts"
            "externalized_bssu_monopolar"

            - fooof: str "yes" or "no"

            - only_directional: str "yes" or "no"
    """

    # get the correct contacts to plot
    if only_directional == "yes":
        contacts = MONOPOLAR_DIRECTIONAL
        directionality_filename = "directional"
    else:
        contacts = MONOPOLAR_ALL
        directionality_filename = "all_contacts"

    # get correct filename
    filename_load_data = FILENAME_DICT[method]

    # load data
    loaded_data = loadResults.load_pickle_group_result(
        filename=filename_load_data, fooof_version="v2"
    )

    # included subjects
    subject_hemisphere_unique = list(loaded_data.keys())

    for sub_hem in subject_hemisphere_unique:
        # get bids ID, and hemisphere
        bids_id = get_bids_id_from_sub_hem(subject_hemisphere=sub_hem)["bids_id"]
        hemisphere = get_bids_id_from_sub_hem(subject_hemisphere=sub_hem)["hemisphere"]

        # get data
        sub_hem_data = loaded_data[sub_hem]

        # get frequencies
        frequencies = sub_hem_data["frequencies"]
        # get power dictionary (all contacts)
        power_dict = sub_hem_data["weighted_power"]

        # figure path
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        # figure layout for one subject hemisphere
        fig = plt.figure(figsize=(20, 20), layout="tight")
        fig.suptitle(f"Power Spectra: {sub_hem}, {method}", fontsize=55, y=1.02)

        for cont in contacts:
            power_from_contact = power_dict[cont]

            # plot power spectra
            plt.subplot(1, 1, 1)

            plt.plot(frequencies, power_from_contact, label=f"{cont}", linewidth=3)

            plt.xlabel("Frequency [Hz]", fontdict={"size": 40})
            plt.ylabel("PSD", fontdict={"size": 40})

            # plt.ylim(1, 100)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)

            # Plot the legend only for the first row "postop"
            plt.legend(loc="upper right", edgecolor="black", fontsize=40)

        # save figure
        io_externalized.save_fig_png_and_svg(
            path=figures_path,
            filename=f"power_spectra_{method}_{sub_hem}_{directionality_filename}",
            figure=fig,
        )


def plot_externalized_FOOOF_power_spectra(method: str, only_directional: str):
    """
    Input:
        - method: str
            "euclidean_directional_externalized_bssu"
            "JLB_directional_externalized_bssu"
            "detec_strelow_contacts_externalized_bssu"
            "euclidean_directional"
            "JLB_directional"
            "detec_strelow_contacts"
            "externalized_bssu_monopolar"
            "externalized_one_to_zero_two_to_three"

            - fooof: str "yes" or "no"

            - only_directional: str "yes" or "no"
    """

    # get the correct contacts to plot
    if only_directional == "yes":
        contacts = MONOPOLAR_DIRECTIONAL
        directionality_filename = "directional"
    else:
        contacts = MONOPOLAR_ALL
        directionality_filename = "all_contacts"

    if method == "externalized_one_to_zero_two_to_three":
        loaded_data = io_externalized.load_externalized_pickle(
            filename=f"fooof_externalized_group_only_high_pass_filtered",
            reference="bipolar_to_lowermost",
            fooof_version="v2",
        )
    
    else:
    # get correct filename
        print("only works for externalized FOOOF 0-1 and 2-3 (only high pass filtered, FOOOF version 2)")

    # included subjects
    subject_hemisphere_unique = loaded_data.subject_hemisphere.unique()

    for sub_hem in subject_hemisphere_unique:
        # get bids ID, and hemisphere
        bids_id = get_bids_id_from_sub_hem(subject_hemisphere=sub_hem)["bids_id"]
        hemisphere = get_bids_id_from_sub_hem(subject_hemisphere=sub_hem)["hemisphere"]

        # get data
        sub_hem_data = loaded_data.loc[loaded_data.subject_hemisphere == sub_hem]

        # figure path
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        # figure layout for one subject hemisphere
        fig = plt.figure(figsize=(20, 20), layout="tight")
        fig.suptitle(f"Power Spectra: {sub_hem}, {method}", fontsize=55, y=1.02)

        for cont in contacts:
            # there is no fooof data for contact 0
            if cont == "0":
                continue

            cont_data = sub_hem_data.loc[sub_hem_data.contact == cont]

            # check if data exists:
            if len(cont_data) == 0:
                print(f"No data for {sub_hem}, {cont}")
                continue

            fooof_power_spectrum = cont_data["fooof_power_spectrum"].values[0]
            frequencies = np.arange(1, (len(fooof_power_spectrum) + 1))

            # plot power spectra
            plt.subplot(1, 1, 1)

            plt.plot(frequencies, fooof_power_spectrum, label=f"{cont}", linewidth=3)

            plt.xlabel("Frequency [Hz]", fontdict={"size": 40})
            plt.ylabel("PSD", fontdict={"size": 40})

            # plt.ylim(1, 100)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)

            # Plot the legend only for the first row "postop"
            plt.legend(loc="upper right", edgecolor="black", fontsize=40)

        # save figure
        io_externalized.save_fig_png_and_svg(
            path=figures_path,
            filename=f"power_spectra_{method}_{sub_hem}_{directionality_filename}",
            figure=fig,
        )


def plot_short_power_spectra_externalized_bssu(sub_hem_list: list, filtered: str, sec_per_epoch: int, number_epochs: int):
    """
    Input: 
        - sub_hem_list: list e.g. ["024_Right", "024_Left"]
        - filtered: str, "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
        - sec_per_epoch: int, 20
        - number_epochs: int, 9

        Higher frequency resolution here! 
        e.g. 20 sec epochs with 50% overlap: 1/20 sec = 0.05 Hz resolution
    
    """

    # load data
    loaded_data = io_externalized.load_externalized_pickle(
        filename=f"power_spectra_BSSU_externalized_{filtered}_2min_and_{sec_per_epoch}sec_{number_epochs}epochs")
    
    # add a column "sub_hem" with merged columns "subject" + "hemisphere"
    loaded_data["sub_hem"] = loaded_data["subject"] + "_" + loaded_data["hemisphere"]
    
    if sub_hem_list == ["all"]:
        sub_hem_unique = loaded_data["sub_hem"].unique()
    
    else: 
        sub_hem_unique = sub_hem_list

    for sub_hem in sub_hem_unique:
        # get bids ID 
        bids_id = get_bids_id_from_sub_hem(subject_hemisphere=sub_hem)["bids_id"]

        # figure path
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        # get data
        sub_hem_data = loaded_data.loc[loaded_data.sub_hem == sub_hem]

        for chan in BSSU_CHANNELS_12:

            fig = plt.figure(figsize=(20, 20), layout="tight")
            fig.suptitle(
                f"Power Spectra BSSU externalized: {sub_hem}, {chan}",
                fontsize=55,
                y=1.02,
            )
            chan_data = sub_hem_data.loc[sub_hem_data.channel == chan]
            frequencies = chan_data["frequencies"].values[0]

            # get 2min power spectrum data
            power_2min = chan_data["power_spectrum_2_min"].values[0]

            # plot power spectra
            plt.subplot(1, 1, 1)

            # plot the average power spectrum of the 2 min recording
            plt.plot(
                frequencies,
                power_2min,
                label="2 min",
                linewidth=7,
                color="black",
            )
            
            # plot the 20 sec power spectra
            for epoch in range(1, number_epochs + 1):
                
                power_epoch = chan_data[f"power_spectrum_{sec_per_epoch}_sec_{epoch}"].values[0]

                if len(power_epoch) == 0:
                    print(f"Epoch {epoch} of {sub_hem}, {chan} is empty")
                    continue

                plt.plot(
                    frequencies,
                    power_epoch,
                    label=f"{epoch}: {sec_per_epoch} sec",
                    linewidth=3,
                )

                plt.xlabel("Frequency [Hz]", fontdict={"size": 40})
                plt.ylabel("PSD +- SEM", fontdict={"size": 40})

                plt.xlim(0, 80)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)

                # Plot the legend only for the first row "postop"
                plt.legend(loc="upper right", edgecolor="black", fontsize=40)
            
            # save figure
            io_externalized.save_fig_png_and_svg(
                    path=figures_path,
                    filename=f"power_spectra_BSSU_externalized_{number_epochs}x_{sec_per_epoch}_sec_{sub_hem}_{chan}_{filtered}",
                    figure=fig,
                )






    
    



def plot_power_spectra_20sec_externalized_bssu(sub_hem:str, chan:str, fourier_transform_2min, chunks:dict):
    """
    
    Watch out: fourier transform for 20 sec chunks with 50% overlap 
    """
    # figure layout for one subject hemisphere
    fig = plt.figure(figsize=(20, 20), layout="tight")
    fig.suptitle(
        f"Power Spectra BSSU externalized: {sub_hem}, {chan}",
        fontsize=55,
        y=1.02,
    )

    # plot power spectra
    plt.subplot(1, 1, 1)

    # plot the average power spectrum of the 2 min recording
    plt.plot(
        fourier_transform_2min["frequencies"],
        fourier_transform_2min["average_Zxx"],
        label="2 min",
        linewidth=7,
        color="black",
    )
    plt.fill_between(
        fourier_transform_2min["frequencies"],
        fourier_transform_2min["average_Zxx"]
        - fourier_transform_2min["sem_Zxx"],
        fourier_transform_2min["average_Zxx"]
        + fourier_transform_2min["sem_Zxx"],
        color="lightgray",
        alpha=0.5,
    )

    for c in chunks.keys():

        # check if array empty when the recording was too short, too much artifact cleaning
        if len(chunks[c]) == 0:
            print(f"Chunk {c} of {sub_hem}, {chan} is empty")
            continue

        fourier_transform_chunk = (
            externalized_lfp_preprocessing.fourier_transform_to_psd(
                sfreq=250, lfp_data=chunks[c]
            )
        )

        plt.plot(
            fourier_transform_chunk["frequencies"],
            fourier_transform_chunk["average_Zxx"],
            label=f"{c}: 20sec",
            linewidth=3,
        )

        plt.fill_between(
            fourier_transform_chunk["frequencies"],
            fourier_transform_chunk["average_Zxx"]
            - fourier_transform_chunk["sem_Zxx"],
            fourier_transform_chunk["average_Zxx"]
            + fourier_transform_chunk["sem_Zxx"],
            color="lightgray",
            alpha=0.5,
        )

        plt.xlabel("Frequency [Hz]", fontdict={"size": 40})
        plt.ylabel("PSD +- SEM", fontdict={"size": 40})

        plt.xlim(0, 80)
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)

        # Plot the legend only for the first row "postop"
        plt.legend(loc="upper right", edgecolor="black", fontsize=40)
    
    return fig



def bssu_externalized_power_spectra_20sec(incl_sub:list, filtered:str):
    """
    Input:
        - incl_sub: list e.g. ["noBIDS24"]
        - filtered: str, "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"

    Data:
        - externalized data re-referenced to BSSU
        - artefact free
        - 2 min duration
        - sfreq 250 Hz
        - filtered LFP: notch and band pass filtered

    Plot each channel of the dataframe in one plot: 6 chunks of 20 sec each and the average Power Spectrum of 2 min

    Output:
        - Figure: Power Spectra of 2 min and 6 chunks of 20 sec
        - Data: Power Spectra of 2 min and 6 chunks of 20 sec saved as "power_spectra_BSSU_externalized_20sec" in bids-id results folder


    """

    filter_name = FILTER[filtered]

    # load the BSSU externalized data
    extern_bssu_data = io_externalized.load_externalized_pickle(
        filename="externalized_directional_bssu_channels",
        reference="bipolar_to_lowermost",
    )
    
    if incl_sub == ["all"]:
        bids_id_unique = list(extern_bssu_data["BIDS_id"].unique())
    
    else:
        bids_id_unique = incl_sub

    # for every hemisphere, plot all BSSU channels
    for bids_id in bids_id_unique:
        sub_result_df = pd.DataFrame()

        # get data
        sub_data = extern_bssu_data.loc[extern_bssu_data.BIDS_id == bids_id]

        # figure path
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        results_path = find_folders.get_monopolar_project_path(
            folder="results", sub=bids_id
        )

        for hem in HEMISPHERES:
            # get data for one hemisphere
            hem_data = sub_data.loc[sub_data.hemisphere == hem]
            sub_hem = hem_data["subject_hemisphere"].values[0]

            # get the channels
            for chan in BSSU_CHANNELS:
                chan_data = hem_data.loc[hem_data.bipolar_channel == chan]

                # get the fitered time series
                filtered_time_series = chan_data[filter_name].values[0]

                # fourier transform of the 2 min time series
                fourier_transform_2min = (
                    externalized_lfp_preprocessing.fourier_transform_to_psd(
                        sfreq=250, lfp_data=filtered_time_series
                    )
                )

                # cut in 20 sec chunks --> dictionary with keys 1 to 6
                chunks = externalized_lfp_preprocessing.cut_lfp_in_20_sec_chunks(
                    time_series=filtered_time_series
                )

                figure = plot_power_spectra_20sec_externalized_bssu(sub_hem=sub_hem, 
                                                                    chan=chan, 
                                                                    fourier_transform_2min=fourier_transform_2min, 
                                                                    chunks=chunks)

                # save figure
                io_externalized.save_fig_png_and_svg(
                    path=figures_path,
                    filename=f"power_spectra_BSSU_externalized_20sec_{sub_hem}_{chan}_{filtered}",
                    figure=figure,
                )

                # save the data
                sub_result = {
                    "bids_id": [bids_id],
                    "hemisphere": [hem],
                    "channel": [chan],
                    "chunks": [chunks],
                    "fourier_transform_2min": [fourier_transform_2min],
                    "filtered": [filtered],
                }

                sub_result_df_single = pd.DataFrame(sub_result)
                sub_result_df = pd.concat([sub_result_df, sub_result_df_single],ignore_index=True)
        
        # save subject result
        io_externalized.save_sub_result_as_pickle(data=sub_result_df, 
                                                  filename=f"power_spectra_BSSU_externalized_20sec_{filtered}",
                                                  results_path=results_path)

    return sub_result_df


def group_20sec_power_spectra_externalized_bssu(incl_bids_id: list, filtered:str):
    """
    Reads all "power_spectra_BSSU_externalized_20sec" files from the results folder and groups them together
    
    Input:
        - incl_sub: list e.g. ["L003", "noBIDS24"]
        - filtered: str, "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"

    """
    group_20sec_dataframe = pd.DataFrame()  # empty dataframe

    if incl_bids_id == ["all"]:
        # get a list of all bids ids
        bids_id_list = sub_recordings_dict.get_bids_id_all_list()
    
    else:
        bids_id_list = incl_bids_id
    
    for bids_id in bids_id_list:
        # get the results path
        results_path = find_folders.get_monopolar_project_path(
            folder="results", sub=bids_id
        )

        # check if file exists
        if os.path.exists(os.path.join(results_path, f"power_spectra_BSSU_externalized_20sec_{filtered}.pickle")) == False:
            print(f"No data for {bids_id}")
            continue

        # load the data
        sub_result_df = io_externalized.load_sub_result_pickle(
            bids_id=bids_id,
            filename=f"power_spectra_BSSU_externalized_20sec_{filtered}",
        )

        # add subject to the dataframe
        sub = sub_recordings_dict.get_sub_from_bids_id(bids_id)
        sub_result_df["subject"] = sub

        # add the dataframe to the group dataframe
        group_20sec_dataframe = pd.concat(
            [group_20sec_dataframe, sub_result_df], ignore_index=True
        )

    # save as pickle
    io_externalized.save_result_dataframe_as_pickle(data=group_20sec_dataframe,filename=f"power_spectra_BSSU_externalized_20sec_group_{filtered}")

    return group_20sec_dataframe

    

            
            















                
