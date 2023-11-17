""" Read and preprocess externalized LFPs"""


import os
import pickle

import fooof
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof.plts.spectra import plot_spectrum
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import loadResults as loadResults
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..externalized_lfp import feats_ssd as feats_ssd

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

# patient_metadata = load_data.load_patient_metadata_externalized()
PATIENT_METADATA = io_externalized.load_excel_data(filename="patient_metadata")
HEMISPHERES = ["Right", "Left"]
DIRECTIONAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]
ALL_CONTACTS = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
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


# list of subjects with no BIDS transformation yet -> load these via poly5reader instead of BIDS
SUBJECTS_NO_BIDS = ["24", "28", "29", "48", "49", "56"]


def preprocess_externalized_lfp_referenced(sub: list, reference: str):
    """
    Input:
        - sub: str e.g. [
            "25", "30", "32", "47", "52", "59",
            "61", "64", "67", "69", "71",
            "72", "75", "77", "79", "80"]

        - reference_to_lowermost_ipsilateral, "bipolar_to_lowermost" or "no"

    Load the BIDS .vhdr files with mne_bids.read_raw_bids(bids_path=bids_path)

    if subject doesn't have a BIDS file yet -> load via Poly5reader

    - crop the data to only 2 minutes: 1min - 3min
    - downsample the data to
        1) sfreq=4000 Hz (if it was 4096Hz originally)
        2) sfreq=250 Hz

    - reference all channels of one hemisphere to the lowermost ipsilateral channel

    - Filter 2 versions:
        1) notch filter 50 Hz
        2) notch filter 50 Hz + band-pass filter 5-95 Hz (filterorder 2)

    - save the data of all contacts into Dataframe:
        1) externalized_preprocessed_data -> all versions of the data filtered, unfiltered, 250 Hz, 4000 Hz
        2) externalized_recording_info_original -> information about the original recording
        3) mne_objects_cropped_2_min -> MNE objects of the 4000 Hz and 250 Hz data, unfiltered


    """

    group_data = pd.DataFrame()
    group_originial_rec_info = pd.DataFrame()

    mne_objects = {}

    for patient in sub:
        loaded_patient = io_externalized.load_patient_data(patient=patient)

        mne_data = loaded_patient["mne_data"]
        subject_info = loaded_patient["subject_info"]
        bids_ID = loaded_patient["bids_ID"]

        # from bids_id only keep the part after sub-
        bids_ID = bids_ID.split("-")
        bids_ID = bids_ID[1]

        subject = f"0{patient}"

        recording_info = {}
        processed_recording = {}

        # get info
        ch_names = mne_data.info["ch_names"]
        ch_names_LFP = [chan for chan in ch_names if "LFP" in chan]
        bads = mne_data.info[
            "bads"
        ]  # channel L_01 is mostly used as reference, "bad" channel is the reference
        chs = mne_data.info["chs"]  # list, [0] is dict
        sfreq = mne_data.info["sfreq"]
        n_times = mne_data.n_times  # number of timestamps
        rec_duration = (n_times / sfreq) / 60  # duration in minutes

        for hem in HEMISPHERES:
            mne_copy = mne_data.copy()

            ch_names_hemisphere = [
                chan for chan in ch_names_LFP if f"_{hem[0]}" in chan
            ]
            print(ch_names_hemisphere)

            # pick LFP channels of one hemisphere only
            mne_copy = mne_copy.pick(ch_names_hemisphere)

            recording_info["original_information"] = [
                subject,
                hem,
                bids_ID,
                ch_names,
                bads,
                sfreq,
                subject_info,
                n_times,
                rec_duration,
            ]

            originial_rec_info_columns = [
                "subject",
                "hemisphere",
                "BIDS_id",
                "ch_names",
                "bads",
                "sfreq",
                "subject_info",
                "number_time_stamps",
                "recording_duration",
            ]

            originial_rec_info = pd.DataFrame.from_dict(
                recording_info, orient="index", columns=originial_rec_info_columns
            )

            # select a period of 2 minutes with no aratefacts, default start at 1 min until 3 min
            mne_copy = mne_copy.crop(60, 180)

            # downsample all to 4000 Hz
            if int(sfreq) != 4000:
                mne_copy = mne_copy.copy().resample(sfreq=4000)
                sfreq = mne_copy.info["sfreq"]

            # downsample from TMSi sampling frequency to 250 sfreq (like Percept)
            resampled_250 = mne_copy.copy().resample(sfreq=250)
            sfreq_250 = resampled_250.info["sfreq"]
            # cropped data should have 30000 samples (2 min of sfreq 250)

            if reference == "bipolar_to_lowermost":
                # reference all data with sfreq 4000 Hz
                reference_channel = mne_copy.get_data(picks=f"LFP_{hem[0]}_01_STN_MT")[
                    0
                ]
                reference_channel_250 = resampled_250.get_data(
                    picks=f"LFP_{hem[0]}_01_STN_MT"
                )[0]
                reference_name = "_bipolar_to_lowermost"

                print("All channels referenced to the lowermost ipsilateral channel.")

                # lfp_data = mne_copy.get_data()
                # referenced_data = lfp_data - reference_channel
                # # problem: mne_copy will be updated for Right and then Left, so always Left 01 as reference in the end!
                # mne_copy._data = referenced_data

                # # reference all data with sfreq 250 Hz
                # reference_channel_250 = resampled_250.get_data(picks=f"LFP_{hem[0]}_01_STN_MT")[0]
                # lfp_data_250 = resampled_250.get_data()
                # referenced_data_250 = lfp_data_250 - reference_channel_250
                # resampled_250._data = referenced_data_250

            elif reference == "no":
                reference_name = ""
                print("Original common reference")

            # save the mne object
            mne_objects[f"{patient}_{hem}_4000Hz_2min"] = mne_copy
            mne_objects[f"{patient}_{hem}_resampled_250Hz"] = resampled_250

            ########## save processed LFP data in dataframe ##########
            for idx, chan in enumerate(
                ch_names_hemisphere
            ):  # 8 channels per hemisphere
                # reference each channel to the lowermost ipsilateral channel
                lfp_chan_data = mne_copy.get_data(picks=chan)[0]
                referenced_lfp_chan = lfp_chan_data - reference_channel
                time_stamps = mne_copy[idx][1]

                lfp_chan_data_250 = resampled_250.get_data(picks=chan)[0]
                referenced_lfp_chan_250 = lfp_chan_data_250 - reference_channel_250
                time_stamps_250 = resampled_250[idx][1]

                # ch_name corresponding to Percept -> TODO: is the order always correct???? 02 = 1A? could it also be 1B?
                if "_01_" in chan:
                    monopol_chan_name = "0"

                elif "_02_" in chan:
                    monopol_chan_name = "1A"

                elif "_03_" in chan:
                    monopol_chan_name = "1B"

                elif "_04_" in chan:
                    monopol_chan_name = "1C"

                elif "_05_" in chan:
                    monopol_chan_name = "2A"

                elif "_06_" in chan:
                    monopol_chan_name = "2B"

                elif "_07_" in chan:
                    monopol_chan_name = "2C"

                elif "_08_" in chan:
                    monopol_chan_name = "3"

                # hemisphere
                if "_L_" in chan:
                    hemisphere = "Left"

                elif "_R_" in chan:
                    hemisphere = "Right"

                # subject_hemisphere
                subject_hemisphere = f"{subject}_{hemisphere}"

                # only high-pass filter 1 Hz
                only_high_pass_lfp_4000 = (
                    externalized_lfp_preprocessing.high_pass_filter_externalized(
                        fs=sfreq, signal=referenced_lfp_chan
                    )
                )
                only_high_pass_lfp_250 = (
                    externalized_lfp_preprocessing.high_pass_filter_externalized(
                        fs=sfreq, signal=referenced_lfp_chan_250
                    )
                )

                # notch filter 50 Hz
                notch_filtered_lfp_4000 = (
                    externalized_lfp_preprocessing.notch_filter_externalized(
                        fs=sfreq, signal=referenced_lfp_chan
                    )
                )
                notch_filtered_lfp_250 = (
                    externalized_lfp_preprocessing.notch_filter_externalized(
                        fs=sfreq_250, signal=referenced_lfp_chan_250
                    )
                )

                # band pass filter 5-95 Hz, Butter worth filter order 3
                filtered_lfp_4000 = (
                    externalized_lfp_preprocessing.band_pass_filter_externalized(
                        fs=sfreq, signal=notch_filtered_lfp_4000
                    )
                )
                filtered_lfp_250 = (
                    externalized_lfp_preprocessing.band_pass_filter_externalized(
                        fs=sfreq_250, signal=notch_filtered_lfp_250
                    )
                )

                # number of samples
                n_samples_250 = len(filtered_lfp_250)

                processed_recording[chan] = [
                    reference,
                    bids_ID,
                    subject,
                    hemisphere,
                    subject_hemisphere,
                    chan,
                    monopol_chan_name,
                    referenced_lfp_chan,
                    time_stamps,
                    sfreq,
                    sfreq_250,
                    referenced_lfp_chan_250,
                    time_stamps_250,
                    filtered_lfp_4000,
                    filtered_lfp_250,
                    only_high_pass_lfp_4000,
                    only_high_pass_lfp_250,
                    n_samples_250,
                ]

        preprocessed_dataframe_columns = [
            "reference",
            "BIDS_id",
            "subject",
            "hemisphere",
            "subject_hemisphere",
            "original_ch_name",
            "contact",
            "lfp_2_min",
            "time_stamps",
            "sfreq",
            "sfreq_250Hz",
            "lfp_resampled_250Hz",
            "time_stamps_250Hz",
            "filtered_lfp_4000Hz",
            "filtered_lfp_250Hz",
            "only_high_pass_lfp_4000Hz",
            "only_high_pass_lfp_250Hz",
            "n_samples_250Hz",
        ]

        preprocessed_dataframe = pd.DataFrame.from_dict(
            processed_recording, orient="index", columns=preprocessed_dataframe_columns
        )

        group_data = pd.concat([group_data, preprocessed_dataframe])
        group_originial_rec_info = pd.concat(
            [group_originial_rec_info, originial_rec_info]
        )

    # save dataframes
    io_externalized.save_result_dataframe_as_pickle(
        data=group_data, filename=f"externalized_preprocessed_data{reference_name}"
    )
    io_externalized.save_result_dataframe_as_pickle(
        data=group_originial_rec_info,
        filename=f"externalized_recording_info_original{reference_name}",
    )
    io_externalized.save_result_dataframe_as_pickle(
        data=mne_objects, filename=f"mne_objects_cropped_2_min{reference_name}"
    )

    return {
        "group_originial_rec_info": group_originial_rec_info,
        "group_data": group_data,
        "mne_objects": mne_objects,
    }


def fourier_transform_time_frequency_plots(incl_bids_id: list, reference=None):
    """
    Input:
        - incl_bids_id: list of bids_id ["L001", "L013"] or ["all"]
        - reference_to_lowermost_ipsilateral, "bipolar_to_lowermost" or "no"

    Load the preprocessed data: externalized_preprocessed_data.pickle

    - For each subject, plot a Time Frequency figure of all channels for Left and Right hemisphere
        1) extract only the filtered data, sfreq=250 Hz

    - save figure into subject figures folder

    """

    # load the dataframe with all filtered LFP data
    preprocessed_data = io_externalized.load_externalized_pickle(
        filename="externalized_preprocessed_data", reference=reference
    )

    # get all subject_hemispheres
    if "all" in incl_bids_id:
        BIDS_id_unique = list(preprocessed_data.BIDS_id.unique())

    else:
        BIDS_id_unique = incl_bids_id

    # plot all time frequency plots of the 250 Hz sampled filtered LFPs
    for BIDS_id in BIDS_id_unique:
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=BIDS_id
        )
        subject_data = preprocessed_data.loc[preprocessed_data.BIDS_id == BIDS_id]
        sub = subject_data.subject.values[0]

        for hem in HEMISPHERES:
            sub_hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(sub_hem_data.contact.values)

            # Figure of one subject_hemisphere with all 8 channels
            # 4 columns, 2 rows

            fig = plt.figure(figsize=(30, 30), layout="tight")

            for c, contact in enumerate(contacts):
                contact_data = sub_hem_data.loc[sub_hem_data.contact == contact]

                # filtered LFP from one contact, resampled to 250 Hz
                filtered_lfp_250 = contact_data.filtered_lfp_250Hz.values[0]
                sfreq = 250

                # Calculate the short time Fourier transform (STFT) using hamming window
                window_length = int(sfreq)  # 1 second window length
                overlap = window_length // 4  # 25% overlap

                frequencies, times, Zxx = signal.stft(
                    filtered_lfp_250,
                    fs=sfreq,
                    nperseg=window_length,
                    noverlap=overlap,
                    window="hamming",
                )
                # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
                # times: len=161, 0, 0.75, 1.5 .... 120.75
                # Zxx: 126 arrays, each len=161
                # Zxx with imaginary values -> take the absolute!
                # to get power -> **2

                plt.subplot(4, 2, c + 1)  # row 1: 1, 2, 3, 4; row 2: 5, 6, 7, 8
                plt.title(f"Channel {contact}", fontdict={"size": 40})
                plt.pcolormesh(
                    times, frequencies, np.abs(Zxx), shading="auto", cmap="viridis"
                )

                plt.xlabel("Time [s]", fontdict={"size": 30})
                plt.ylabel("Frequency [Hz]", fontdict={"size": 30})
                plt.yticks(np.arange(0, 512, 30), fontsize=20)
                plt.ylim(1, 100)
                plt.xticks(fontsize=20)

            fig.suptitle(
                f"Time Frequency sub-{sub}, {hem} hemisphere, fs = 250 Hz, reference: {reference}",
                fontsize=55,
                y=1.02,
            )
            plt.show()

            fig.savefig(
                os.path.join(
                    figures_path,
                    f"Time_Frequency_sub{sub}_{hem}_filtered_250Hz_resampled_reference_{reference}.png",
                ),
                bbox_inches="tight",
            )


def clean_artefacts(reference=None):
    """
    Input:
    - reference_to_lowermost_ipsilateral, "bipolar_to_lowermost" or "no"

    Clean artefacts:

    - Load the artefact Excel sheet with the time in seconds of when visually inspected artefacts start and end
    - load the preprocessed data

    - clean the artefacts from 3 versions of 250 Hz resampled data:
        - lfp_resampled_250 Hz -> unfiltered LFP, resampled to 250 Hz
        - filtered_lfp_250Hz -> notch + band-pass filtered, resampled to 250 Hz
        - only_high_pass_lfp_250Hz -> only high-pass 1 Hz

    - Plot again the clean Time Frequency plots (sfreq=250 Hz, filtered, artefact-free) to check, if artefacts are gone

    - copy the old preprocessed dataframe, replace the original data by the clean artefact-free data:
        externalized_preprocessed_data_artefact_free.pickle

    """
    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    sfreq = 250

    # load data
    artefacts_excel = io_externalized.load_excel_data(filename="movement_artefacts")
    preprocessed_data = io_externalized.load_externalized_pickle(
        filename="externalized_preprocessed_data", reference=reference
    )

    # artefact_free_dataframe= pd.DataFrame()
    artefact_free_dataframe = preprocessed_data.copy()
    artefact_free_dataframe = artefact_free_dataframe.reset_index(drop=True)

    # check which subjects have artefacts
    artefacts_excel = artefacts_excel.loc[artefacts_excel.contacts == "all"]

    BIDS_id_artefacts = list(artefacts_excel.BIDS_key.unique())

    for bids_id in BIDS_id_artefacts:
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        # data only of one subject
        subject_artefact_data = artefacts_excel.loc[artefacts_excel.BIDS_key == bids_id]
        subject_data = preprocessed_data.loc[preprocessed_data.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        # check which hemispheres have artefacts
        hemispheres_with_artefacts = []

        if "Right" in subject_artefact_data.hemisphere.values:
            hemispheres_with_artefacts.append("Right")

        if "Left" in subject_artefact_data.hemisphere.values:
            hemispheres_with_artefacts.append("Left")

        for hem in hemispheres_with_artefacts:
            # get artefact data from one subject hemisphere
            hem_artefact_data = subject_artefact_data.loc[
                subject_artefact_data.hemisphere == hem
            ]

            artefact_samples_list = []

            # TODO: shorten this paragraph!
            # for art_id in range(0, 3):
            #     start = hem_artefact_data[f"artefact{art_id}_start"]
            #     if start.isna().any():
            #         break

            #     start = start.values[0]
            #     stop = hem_artefact_data[f"artefact{art_id}_stop"].values[0]

            artefact1_start = hem_artefact_data.artefact1_start.values[0]
            artefact1_stop = hem_artefact_data.artefact1_stop.values[0]

            # calculate the samples: X sample = 250 Hz * second
            sample_start1 = int(sfreq * artefact1_start)
            sample_stop1 = int(sfreq * artefact1_stop)

            artefact_samples_list.append(sample_start1)

            # check if there are more artefacts
            if hem_artefact_data["artefact2_start"].notna().any():
                artefact2_start = hem_artefact_data.artefact2_start.values[0]
                artefact2_stop = hem_artefact_data.artefact2_stop.values[0]

                sample_start2 = int(sfreq * artefact2_start)
                sample_stop2 = int(sfreq * artefact2_stop)

                artefact_samples_list.append(sample_start2)

            if hem_artefact_data["artefact3_start"].notna().any():
                artefact3_start = hem_artefact_data.artefact3_start.values[0]
                artefact3_stop = hem_artefact_data.artefact3_stop.values[0]

                sample_start3 = int(sfreq * artefact3_start)
                sample_stop3 = int(sfreq * artefact3_stop)

                artefact_samples_list.append(sample_start3)

            # get lfp data from one subject hemisphere
            hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(hem_data.contact.values)

            # Figure of one subject_hemisphere with all 8 channels
            # 4 columns, 2 rows

            fig = plt.figure(figsize=(30, 30), layout="tight")

            for c, contact in enumerate(contacts):
                contact_data = hem_data.loc[hem_data.contact == contact]

                lfp_to_clean = [
                    "filtered_lfp_250Hz",
                    "lfp_resampled_250Hz",
                    "only_high_pass_lfp_250Hz",
                ]

                for data in lfp_to_clean:
                    # get LFP data from one contact
                    lfp_data = contact_data[data].values[0]
                    # column_name = data

                    # clean artefacts from LFP data
                    # check how many artefacts 1-3?
                    if len(artefact_samples_list) == 1:
                        data_clean_1 = lfp_data[0 : sample_start1 + 1]
                        data_clean_2 = lfp_data[sample_stop1:30000]
                        clean_data = np.concatenate([data_clean_1, data_clean_2])

                    elif len(artefact_samples_list) == 2:
                        data_clean_1 = lfp_data[0 : sample_start1 + 1]
                        data_clean_2 = lfp_data[sample_stop1 : sample_start2 + 1]
                        data_clean_3 = lfp_data[sample_stop2:30000]
                        clean_data = np.concatenate(
                            [data_clean_1, data_clean_2, data_clean_3]
                        )

                    elif len(artefact_samples_list) == 3:
                        data_clean_1 = lfp_data[0 : sample_start1 + 1]
                        data_clean_2 = lfp_data[sample_stop1 : sample_start2 + 1]
                        data_clean_3 = lfp_data[sample_stop2 : sample_start3 + 1]
                        data_clean_4 = lfp_data[sample_stop3:30000]
                        clean_data = np.concatenate(
                            [data_clean_1, data_clean_2, data_clean_3, data_clean_4]
                        )

                    # replace artefact_free data in the copied original dataframe
                    # get the index of the contact you're in
                    row_index = artefact_free_dataframe[
                        (artefact_free_dataframe["BIDS_id"] == bids_id)
                        & (artefact_free_dataframe["hemisphere"] == hem)
                        & (artefact_free_dataframe["contact"] == contact)
                    ]
                    row_index = row_index.index[0]

                    # filtered LFP, resampled to 250 Hz
                    artefact_free_dataframe.at[row_index, data] = clean_data
                    artefact_free_dataframe.loc[row_index, "n_samples_250Hz"] = len(
                        clean_data
                    )

                    # plot only the filtered 250 Hz Power Spectra, separately for each subject hemisphere
                    if data == "filtered_lfp_250Hz":
                        ############################# Calculate the short time Fourier transform (STFT) using hamming window #############################
                        window_length = int(sfreq)  # 1 second window length
                        overlap = window_length // 4  # 25% overlap

                        frequencies, times, Zxx = signal.stft(
                            clean_data,
                            fs=sfreq,
                            nperseg=window_length,
                            noverlap=overlap,
                            window="hamming",
                        )
                        # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
                        # times: len=161, 0, 0.75, 1.5 .... 120.75
                        # Zxx: 126 arrays, each len=161
                        # Zxx with imaginary values -> take the absolute!
                        # to get power -> **2

                        plt.subplot(4, 2, c + 1)  # row 1: 1, 2, 3, 4; row 2: 5, 6, 7, 8
                        plt.title(f"Channel {contact}", fontdict={"size": 40})
                        plt.pcolormesh(
                            times,
                            frequencies,
                            np.abs(Zxx),
                            shading="auto",
                            cmap="viridis",
                        )

                        plt.xlabel("Time [s]", fontdict={"size": 30})
                        plt.ylabel("Frequency [Hz]", fontdict={"size": 30})
                        plt.yticks(np.arange(0, 512, 30), fontsize=20)
                        plt.ylim(1, 100)
                        plt.xticks(fontsize=20)

            fig.suptitle(
                f"Time Frequency sub-{sub}, {hem} hemisphere, fs = 250 Hz, artefact-free, reference {reference}",
                fontsize=55,
                y=1.02,
            )
            plt.show()

            fig.savefig(
                os.path.join(
                    figures_path,
                    f"Time_Frequency_sub{sub}_{hem}_filtered_250Hz_resampled_artefact_free_reference{reference_name}.png",
                ),
                bbox_inches="tight",
            )

    # save dataframe
    io_externalized.save_result_dataframe_as_pickle(
        data=artefact_free_dataframe,
        filename=f"externalized_preprocessed_data_artefact_free{reference_name}",
    )

    return artefact_free_dataframe


def fourier_transform_to_psd(reference=None):
    """
    Input:
        - reference, "bipolar_to_lowermost" or "no"

    Load the artefact free data (only the sfreq=250 Hz data is artefact-free!!):
        - 2 min rest
        - artefacts removed
        - resampled to 250 Hz
        - 3 versions: filtered (notch, band-pass), only high-pass-filtered 1 Hz and unfiltered

    calculate the power spectrum for both filtered and unfiltered LFP:
        - window length = 250 # 1 second window length
        - overlap = window_length // 4 # 25% overlap
        - window = hann(window_length, sym=False)
        - frequencies, times, Zxx = scipy.signal.spectrogram(band_pass_filtered, fs=fs, window=window, noverlap=overlap, scaling="density", mode="psd", axis=0)

    Plot the Power Spectra only of the filtered sfreq=250Hz data

    Save the Power Spectra of all filtered and unfiltered data in one dataframe:
        - externalized_power_spectra_250Hz_artefact_free.pickle

    """

    sfreq = 250
    hemispheres = ["Right", "Left"]

    power_spectra_dict = {}

    artefact_free_lfp = io_externalized.load_externalized_pickle(
        filename="externalized_preprocessed_data_artefact_free", reference=reference
    )

    BIDS_id_unique = list(artefact_free_lfp.BIDS_id.unique())

    for bids_id in BIDS_id_unique:
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        # data only of one subject
        subject_data = artefact_free_lfp.loc[artefact_free_lfp.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        for hem in hemispheres:
            hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(hem_data.contact.values)
            subject_hemisphere = f"{sub}_{hem}"

            # Figure of one subject_hemisphere with all 8 channels
            # 4 columns, 2 rows
            fig = plt.figure(figsize=(30, 30), layout="tight")

            for c, contact in enumerate(contacts):
                contact_data = hem_data.loc[hem_data.contact == contact]

                original_ch_name = contact_data.original_ch_name.values[0]

                # get LFP data from one contact

                loop_over_data = [
                    "filtered_lfp_250Hz",
                    "lfp_resampled_250Hz",
                    "only_high_pass_lfp_250Hz",
                ]
                filter_details = [
                    "notch_and_band_pass_filtered",
                    "unfiltered",
                    "only_high_pass_filtered",
                ]

                for f, filt in enumerate(loop_over_data):
                    lfp_data = contact_data[f"{filt}"].values[0]
                    filter_id = filter_details[f]

                    ######### short time fourier transform to calculate PSD #########
                    window_length = int(sfreq)  # 1 second window length
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

                    # save power spectra values
                    power_spectra_dict[f"{bids_id}_{hem}_{contact}_{filter_id}"] = [
                        bids_id,
                        sub,
                        hem,
                        subject_hemisphere,
                        contact,
                        original_ch_name,
                        filter_id,
                        lfp_data,
                        frequencies,
                        times,
                        Zxx,
                        average_Zxx,
                        std_Zxx,
                        sem_Zxx,
                        sfreq,
                        reference,
                    ]

                    if filter_id == "notch_and_band_pass_filtered":
                        plt.subplot(4, 2, c + 1)  # row 1: 1, 2, 3, 4; row 2: 5, 6, 7, 8
                        plt.title(f"Channel {contact}", fontdict={"size": 40})
                        plt.plot(frequencies, average_Zxx)
                        plt.fill_between(
                            frequencies,
                            average_Zxx - sem_Zxx,
                            average_Zxx + sem_Zxx,
                            color="lightgray",
                            alpha=0.5,
                        )

                        plt.xlabel("Frequency [Hz]", fontdict={"size": 30})
                        plt.ylabel("PSD", fontdict={"size": 30})
                        # plt.ylim(1, 100)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)

            fig.suptitle(
                f"Power Spectrum sub-{sub}, {hem} hemisphere, fs = 250 Hz, filtered and artefact-free, reference {reference}",
                fontsize=55,
                y=1.02,
            )
            plt.show()

            fig.savefig(
                os.path.join(
                    figures_path,
                    f"Power_spectrum_sub{sub}_{hem}_filtered_250Hz_resampled_artefact_free_reference_{reference}.png",
                ),
                bbox_inches="tight",
            )

    power_spectra_df = pd.DataFrame(power_spectra_dict)
    power_spectra_df.rename(
        index={
            0: "BIDS_id",
            1: "subject",
            2: "hemisphere",
            3: "subject_hemisphere",
            4: "contact",
            5: "original_ch_name",
            6: "filtered",
            7: "lfp_data",
            8: "frequencies",
            9: "times",
            10: "power",
            11: "power_average_over_time",
            12: "power_std",
            13: "power_sem",
            14: "sfreq",
            15: "reference",
        },
        inplace=True,
    )
    power_spectra_df = power_spectra_df.transpose()

    # save dataframe
    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    io_externalized.save_result_dataframe_as_pickle(
        data=power_spectra_df,
        filename=f"externalized_power_spectra_250Hz_artefact_free{reference_name}",
    )

    return power_spectra_df


def re_reference_externalized(incl_bids_id: list, reference: str, new_reference: str):
    """
    Simulating the segmental BSSU Percept channels with the externalized LFP data
    Input:
        - reference: "bipolar_to_lowermost" or "no"
        - new_reference: "bssu", "one_to_zero_two_to_three"


    """

    channels_group_data = pd.DataFrame()

    # load the dataframe with all filtered LFP data
    preprocessed_data = io_externalized.load_externalized_pickle(
        filename="externalized_preprocessed_data_artefact_free", reference=reference
    )

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"
    elif reference == "no":
        reference_name = ""

    # get all subject_hemispheres
    if "all" in incl_bids_id:
        BIDS_id_unique = list(preprocessed_data.BIDS_id.unique())

    else:
        BIDS_id_unique = incl_bids_id

    # re-reference channels to 15 bipolar channels as in Percept BSSU
    for BIDS_id in BIDS_id_unique:
        subject_data = preprocessed_data.loc[preprocessed_data.BIDS_id == BIDS_id]
        sub = subject_data.subject.values[0]

        for hem in HEMISPHERES:
            sub_hem = f"{sub}_{hem}"
            contact_lfp_dict = {}

            sub_hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(sub_hem_data.contact.values)
            time_stamps_250Hz = sub_hem_data.time_stamps_250Hz.values[0]

            # save the relevant LFP data from each contact into dictionary: unfiltered LFP, sf= 250 Hz
            for contact in contacts:
                contact_data = sub_hem_data.loc[sub_hem_data.contact == contact]

                # unfiltered LFP, sf= 250 Hz
                unfiltered_lfp_250 = contact_data.lfp_resampled_250Hz.values[0]

                contact_lfp_dict[
                    str(contact)
                ] = unfiltered_lfp_250  # dictionary with LFP of each contact

            # calculate the power for contact 1 (1A+1B+1C) and 2 (2A+2B+2C)
            contact_lfp_dict["1"] = (
                contact_lfp_dict["1A"] + contact_lfp_dict["1B"] + contact_lfp_dict["1C"]
            ) / 3
            contact_lfp_dict["2"] = (
                contact_lfp_dict["2A"] + contact_lfp_dict["2B"] + contact_lfp_dict["2C"]
            ) / 3

            if new_reference == "BSSU":
                # re-reference the LFP data of each contact to the 15 bipolar channels
                for bssu_chan in BSSU_CHANNELS:
                    # get the 2 contacts of the bipolar channel
                    if len(bssu_chan) == 2:
                        contact1 = bssu_chan[0]
                        contact2 = bssu_chan[1]

                    elif len(bssu_chan) == 4:
                        contact1 = bssu_chan[:2]
                        contact2 = bssu_chan[2:]

                    # get the LFP data of the 2 contacts
                    contact1_lfp = contact_lfp_dict[str(contact1)]
                    contact2_lfp = contact_lfp_dict[str(contact2)]

                    # bipolar re-referencing
                    bipolar_lfp = contact1_lfp - contact2_lfp

                    # FILTER
                    # only high-pass filter 1 Hz
                    only_high_pass_lfp_250 = (
                        externalized_lfp_preprocessing.high_pass_filter_externalized(
                            fs=250, signal=bipolar_lfp
                        )
                    )

                    # notch filter 50 Hz
                    notch_filtered_lfp_250 = (
                        externalized_lfp_preprocessing.notch_filter_externalized(
                            fs=250, signal=bipolar_lfp
                        )
                    )

                    # band pass filter 5-95 Hz, Butter worth filter order 3
                    filtered_lfp_250 = (
                        externalized_lfp_preprocessing.band_pass_filter_externalized(
                            fs=250, signal=notch_filtered_lfp_250
                        )
                    )

                    # number of samples
                    n_samples_250 = len(filtered_lfp_250)

                    # save the bipolar LFP data into the dictionary

                    channels_dict = {
                        "BIDS_id": [BIDS_id],
                        "subject": [sub],
                        "hemisphere": [hem],
                        "session": ["pre-IPG"],
                        "subject_hemisphere": [sub_hem],
                        "bipolar_channel": [bssu_chan],
                        "time_stamps_250Hz": [time_stamps_250Hz],
                        "lfp_resampled_250Hz": [bipolar_lfp],
                        "filtered_lfp_250Hz": [filtered_lfp_250],
                        "only_high_pass_lfp_250Hz": [only_high_pass_lfp_250],
                        "n_samples_250Hz": [n_samples_250],
                    }
                    channels_single = pd.DataFrame(channels_dict)

                    # save the bipolar LFP data into a dataframe
                    channels_group_data = pd.concat(
                        [channels_group_data, channels_single]
                    )

            elif new_reference == "one_to_zero_two_to_three":
                for re_ref_chan in ALL_CONTACTS:
                    # reference all directional contacts from level 1 to 0
                    if re_ref_chan[0] == "1":
                        re_referenced_channel = (
                            contact_lfp_dict[re_ref_chan] - contact_lfp_dict["0"]
                        )

                    elif re_ref_chan[0] == "2":
                        re_referenced_channel = (
                            contact_lfp_dict[re_ref_chan] - contact_lfp_dict["3"]
                        )

                    elif re_ref_chan[0] == "0":
                        re_referenced_channel = contact_lfp_dict["0"]

                    elif re_ref_chan[0] == "3":
                        re_referenced_channel = contact_lfp_dict["3"]

                    # FILTER
                    # only high-pass filter 1 Hz
                    only_high_pass_lfp_250 = (
                        externalized_lfp_preprocessing.high_pass_filter_externalized(
                            fs=250, signal=re_referenced_channel
                        )
                    )

                    # notch filter 50 Hz
                    notch_filtered_lfp_250 = (
                        externalized_lfp_preprocessing.notch_filter_externalized(
                            fs=250, signal=re_referenced_channel
                        )
                    )

                    # band pass filter 5-95 Hz, Butter worth filter order 3
                    filtered_lfp_250 = (
                        externalized_lfp_preprocessing.band_pass_filter_externalized(
                            fs=250, signal=notch_filtered_lfp_250
                        )
                    )

                    # number of samples
                    n_samples_250 = len(filtered_lfp_250)

                    # save the bipolar LFP data into the dictionary

                    channels_dict = {
                        "BIDS_id": [BIDS_id],
                        "subject": [sub],
                        "hemisphere": [hem],
                        "session": ["pre-IPG"],
                        "subject_hemisphere": [sub_hem],
                        "bipolar_channel": [re_ref_chan],
                        "time_stamps_250Hz": [time_stamps_250Hz],
                        "lfp_resampled_250Hz": [re_referenced_channel],
                        "filtered_lfp_250Hz": [filtered_lfp_250],
                        "only_high_pass_lfp_250Hz": [only_high_pass_lfp_250],
                        "n_samples_250Hz": [n_samples_250],
                    }
                    channels_single = pd.DataFrame(channels_dict)

                    # save the bipolar LFP data into a dataframe
                    channels_group_data = pd.concat(
                        [channels_group_data, channels_single]
                    )

    # save dataframe
    io_externalized.save_result_dataframe_as_pickle(
        data=channels_group_data,
        filename=f"externalized_directional_{new_reference}_channels{reference_name}",
    )

    return channels_group_data


def fourier_transform_to_psd_re_ref_externalized(
    incl_BIDS: list, monopolar_or_bipolar: str, new_reference: str, reference=None
):
    """
    Input:
        - reference, "bipolar_to_lowermost" or "no"
        - incl_BIDS: list of bids_id ["L001", "L013"] or ["all"]
        - monopol_or_bipol: "monopolar" or "bipolar"
        - new_reference: "bssu", "one_to_zero_two_to_three"

    Load the artefact free data (only the sfreq=250 Hz data is artefact-free!!):
        - 2 min rest
        - artefacts removed
        - resampled to 250 Hz
        - 3 versions: filtered (notch, band-pass), only high-pass-filtered 1 Hz and unfiltered

    calculate the power spectrum for both filtered and unfiltered LFP:
        - window length = 250 # 1 second window length
        - overlap = window_length // 4 # 25% overlap
        - window = hann(window_length, sym=False)
        - frequencies, times, Zxx = scipy.signal.spectrogram(band_pass_filtered, fs=fs, window=window, noverlap=overlap, scaling="density", mode="psd", axis=0)

    Plot the Power Spectra only of the filtered sfreq=250Hz data

    Save the Power Spectra of all filtered and unfiltered data in one dataframe:
        - externalized_power_spectra_250Hz_artefact_free.pickle

    """

    sfreq = 250
    hemispheres = ["Right", "Left"]

    power_spectra_dict = {}

    if monopolar_or_bipolar == "bipolar":
        artefact_free_lfp = io_externalized.load_externalized_pickle(
            filename=f"externalized_directional_{new_reference}_channels",
            reference=reference,
        )
        fig_title = f"{new_reference} re-referenced externalized LFP"

        if new_reference == "BSSU":
            fname_extension = "BSSU_"
        elif new_reference == "one_to_zero_two_to_three":
            fname_extension = "one_to_zero_two_to_three_"

    elif monopolar_or_bipolar == "monopolar":
        artefact_free_lfp = io_externalized.load_externalized_pickle(
            filename="externalized_preprocessed_data_artefact_free", reference=reference
        )
        fig_title = "Directional externalized LFP, common reference=lowermost contact ipsilateral"
        fname_extension = ""

    # get all subject_hemispheres
    if "all" in incl_BIDS:
        BIDS_id_unique = list(artefact_free_lfp.BIDS_id.unique())

    else:
        BIDS_id_unique = incl_BIDS

    for bids_id in BIDS_id_unique:
        figures_path = find_folders.get_monopolar_project_path(
            folder="figures", sub=bids_id
        )

        # data only of one subject
        subject_data = artefact_free_lfp.loc[artefact_free_lfp.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        for hem in hemispheres:
            hem_data = subject_data.loc[subject_data.hemisphere == hem]

            if monopolar_or_bipolar == "bipolar":
                channel_list = list(hem_data.bipolar_channel.values)
                c_id = "bipolar_channel"

            elif monopolar_or_bipolar == "monopolar":
                channel_list = list(hem_data.contact.values)
                c_id = "contact"

            subject_hemisphere = f"{sub}_{hem}"

            # Figure of one subject_hemisphere with all 9 directional channels
            fig = plt.figure(figsize=(20, 20), layout="tight")

            for chan in channel_list:
                channel_data = hem_data.loc[hem_data[c_id] == chan]

                # get LFP data from one channel

                loop_over_data = [
                    "filtered_lfp_250Hz",
                    "lfp_resampled_250Hz",
                    "only_high_pass_lfp_250Hz",
                ]
                filter_details = [
                    "notch_and_band_pass_filtered",
                    "unfiltered",
                    "only_high_pass_filtered",
                ]

                for f, filt in enumerate(loop_over_data):
                    lfp_data = channel_data[f"{filt}"].values[0]
                    filter_id = filter_details[f]

                    ######### short time fourier transform to calculate PSD #########
                    window_length = int(sfreq)  # 1 second window length
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

                    # save power spectra values
                    power_spectra_dict[f"{bids_id}_{hem}_{chan}_{filter_id}"] = [
                        bids_id,
                        sub,
                        hem,
                        subject_hemisphere,
                        chan,
                        filter_id,
                        lfp_data,
                        frequencies,
                        times,
                        Zxx,
                        average_Zxx,
                        std_Zxx,
                        sem_Zxx,
                        sfreq,
                        reference,
                    ]

                    if filter_id == "notch_and_band_pass_filtered":
                        plt.subplot(1, 1, 1)
                        plt.title(fig_title, fontdict={"size": 40})
                        plt.plot(frequencies, average_Zxx, label=f"{chan}", linewidth=3)
                        plt.fill_between(
                            frequencies,
                            average_Zxx - sem_Zxx,
                            average_Zxx + sem_Zxx,
                            color="lightgray",
                            alpha=0.5,
                        )

                        plt.xlabel("Frequency [Hz]", fontdict={"size": 40})
                        plt.ylabel("PSD", fontdict={"size": 40})
                        # plt.ylim(1, 100)
                        plt.xticks(fontsize=35)
                        plt.yticks(fontsize=35)

                        # Plot the legend only for the first row "postop"
                        plt.legend(loc="upper right", edgecolor="black", fontsize=40)

            fig.suptitle(
                f"Power Spectrum sub-{sub}, {hem} hemisphere, fs = 250 Hz, filtered and artefact-free, reference {reference}",
                fontsize=55,
                y=1.02,
            )
            plt.show()

            io_externalized.save_fig_png_and_svg(
                path=figures_path,
                filename=f"One_plot_with_rings_Power_spectrum_{fname_extension}sub{sub}_{hem}_filtered_250Hz_resampled_artefact_free_reference_{reference}",
                figure=fig,
            )

    power_spectra_df = pd.DataFrame(power_spectra_dict)
    power_spectra_df.rename(
        index={
            0: "BIDS_id",
            1: "subject",
            2: "hemisphere",
            3: "subject_hemisphere",
            4: "channel",
            5: "filtered",
            6: "lfp_data",
            7: "frequencies",
            8: "times",
            9: "power",
            10: "power_average_over_time",
            11: "power_std",
            12: "power_sem",
            13: "sfreq",
            14: "reference",
        },
        inplace=True,
    )
    power_spectra_df = power_spectra_df.transpose()

    # save dataframe
    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    io_externalized.save_result_dataframe_as_pickle(
        data=power_spectra_df,
        filename=f"fourier_transform_externalized_{fname_extension}power_spectra_250Hz_artefact_free{reference_name}",
    )

    return power_spectra_df


def fooof_model_predefined(fooof_version: str):
    """
    Choose between version 1 and 2: "v1" or "v2"

    FOOOF settings v1:
        freq_range = [1, 95]  # frequency range to fit FOOOF model

        model = fooof.FOOOF(
            peak_width_limits=[2, 15.0],
            max_n_peaks=6,
            min_peak_height=0.2,
            peak_threshold=2.0,
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    FOOOF settings v2: -> this is Moritz setting, that he used for externalized recordings
        freq_range = [2, 45]
            - above 45 Hz is not of interest in this study (we only look at beta currently) and there is a lot of noise above 50 Hz,
            - especially amplification noise visible as a plateau in the high frequencies that overtakes the signal

        model = fooof.FOOOF(
            peak_width_limits=[1, 20.0],
            max_n_peaks=5,
            min_peak_height=0.1
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    """

    allowed = ["v1", "v2"]

    if fooof_version not in allowed:
        print(f"fooof_version input must be in {allowed}.")

    if fooof_version == "v1":
        freq_range = [1, 95]  # frequency range to fit FOOOF model

        model = fooof.FOOOF(
            peak_width_limits=[2, 15.0],
            max_n_peaks=6,
            min_peak_height=0.2,
            peak_threshold=2.0,
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    elif fooof_version == "v2":
        freq_range = [2, 45]  # frequency range to fit FOOOF model

        model = fooof.FOOOF(
            peak_width_limits=[3, 20.0],
            max_n_peaks=3,
            min_peak_height=0.1,
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    return {"freq_range": freq_range, "model": model}


def fooof_model_settings(
    fooof_version: str,
    bids_id: str,
    power_spectra_data: pd.DataFrame,
    filtered: str,
    monopolar_or_bipolar: str,
    new_reference: str,
    reference=None,
):
    """
    Input:
        - fooof_version: str, "v1", "v2"
        - bids_id: str, e.g. "L001"
        - power_spectra_data: Dataframe with power spectra
        - filtered: str, e.g. "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"
        - reference: str "bipolar_to_lowermost" or "no"

    Requirements of the dataframe input:
        - the dataframe has to be loaded first
        e.g. power_spectra_data = load_data.load_externalized_pickle(filename="externalized_power_spectra_250Hz_artefact_free")

        - first select only the rows with UNFILTERED OR NOTCH-FILTERED DATA
        e.g. power_spectra_data = power_spectra_data.loc[power_spectra_data.filtered == filtered]

        - if you want to run multiple subjects: loop over a list
        e.g. BIDS_id_unique = list(power_spectra_data.BIDS_id.unique())


    For a single subject, for each hemisphere seperately
        - run the FOOOF model, plot the raw Power spectra and the FOOOFed periodic power spectra
        - extract important variables and save them into a results dataframe
        - extract also which channel is used as common reference and save it into a dataframe

    """

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    monopolar_or_bipolar_dict = {"monopolar": "contact", "bipolar": "channel"}

    figures_path = find_folders.get_monopolar_project_path(
        folder="figures", sub=bids_id
    )

    common_reference_contacts = {}
    fooof_results_df = pd.DataFrame()

    # data only of one subject
    subject_data = power_spectra_data.loc[power_spectra_data.BIDS_id == bids_id]
    sub = subject_data.subject.values[0]

    for hem in HEMISPHERES:
        hem_data = subject_data.loc[subject_data.hemisphere == hem]
        contacts = list(
            hem_data[monopolar_or_bipolar_dict[monopolar_or_bipolar]].values
        )
        subject_hemisphere = f"{sub}_{hem}"

        for c, contact in enumerate(contacts):
            contact_data = hem_data.loc[
                hem_data[monopolar_or_bipolar_dict[monopolar_or_bipolar]] == contact
            ]

            if monopolar_or_bipolar == "monopolar":
                original_ch_name = contact_data.original_ch_name.values[0]

                # check if the power spectrum contains only 0 -> in case of a contact used as common reference
                if np.all(power_spectrum == 0):
                    common_reference_contacts[f"{sub}_{hem}_{contact}"] = [
                        bids_id,
                        sub,
                        hem,
                        contact,
                        original_ch_name,
                    ]
                    print(
                        f"Sub-{sub}, {hem}: contact {contact} is used as common reference."
                    )
                    continue

            # get the data to fit
            power_spectrum = contact_data.power_average_over_time.values[0]
            freqs = contact_data.frequencies.values[0]

            # whenever there are only 0 values, FOOOF won't work, so skip this channel
            if np.all(power_spectrum == 0):
                print(
                    f"Sub-{sub}, {hem}: contact {contact} is used as common reference."
                )
                continue

            ############ SET PLOT LAYOUT ############
            fig, ax = plt.subplots(4, 1, figsize=(7, 20))

            # Plot the unfiltered Power spectrum in first ax
            plot_spectrum(
                freqs, power_spectrum, log_freqs=False, log_powers=False, ax=ax[0]
            )
            ax[0].grid(False)

            ############ SET FOOOF MODEL ############

            model_version = fooof_model_predefined(fooof_version=fooof_version)
            freq_range = model_version["freq_range"]
            model = model_version["model"]

            # always fit a large Frequency band, later you can select Peaks within specific freq bands
            model.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

            # Plot an example power spectrum, with a model fit in second ax
            # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
            model.plot(ax=ax[1], plt_log=True)  # to evaluate the aperiodic component
            model.plot(
                ax=ax[2], plt_log=False
            )  # To see the periodic component better without log in frequency axis
            ax[1].grid(False)
            ax[2].grid(False)

            # check if fooof attributes are None:
            if model._peak_fit is None:
                print(f"subject {sub}, {hem}, {contact}: model peak fit is None.")
                continue

            if model._ap_fit is None:
                print(f"subject {sub}, {hem}, {contact}: model aperiodic fit is None.")
                continue

            # plot only the fooof spectrum of the periodic component
            fooof_power_spectrum = 10 ** (model._peak_fit + model._ap_fit) - (
                10**model._ap_fit
            )
            plot_spectrum(
                np.arange(1, (len(fooof_power_spectrum) + 1)),
                fooof_power_spectrum,
                log_freqs=False,
                log_powers=False,
                ax=ax[3],
            )
            # frequencies: 1-95 Hz with 1 Hz resolution

            # titles
            fig.suptitle(
                f"sub {sub}, {hem} hemisphere, contact: {contact}, reference {reference}",
                fontsize=25,
            )

            ax[0].set_title(
                f"{filtered}, artefact-free power spectrum",
                fontsize=20,
                y=0.97,
                pad=-20,
            )
            ax[3].set_title("power spectrum of periodic component", fontsize=20)

            # mark beta band
            x1 = 13
            x2 = 35
            ax[3].axvspan(x1, x2, color="whitesmoke")
            ax[3].grid(False)

            fig.tight_layout()

            # save figures
            if monopolar_or_bipolar == "monopolar":
                fname_extension = ""
            elif monopolar_or_bipolar == "bipolar":
                if new_reference == "BSSU":
                    fname_extension = "BSSU_"
                elif new_reference == "one_to_zero_two_to_three":
                    fname_extension = "one_to_zero_two_to_three_"

            io_externalized.save_fig_png_and_svg(
                path=figures_path,
                filename=f"fooof_externalized_{fname_extension}sub{sub}_{hem}_{contact}_250Hz_clean_{filtered}{reference_name}_{fooof_version}",
                figure=fig,
            )

            ############ SAVE APERIODIC PARAMETERS ############
            # goodness of fit
            err = model.get_params("error")
            r_sq = model.r_squared_

            # aperiodic components
            exp = model.get_params("aperiodic_params", "exponent")
            offset = model.get_params("aperiodic_params", "offset")

            # periodic component
            log_power_fooof_periodic_plus_aperiodic = (
                model._peak_fit + model._ap_fit
            )  # periodic+aperiodic component in log Power axis
            fooof_periodic_component = (
                model._peak_fit
            )  # just periodic component, flattened spectrum

            ############ SAVE ALL PEAKS IN ALPHA; HIGH AND LOW BETA ############

            number_peaks = model.n_peaks_

            # get the highest Peak of each frequency band as an array: CF center frequency, Power, BandWidth
            alpha_peak = fooof.analysis.get_band_peak_fm(
                model, band=(8.0, 12.0), select_highest=True, attribute="peak_params"
            )

            low_beta_peak = fooof.analysis.get_band_peak_fm(
                model,
                band=(13.0, 20.0),
                select_highest=True,
                attribute="peak_params",
            )

            high_beta_peak = fooof.analysis.get_band_peak_fm(
                model,
                band=(21.0, 35.0),
                select_highest=True,
                attribute="peak_params",
            )

            beta_peak = fooof.analysis.get_band_peak_fm(
                model,
                band=(13.0, 35.0),
                select_highest=True,
                attribute="peak_params",
            )

            gamma_peak = fooof.analysis.get_band_peak_fm(
                model,
                band=(60.0, 90.0),
                select_highest=True,
                attribute="peak_params",
            )

            # save all results in dictionary
            fooof_results = {
                "reference": [reference],
                "filtered": [filtered],
                "BIDS_id": [bids_id],
                "subject": [sub],
                "hemisphere": [hem],
                "subject_hemisphere": [subject_hemisphere],
                "contact": [contact],
                "fooof_error": [err],
                "fooof_r_sq": [r_sq],
                "fooof_exponent": [exp],
                "fooof_offset": [offset],
                "fooof_power_spectrum": [fooof_power_spectrum],
                "periodic_plus_aperiodic_power_log": [
                    log_power_fooof_periodic_plus_aperiodic
                ],
                "fooof_periodic_flat": [fooof_periodic_component],
                "fooof_number_peaks": [number_peaks],
                "alpha_peak_CF_power_bandWidth": [alpha_peak],
                "low_beta_peak_CF_power_bandWidth": [low_beta_peak],
                "high_beta_peak_CF_power_bandWidth": [high_beta_peak],
                "beta_peak_CF_power_bandWidth": [beta_peak],
                "gamma_peak_CF_power_bandWidth": [gamma_peak],
            }

            fooof_results_single = pd.DataFrame(fooof_results)
            fooof_results_df = pd.concat(
                [fooof_results_df, fooof_results_single], ignore_index=True
            )

    if monopolar_or_bipolar == "monopolar":
        # bids_id, sub, hem, contact, original_ch_name
        common_reference_contacts_columns = [
            "BIDS_id",
            "subject",
            "hemisphere",
            "contact",
        ]
        common_reference_contacts_df = pd.DataFrame.from_dict(
            common_reference_contacts,
            orient="index",
            columns=common_reference_contacts_columns,
        )

    elif monopolar_or_bipolar == "bipolar":
        common_reference_contacts_df = "no common reference contacts"

    return {
        "fooof_results_df": fooof_results_df,
        "common_reference_contacts_df": common_reference_contacts_df,
    }


def externalized_fooof_fit(fooof_version: str, filtered: str, reference=None):
    """
    Input:
        - fooof_version: str "v1" or "v2"
        - filtered: str, "unfiltered", "only_high_pass_filtered"
        - reference: str "bipolar_to_lowermost" or "no"

    Load the Power Spectra data
        - externalized_power_spectra_250Hz_artefact_free.pickle
        - resampled sfreq=250 Hz
        - artefact-free

        2 versions:
        - unfiltered Power Spectra
        - only high-pass filtered Power Spectra

    1) First set and fit a FOOOF model without a knee -> within a frequency range from 1-95 Hz (broad frequency range for fitting the aperiodic component)
        - peak_width_limits=[2, 15.0],  # must be a list, low limit should be more than twice as frequency resolution, usually not more than 15Hz bw
        - max_n_peaks=6,                # 4, 5 sometimes misses important peaks, 6 better even though there might be more false positives in high frequencies
        - min_peak_height=0.2,          # 0.2 detects false positives in gamma but better than 0.35 missing relevant peaks in low frequencies
        - peak_threshold=2.0,           # default 2.0, lower if necessary to detect peaks more sensitively
        - aperiodic_mode="fixed",       # fitting without knee component, because there are no knees found so far in the STN
        - verbose=True,

        frequency range for parameterization: 1-95 Hz

        plot a figure with the raw Power spectrum and the fitted model

    2) save figure into figure folder of each subject:
        figure filename: fooof_externalized_sub{subject}_{hemisphere}_{chan}_250Hz_clean.png

    3) Extract following parameters and save as columns into DF
        - 0: "subject_hemisphere",
        - 1: "subject",
        - 2: "hemisphere",
        - 3: "BIDS_id",
        - 4: "original_ch_name",
        - 5: "contact",
        - 6: "fooof_error",
        - 7: "fooof_r_sq",
        - 8: "fooof_exponent",
        - 9: "fooof_offset",
        - 10: "fooof_power_spectrum", # with 95 values, 1 Hz frequency resolution, so 1-95 Hz
        - 11: "periodic_plus_aperiodic_power_log",
        - 12: "fooof_periodic_flat",
        - 13: "fooof_number_peaks",
        - 14: "alpha_peak_CF_power_bandWidth",
        - 15: "low_beta_peak_CF_power_bandWidth",
        - 16: "high_beta_peak_CF_power_bandWidth",
        - 17: "beta_peak_CF_power_bandWidth",
        - 18: "gamma_peak_CF_power_bandWidth",


    5) save Dataframe into results folder of each subject
        - filename: "fooof_externalized_sub{subject}.json"

    """

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    fooof_results_df = pd.DataFrame()
    common_reference_group_DF = pd.DataFrame()

    power_spectra_data = io_externalized.load_externalized_pickle(
        filename="externalized_power_spectra_250Hz_artefact_free", reference=reference
    )

    # first select only the rows with UNFILTERED OR ONLY HIGH-PASS-FILTERED DATA
    power_spectra_data = power_spectra_data.loc[power_spectra_data.filtered == filtered]

    BIDS_id_unique = list(power_spectra_data.BIDS_id.unique())

    for bids_id in BIDS_id_unique:
        # fit a FOOOF model, plot the spectra and save the results
        bids_id_fooof_result = fooof_model_settings(
            fooof_version=fooof_version,
            bids_id=bids_id,
            power_spectra_data=power_spectra_data,
            filtered=filtered,
            new_reference="no",
            reference=reference,
        )

        fooof_results_bids_id_df = bids_id_fooof_result["fooof_results_df"]
        fooof_results_df = pd.concat(
            [fooof_results_df, fooof_results_bids_id_df], ignore_index=True
        )

        common_reference_bids_id_DF = bids_id_fooof_result[
            "common_reference_contacts_df"
        ]
        common_reference_group_DF = pd.concat(
            [common_reference_group_DF, common_reference_bids_id_DF], ignore_index=True
        )

    # save DF as pickle
    io_externalized.save_result_dataframe_as_pickle(
        data=fooof_results_df,
        filename=f"fooof_externalized_group_{filtered}{reference_name}_{fooof_version}",
    )
    io_externalized.save_result_dataframe_as_pickle(
        data=common_reference_group_DF,
        filename=f"externalized_contacts_common_reference{reference_name}_{fooof_version}",
    )

    return fooof_results_df


def externalized_fooof_fit_2(
    fooof_version: str,
    filtered: str,
    monopolar_or_bipolar: str,
    new_reference: str,
    reference=None,
):
    """
    Input:
        - fooof_version: str "v1" or "v2"
        - filtered: str, "unfiltered", "only_high_pass_filtered"
        - reference: str "bipolar_to_lowermost" or "no"
        - new_reference: "bssu", "one_to_zero_two_to_three"

    Load the Power Spectra data
        - externalized_power_spectra_250Hz_artefact_free.pickle
        - resampled sfreq=250 Hz
        - artefact-free

        2 versions:
        - unfiltered Power Spectra
        - only high-pass filtered Power Spectra

    1) First set and fit a FOOOF model without a knee -> within a frequency range from 1-95 Hz (broad frequency range for fitting the aperiodic component)
        - peak_width_limits=[2, 15.0],  # must be a list, low limit should be more than twice as frequency resolution, usually not more than 15Hz bw
        - max_n_peaks=6,                # 4, 5 sometimes misses important peaks, 6 better even though there might be more false positives in high frequencies
        - min_peak_height=0.2,          # 0.2 detects false positives in gamma but better than 0.35 missing relevant peaks in low frequencies
        - peak_threshold=2.0,           # default 2.0, lower if necessary to detect peaks more sensitively
        - aperiodic_mode="fixed",       # fitting without knee component, because there are no knees found so far in the STN
        - verbose=True,

        frequency range for parameterization: 1-95 Hz

        plot a figure with the raw Power spectrum and the fitted model

    2) save figure into figure folder of each subject:
        figure filename: fooof_externalized_sub{subject}_{hemisphere}_{chan}_250Hz_clean.png

    3) Extract following parameters and save as columns into DF
        - 0: "subject_hemisphere",
        - 1: "subject",
        - 2: "hemisphere",
        - 3: "BIDS_id",
        - 4: "original_ch_name",
        - 5: "contact",
        - 6: "fooof_error",
        - 7: "fooof_r_sq",
        - 8: "fooof_exponent",
        - 9: "fooof_offset",
        - 10: "fooof_power_spectrum", # with 95 values, 1 Hz frequency resolution, so 1-95 Hz
        - 11: "periodic_plus_aperiodic_power_log",
        - 12: "fooof_periodic_flat",
        - 13: "fooof_number_peaks",
        - 14: "alpha_peak_CF_power_bandWidth",
        - 15: "low_beta_peak_CF_power_bandWidth",
        - 16: "high_beta_peak_CF_power_bandWidth",
        - 17: "beta_peak_CF_power_bandWidth",
        - 18: "gamma_peak_CF_power_bandWidth",


    5) save Dataframe into results folder of each subject
        - filename: "fooof_externalized_sub{subject}.json"

    """

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    fooof_results_df = pd.DataFrame()
    common_reference_group_DF = pd.DataFrame()

    if monopolar_or_bipolar == "bipolar":
        if new_reference == "bssu":
            fname_extension = "BSSU_"
        elif new_reference == "one_to_zero_two_to_three":
            fname_extension = "one_to_zero_two_to_three_"

        power_spectra_data = io_externalized.load_externalized_pickle(
            filename=f"fourier_transform_externalized_{fname_extension}power_spectra_250Hz_artefact_free",
            reference=reference,
        )

    elif monopolar_or_bipolar == "monopolar":
        power_spectra_data = io_externalized.load_externalized_pickle(
            filename="externalized_power_spectra_250Hz_artefact_free",
            reference=reference,
        )  # make sure to use this one and not "fourier_transform_externalized..." because original channel names will be needed for the FOOOF function above
        fname_extension = ""

    # first select only the rows with UNFILTERED OR ONLY HIGH-PASS-FILTERED DATA
    power_spectra_data = power_spectra_data.loc[power_spectra_data.filtered == filtered]

    BIDS_id_unique = list(power_spectra_data.BIDS_id.unique())

    for bids_id in BIDS_id_unique:
        # fit a FOOOF model, plot the spectra and save the results
        bids_id_fooof_result = fooof_model_settings(
            fooof_version=fooof_version,
            bids_id=bids_id,
            power_spectra_data=power_spectra_data,
            filtered=filtered,
            monopolar_or_bipolar=monopolar_or_bipolar,
            new_reference=new_reference,
            reference=reference,
        )

        fooof_results_bids_id_df = bids_id_fooof_result["fooof_results_df"]
        fooof_results_df = pd.concat(
            [fooof_results_df, fooof_results_bids_id_df], ignore_index=True
        )

        if monopolar_or_bipolar == "monopolar":
            common_reference_bids_id_DF = bids_id_fooof_result[
                "common_reference_contacts_df"
            ]
            common_reference_group_DF = pd.concat(
                [common_reference_group_DF, common_reference_bids_id_DF],
                ignore_index=True,
            )

    # save DF as pickle
    io_externalized.save_result_dataframe_as_pickle(
        data=fooof_results_df,
        filename=f"fooof_externalized_group_{fname_extension}{filtered}{reference_name}_{fooof_version}",
    )
    if monopolar_or_bipolar == "monopolar":
        io_externalized.save_result_dataframe_as_pickle(
            data=common_reference_group_DF,
            filename=f"externalized_contacts_common_reference{reference_name}_{fooof_version}",
        )

    return fooof_results_df


def calculate_periodic_beta_power(
    fooof_version: str, filtered: str, new_reference: str, reference=None
):
    """
    Input:
        - fooof_version: str "v1" or "v2"
        - filtered: str, "unfiltered", "only_high_pass_filtered"
        - reference: str "bipolar_to_lowermost" or "no"
        - new_reference: "no", "one_to_zero_two_to_three"


    Load the Fooof group data:
        - fooof_externalized_group_notch-filtered.pickle

    1) Add a column with the average beta power 13-35 Hz from the fooof_power_spectrum

    2) Create 2 versions of data with beta ranks and normalized beta values relative to the maximum per lead
        - directional contacts (rank 1-6)
        - all contacts (rank 1-8)


    Save both dataframes as pickle files:
        - fooof_externalized_beta_ranks_all_contacts_{filtered}{reference_name}_{fooof_version}.pickle
        - fooof_externalized_beta_ranks_directional_contacts_{filtered}{reference_name}_{fooof_version}.pickle


    """

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    if new_reference == "one_to_zero_two_to_three":
        fname_extension = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        fname_extension = ""

    # dataframes with beta ranks:
    beta_rank_all_contacts = pd.DataFrame()
    beta_rank_directional_contacts = pd.DataFrame()

    group_fooof_data = io_externalized.load_externalized_pickle(
        filename=f"fooof_externalized_group_{fname_extension}{filtered}",
        fooof_version=fooof_version,
        reference=reference,
    )

    # new column beta_average
    group_fooof_copy = group_fooof_data.copy()
    group_fooof_copy["beta_average"] = group_fooof_copy["fooof_power_spectrum"]
    group_fooof_copy["beta_average"] = group_fooof_copy["beta_average"].apply(
        lambda row: np.mean(row[13:36])
    )

    ########## RANK ALL CONTACTS ##########
    # for each STN rank the beta average -> rank 1-8
    sub_hem_unique = list(group_fooof_copy.subject_hemisphere.unique())

    for stn in sub_hem_unique:
        stn_data = group_fooof_copy.loc[group_fooof_copy.subject_hemisphere == stn]
        stn_data_copy = stn_data.copy()
        stn_data_copy["beta_rank"] = stn_data_copy["beta_average"].rank(
            ascending=False
        )  # rank 1-8

        # relative to maximum value
        max_value = stn_data_copy["beta_average"].max()
        stn_data_copy["beta_relative_to_max"] = (
            stn_data_copy["beta_average"] / max_value
        )

        # cluster values into 3 categories: <40%, 40-70% and >70%
        stn_data_copy["beta_cluster"] = stn_data_copy["beta_relative_to_max"].apply(
            helpers.assign_cluster
        )

        # save to the beta rank dataframe
        beta_rank_all_contacts = pd.concat([beta_rank_all_contacts, stn_data_copy])

    # save DF
    io_externalized.save_result_dataframe_as_pickle(
        data=beta_rank_all_contacts,
        filename=f"fooof_externalized_beta_ranks_all_contacts_{fname_extension}{filtered}{reference_name}_{fooof_version}",
    )

    ########## RANK DIRECTIONAL CONTACTS ONLY ##########
    # select only directional contacts from each STN and rank beta average
    directional_data = group_fooof_copy[
        group_fooof_copy["contact"].isin(DIRECTIONAL_CONTACTS)
    ]
    for stn in sub_hem_unique:
        stn_directional = directional_data.loc[
            directional_data.subject_hemisphere == stn
        ]
        stn_directional_copy = stn_directional.copy()
        stn_directional_copy["beta_rank"] = stn_directional_copy["beta_average"].rank(
            ascending=False
        )  # rank 1-6

        # relative to maximum value
        max_value_dir = stn_directional_copy["beta_average"].max()
        stn_directional_copy["beta_relative_to_max"] = (
            stn_directional_copy["beta_average"] / max_value_dir
        )

        # cluster values into 3 categories: <40%, 40-70% and >70%
        stn_directional_copy["beta_cluster"] = stn_directional_copy[
            "beta_relative_to_max"
        ].apply(io_externalized.assign_cluster)

        # save to the beta rank dataframe
        beta_rank_directional_contacts = pd.concat(
            [beta_rank_directional_contacts, stn_directional_copy]
        )

    # save DF
    io_externalized.save_result_dataframe_as_pickle(
        data=beta_rank_directional_contacts,
        filename=f"fooof_externalized_beta_ranks_directional_contacts_{fname_extension}{filtered}{reference_name}_{fooof_version}",
    )

    return {
        "beta_rank_all_contacts": beta_rank_all_contacts,
        "beta_rank_directional_contacts": beta_rank_directional_contacts,
    }


def SSD_filter_externalized(
    bids_id: str, sub: str, hemisphere: str, fs: int, directional_signals=None
):
    """
    Input:
        - fs: sampling frequency of the signal
        - directional_signals: must be a 2D array with 6 signals of the directional channels
            e.g. of one hemisphere:
            directional_channels = [f"LFP_{hem[0]}_02_STN_MT",
                                f"LFP_{hem[0]}_03_STN_MT",
                                f"LFP_{hem[0]}_04_STN_MT",
                                f"LFP_{hem[0]}_05_STN_MT",
                                f"LFP_{hem[0]}_06_STN_MT",
                                f"LFP_{hem[0]}_07_STN_MT"]

    This function will follow these steps:

        - filter SSD in the beta range (13-35 Hz)
        - plot a figure with a plot of the Raw Power Spectra of all channels and a plot with the first component Power spectrum
        - return the results: ssd_filt_data, ssd_pattern, ssd_eigvals
        - save a dataframe with all relevant values + beta_ranks (ssd_pattern ranked) + beta_relative_to_max (relative to ssd_pattern maximum)

    """

    SSD_result = {}

    figures_path = find_folders.get_monopolar_project_path(
        folder="figures", sub=bids_id
    )

    f_range = 13, 35

    (ssd_filt_data, ssd_pattern, ssd_eigvals) = feats_ssd.get_SSD_component(
        data_2d=directional_signals,
        fband_interest=f_range,
        s_rate=fs,
        use_freqBand_filtered=True,
        return_comp_n=0,
    )

    # check if length of ssd pattern and ssd eigenvalues is 6, if  < 6, a directional contact was used as common reference, exclude that patient.
    if len(ssd_eigvals) < 6:
        print(
            f"Sub-{sub}, {hemisphere} hemisphere, only has {len(ssd_pattern)} Eigenvalues."
        )

    # Figure with 2 subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    ########### PLOT 1: Power Spectra of all directional channels without SSD ###########
    for c, chan in enumerate(DIRECTIONAL_CONTACTS):
        if c == 5 and len(ssd_eigvals) < 6:
            ssd_pattern_chan = 0.0
            ssd_eigvals_chan = 0.0

        else:
            ssd_pattern_chan = abs(
                ssd_pattern[0][c]
            )  # first component only, absolute values because the direction of the vector can be positive or negative
            ssd_eigvals_chan = ssd_eigvals[c]

        # filter to plot
        notch_filtered = externalized_lfp_preprocessing.notch_filter_externalized(
            fs=fs, signal=directional_signals[c]
        )
        band_pass_and_notch = (
            externalized_lfp_preprocessing.band_pass_filter_externalized(
                fs=fs, signal=notch_filtered
            )
        )

        ######### short time fourier transform to calculate PSD #########
        window_length = int(fs)  # 1 second window length
        overlap = window_length // 4  # 25% overlap

        # Calculate the short-time Fourier transform (STFT) using Hann window
        window = hann(window_length, sym=False)

        frequencies, times, Zxx = scipy.signal.spectrogram(
            band_pass_and_notch,
            fs=fs,
            window=window,
            noverlap=overlap,
            scaling="density",
            mode="psd",
            axis=0,
        )

        # average PSD across duration of the recording
        average_Zxx = np.mean(Zxx, axis=1)
        std_Zxx = np.std(Zxx, axis=1)
        sem_Zxx = std_Zxx / np.sqrt(Zxx.shape[1])

        axes[0].plot(
            frequencies,
            average_Zxx,
            label=f"Contact{chan}_ssd_pattern_{ssd_pattern_chan}",
        )
        axes[0].fill_between(
            frequencies,
            average_Zxx - sem_Zxx,
            average_Zxx + sem_Zxx,
            color="lightgray",
            alpha=0.5,
        )
        axes[0].set_title(
            f"Power Spectra: sub{bids_id} {hemisphere} hemisphere", fontsize=20
        )
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Frequency [Hz]", fontsize=20)
        axes[0].set_ylabel("PSD [uV^2/Hz]", fontsize=20)
        axes[0].axvline(x=13, color="black", linestyle="--")
        axes[0].axvline(x=35, color="black", linestyle="--")
        axes[0].legend()

        # save result for each contact
        sub_hem = f"{sub}_{hemisphere}"
        SSD_result[chan] = [
            bids_id,
            sub,
            hemisphere,
            sub_hem,
            chan,
            ssd_filt_data,
            ssd_pattern_chan,
            ssd_eigvals_chan,
        ]

    ########### PLOT 2: Power Spectrum of the first SSD component ###########
    # use Welch to transform the time domain data to frequency domain data
    f, psd = signal.welch(
        ssd_filt_data, axis=-1, nperseg=fs, fs=fs
    )  # ssd_filt_data is only one array with the PSD of the FIRST component

    # plot the first component Power spectrum of each recording group
    axes[1].plot(f, psd)

    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Frequency", fontsize=20)
    axes[1].set_ylim(-0.005, 0.12)
    axes[1].set_title(f"Power Spectrum beta first component (13-35 Hz)", fontsize=20)
    axes[1].axvline(x=13, color="black", linestyle="--")
    axes[1].axvline(x=35, color="black", linestyle="--")

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            figures_path,
            f"sub-{bids_id}_{hemisphere}_externalized_directional_SSD_beta_first_component_Power_spectrum.png",
        ),
        bbox_inches="tight",
    )

    plt.close()

    # save the dataframe
    SSD_result_columns = [
        "BIDS_id",
        "subject",
        "hemisphere",
        "subject_hemisphere",
        "contact",
        "ssd_filtered_timedomain",
        "ssd_pattern",
        "ssd_eigvals",
    ]

    SSD_result_dataframe = pd.DataFrame.from_dict(
        SSD_result, orient="index", columns=SSD_result_columns
    )

    # add a column with beta_ranks and normalized beta values relative to max
    SSD_result_dataframe_copy = SSD_result_dataframe.copy()
    SSD_result_dataframe_copy["beta_rank"] = SSD_result_dataframe["ssd_pattern"].rank(
        ascending=False
    )  # rank 1-6

    # normalize to maximum value
    max_value_dir = SSD_result_dataframe_copy["ssd_pattern"].max()
    SSD_result_dataframe_copy["beta_relative_to_max"] = (
        SSD_result_dataframe_copy["ssd_pattern"] / max_value_dir
    )

    # cluster values into 3 categories: <40%, 40-70% and >70%
    SSD_result_dataframe_copy["beta_cluster"] = SSD_result_dataframe_copy[
        "ssd_pattern"
    ].apply(helpers.assign_cluster)

    return {
        "ssd_filt_data": ssd_filt_data,
        "ssd_pattern": ssd_pattern,
        "ssd_eigvals": ssd_eigvals,
        "SSD_result_dataframe_copy": SSD_result_dataframe_copy,
    }


def directional_SSD_externalized(reference=None):
    """
    Input:
        - reference: str "bipolar_to_lowermost" or "no"

    STEPS:
        - Load the preprocessed data (cropped 2 min, downsampled 250 Hz, artefact-free)
        - for each subject and each hemisphere apply the SSD
        - concatenate all Dataframes to one: SSD_group_result
        - save the results Dataframe:
            SSD_directional_externalized_channels{reference_name}.pickle


    """
    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    SSD_group_result = pd.DataFrame()

    artefact_free_lfp = io_externalized.load_externalized_pickle(
        filename="externalized_preprocessed_data_artefact_free", reference=reference
    )

    BIDS_id_unique = list(artefact_free_lfp.BIDS_id.unique())

    for bids_id in BIDS_id_unique:
        # data only of one subject
        subject_data = artefact_free_lfp.loc[artefact_free_lfp.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        # pick the directional channels of both hemispheres
        for h, hem in enumerate(HEMISPHERES):
            directional_channels = [
                f"LFP_{hem[0]}_02_STN_MT",
                f"LFP_{hem[0]}_03_STN_MT",
                f"LFP_{hem[0]}_04_STN_MT",
                f"LFP_{hem[0]}_05_STN_MT",
                f"LFP_{hem[0]}_06_STN_MT",
                f"LFP_{hem[0]}_07_STN_MT",
            ]

            channels_in_hemisphere = subject_data.loc[
                subject_data.original_ch_name.isin(directional_channels)
            ]

            # get unfiltered 250 Hz downsampled, artefact-free LFP from all 6 directional channels from one hemisphere
            unfiltered_data = channels_in_hemisphere.lfp_resampled_250Hz.values
            unfiltered_data = np.array(
                unfiltered_data.tolist()
            )  # transform to 2D array, so it fits the requirements for feats_ssd

            sfreq = 250

            # apply the SSD filter, plot the Power Spectra
            SSD_result_single = SSD_filter_externalized(
                bids_id=bids_id,
                sub=sub,
                hemisphere=hem,
                fs=sfreq,
                directional_signals=unfiltered_data,
            )
            SSD_result_dataframe_single = SSD_result_single["SSD_result_dataframe_copy"]

            # append the result to the group result Dataframe
            SSD_group_result = pd.concat(
                [SSD_group_result, SSD_result_dataframe_single], ignore_index=True
            )

    # save the results dataframe
    io_externalized.save_result_dataframe_as_pickle(
        data=SSD_group_result,
        filename=f"SSD_directional_externalized_channels{reference_name}",
    )

    return SSD_group_result
