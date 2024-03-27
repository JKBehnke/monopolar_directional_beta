""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram

import json
import h5py
from pathlib import Path
import mne
import mne_bids
from mne_bids import BIDSPath, inspect_dataset, mark_channels


import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..utils import tmsi_poly5reader

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

HEMISPHERES = ["Right", "Left"]

# list of subjects with no BIDS transformation yet -> load these via poly5reader instead of BIDS
SUBJECTS_NO_BIDS = ["24", "28", "29", "48", "49", "56"]


######## load data ########
def load_patient_metadata_externalized():
    """
    Input:


    Load the file: movement_artifacts_from_raw_time_series_band-pass.pickle  # always band-pass because filtered signal is easier to find movement artifacts
    from the group result folder

    """

    # find the path to the results folder
    path = find_folders.get_monopolar_project_path(folder="data")

    # create filename
    filename = "patient_metadata.xlsx"

    filepath = os.path.join(path, filename)

    # load the file
    data = pd.read_excel(
        filepath, keep_default_na=True, sheet_name="patient_metadata"
    )  # all sheets are loaded
    print("Excel file loaded: ", filename, "\nloaded from: ", path)

    return data


def load_excel_data(filename: str):
    """
    Input:
        - filename: "patient_metadata", "movement_artefacts", "ecg_artifacts"

    """

    patient_metadata_sheet = ["patient_metadata", "movement_artefacts"]

    # find the path to the results folder
    path = find_folders.get_monopolar_project_path(folder="data")

    # create filename
    f_name = f"{filename}.xlsx"

    if filename in patient_metadata_sheet:
        sheet_name = "patient_metadata"
    
    elif filename == "ecg_artifacts":
        sheet_name = "ecg_artifacts"

    filepath = os.path.join(path, f_name)

    # load the file
    data = pd.read_excel(filepath, keep_default_na=True, sheet_name=sheet_name)
    print("Excel file loaded: ", f_name, "\nloaded from: ", path)

    return data


def load_externalized_Poly5_files(sub: str):
    """
    Input:
        - sub: str e.g. "24"

    filepath: '/Users/jenniferbehnke/Dropbox/work/ResearchProjects/Monopolar_power_estimation/data/externalized_lfp/
    -> subject path depending on the externalized patient ID

    load the correct Poly5 file of the input subject
    - externalized LFP
    - Med Off
    - Stim Off
    - Rest


    """

    subject_folder_path = find_folders.get_monopolar_project_path(
        folder="data_sub", sub=sub
    )

    # check if there is a .Poly5 file
    files = os.listdir(subject_folder_path)
    for f in files:
        if f.endswith(".Poly5"):
            filename = f

    filepath = os.path.join(subject_folder_path, filename)

    # load the Poly5 file
    raw_file = tmsi_poly5reader.Poly5Reader(filepath)
    raw_file = raw_file.read_data_MNE()
    raw_file.load_data()

    return raw_file


def load_BIDS_externalized_vhdr_files(sub: str):
    """

    BIDS_root: '/Users/jenniferbehnke/OneDrive - Charité - Universitätsmedizin Berlin/BIDS_01_Berlin_Neurophys/rawdata/'
    -> subject path depending on the externalized patient ID

    load the correct vhdr file of the input subject

    BIDS structure:
    - ECoG + LFP: sub-EL... > "ses-EcogLfpMedOff01" > "ieeg" > filename containing Rest, StimOff, run-1, endswith .vhdr
    - only LFP: sub-L... > "ses-LfpMedOff01"  > "ieeg" > filename containing Rest, StimOff, run-1, endswith .vhdr

        EL session = "EcogLfpMedOff01"
        L session = "LfpMedOff01"

        task = "Rest"
        aquisition = "StimOff"
        run = "1"
        datatype = "ieeg"
        extension = ".vhdr"
        suffix = "ieeg"


    """

    # get the BIDS key from the subject
    local_path = find_folders.get_monopolar_project_path(folder="data")
    patient_metadata = pd.read_excel(
        os.path.join(local_path, "patient_metadata.xlsx"),
        keep_default_na=True,
        sheet_name="patient_metadata",
    )

    # change column "patient_ID" to strings
    patient_metadata["patient_ID"] = patient_metadata.patient_ID.astype(str)
    sub_BIDS_ID = patient_metadata.loc[
        patient_metadata.patient_ID == sub
    ]  # row of subject

    # check if the subject has a BIDS key
    if pd.isna(sub_BIDS_ID.BIDS_key.values[0]):
        print(f"The subject {sub} has no BIDS key yet.")
        return "no BIDS key"

    # only if there is a BIDS key.
    else:
        sub_BIDS_ID = sub_BIDS_ID.BIDS_key.values[0]

        raw_data_folder = find_folders.get_onedrive_path_externalized_bids(
            folder="rawdata"
        )
        bids_root = raw_data_folder
        bids_path = BIDSPath(root=bids_root)

        ########## UPDATE BIDS PATH ##########
        # check if BIDS session directory contains "Dys"
        sessions = os.listdir(os.path.join(raw_data_folder, f"sub-{sub_BIDS_ID}"))
        dys_list = []
        for s in sessions:
            if "MedOffDys" in s:
                dys_list.append("Dys")

        if len(dys_list) == 0:
            dys = ""
            dopa = ""

        else:
            dys = "Dys"
            if sub_BIDS_ID == "EL016":
                dopa = "DopaPre"
            else:
                dopa = "Dopa00"

        # check if the BIDS key has a sub-EL or sub-L folder
        session = f"LfpMedOff{dys}01"

        if "EL" in sub_BIDS_ID:
            session = f"EcogLfpMedOff{dys}01"

        task = "Rest"
        acquisition = f"StimOff{dopa}"

        run = "1"
        if sub_BIDS_ID == "L014":
            run = "2"

        datatype = "ieeg"
        extension = ".vhdr"
        suffix = "ieeg"

        bids_path.update(
            subject=sub_BIDS_ID,
            session=session,
            task=task,
            acquisition=acquisition,
            run=run,
            datatype=datatype,
            extension=extension,
            suffix=suffix,
        )

        # inspect_dataset(bids_path, l_freq=5.0, h_freq=95.0)

        data = mne_bids.read_raw_bids(
            bids_path=bids_path
        )  # datatype: mne.io.brainvision.brainvision.RawBrainVision
        # to work with the data, load the data
        data.load_data()

        return data
    
def load_sub_result_pickle(bids_id: str, filename: str):
    """

    Input:
        - bids_id: str, e.g. "sub-EL001"
        - filename: str, e.g."power_spectra_BSSU_externalized_20sec"
    """
    results_path = find_folders.get_monopolar_project_path(folder="results", sub=bids_id)

    pickle_filename = f"{filename}.pickle"

    filepath = os.path.join(results_path, pickle_filename)

    # load the file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data


def load_externalized_pickle(filename: str, fooof_version=None, reference=None):
    """
    Pickle files in the group results folder of the monopolar estimation project
    Input:
        - filename: str, must be in
            ["externalized_preprocessed_data",
            "externalized_recording_info_original",
            "mne_objects_cropped_2_min",
            "externalized_preprocessed_data_artefact_free",
            "externalized_power_spectra_250Hz_artefact_free",
            "externalized_contacts_common_reference",
            "fooof_externalized_group",
            "fooof_externalized_group_notch-filtered",
            "fooof_externalized_group_only_high_pass_filtered"
            "fooof_externalized_beta_ranks_all_contacts",
            "fooof_externalized_beta_ranks_directional_contacts",
            "SSD_directional_externalized_channels"
            "externalized_directional_bssu_channels",
            "fooof_externalized_group_BSSU_only_high_pass_filtered",
            "power_spectra_BSSU_externalized_20sec_group_{filtered}"
            "power_spectra_BSSU_externalized_{filtered}_2min_and_{sec_per_epoch}sec"

            ]

        - reference: "bipolar_to_lowermost" or "no"
    """

    group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    # create filename and filepath
    pickle_filename = f"{filename}{reference_name}.pickle"

    if fooof_version in ["v1", "v2"]:
        pickle_filename = f"{filename}{reference_name}_{fooof_version}.pickle"

    filepath = os.path.join(group_results_path, pickle_filename)

    # load the file
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_data_to_weight(data_type: str):
    """
    Input:
        - data_type: "fooof", "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"

    """

    # load the correct data type
    if data_type == "fooof":
        loaded_data = load_externalized_pickle(
            filename="fooof_externalized_group_BSSU_only_high_pass_filtered",
            fooof_version="v2",
            reference="bipolar_to_lowermost",
        )

        contact_channel = "contact"
        spectra_column = "fooof_power_spectrum"

    elif data_type == "notch_and_band_pass_filtered":
        loaded_data = load_externalized_pickle(
            filename="fourier_transform_externalized_BSSU_power_spectra_250Hz_artefact_free",
            reference="bipolar_to_lowermost",
        )
        loaded_data = loaded_data.loc[
            loaded_data.filtered == "notch_and_band_pass_filtered"
        ]
        contact_channel = "channel"
        spectra_column = "power_average_over_time"

    elif data_type == "unfiltered":
        loaded_data = load_externalized_pickle(
            filename="fourier_transform_externalized_BSSU_power_spectra_250Hz_artefact_free",
            reference="bipolar_to_lowermost",
        )
        loaded_data = loaded_data.loc[loaded_data.filtered == "unfiltered"]
        contact_channel = "channel"
        spectra_column = "power_average_over_time"

    elif data_type == "only_high_pass_filtered":
        loaded_data = load_externalized_pickle(
            filename="fourier_transform_externalized_BSSU_power_spectra_250Hz_artefact_free",
            reference="bipolar_to_lowermost",
        )
        loaded_data = loaded_data.loc[loaded_data.filtered == "only_high_pass_filtered"]
        contact_channel = "channel"
        spectra_column = "power_average_over_time"

    # rename columns
    loaded_data.rename(columns={contact_channel: "bipolar_channel"}, inplace=True)

    return {"loaded_data": loaded_data, "spectra": spectra_column}


def load_patient_data(patient: str):
    """
    Input:
        - patient: str, e.g. "25"

    First check, if the patient is in the list with no BIDS data yet.
    If no BIDS data exists
        - load data with Poly5Reader
        - rename the channels, so that they match to the BIDS channel names

    If BIDS data exists:
        - load data with mne bids


    return:
        - mne_data as an MNE raw object
        - subject info (empty if not loaded with bids)
        - bids_ID

    """

    # rename channel names, if files were loaded via Poly5reader
    channel_mapping_1 = {
        "LFPR1STNM": "LFP_R_01_STN_MT",
        "LFPR2STNM": "LFP_R_02_STN_MT",
        "LFPR3STNM": "LFP_R_03_STN_MT",
        "LFPR4STNM": "LFP_R_04_STN_MT",
        "LFPR5STNM": "LFP_R_05_STN_MT",
        "LFPR6STNM": "LFP_R_06_STN_MT",
        "LFPR7STNM": "LFP_R_07_STN_MT",
        "LFPR8STNM": "LFP_R_08_STN_MT",
        "LFPL1STNM": "LFP_L_01_STN_MT",
        "LFPL2STNM": "LFP_L_02_STN_MT",
        "LFPL3STNM": "LFP_L_03_STN_MT",
        "LFPL4STNM": "LFP_L_04_STN_MT",
        "LFPL5STNM": "LFP_L_05_STN_MT",
        "LFPL6STNM": "LFP_L_06_STN_MT",
        "LFPL7STNM": "LFP_L_07_STN_MT",
        "LFPL8STNM": "LFP_L_08_STN_MT",
    }

    channel_mapping_2 = {
        "LFP_0_R_S": "LFP_R_01_STN_MT",
        "LFP_1_R_S": "LFP_R_02_STN_MT",
        "LFP_2_R_S": "LFP_R_03_STN_MT",
        "LFP_3_R_S": "LFP_R_04_STN_MT",
        "LFP_4_R_S": "LFP_R_05_STN_MT",
        "LFP_5_R_S": "LFP_R_06_STN_MT",
        "LFP_6_R_S": "LFP_R_07_STN_MT",
        "LFP_7_R_S": "LFP_R_08_STN_MT",
        "LFP_0_L_S": "LFP_L_01_STN_MT",
        "LFP_1_L_S": "LFP_L_02_STN_MT",
        "LFP_2_L_S": "LFP_L_03_STN_MT",
        "LFP_3_L_S": "LFP_L_04_STN_MT",
        "LFP_4_L_S": "LFP_L_05_STN_MT",
        "LFP_5_L_S": "LFP_L_06_STN_MT",
        "LFP_6_L_S": "LFP_L_07_STN_MT",
        "LFP_7_L_S": "LFP_L_08_STN_MT",
    }

    # check if patient is in the list with no BIDS yet
    if patient in SUBJECTS_NO_BIDS:
        mne_data = load_externalized_Poly5_files(sub=patient)

        # rename channels, first check which channel_mapping is correct
        found = False
        for name in mne_data.info["ch_names"]:
            if name in channel_mapping_1:
                found = True
                channel_mapping = channel_mapping_1
                break

            elif name in channel_mapping_2:
                found = True
                channel_mapping = channel_mapping_2
                break

        if found == False:
            print(
                f"Channel names of sub-{patient} are not in channel_mapping_1 or channel_mapping_2."
            )

        mne_data.rename_channels(channel_mapping)

        subject_info = "no_bids"

        # bids_ID
        bids_ID = f"sub-noBIDS{patient}"
        print(f"subject {patient} with bids ID {bids_ID} was loaded.")

    else:
        mne_data = load_BIDS_externalized_vhdr_files(sub=patient)

        subject_info = mne_data.info["subject_info"]
        bids_ID = mne_data.info["subject_info"]["his_id"]
        print(f"subject {patient} with bids ID {bids_ID} was loaded.")

    return {"mne_data": mne_data, "subject_info": subject_info, "bids_ID": bids_ID}


############# SAVE DATA #############

def save_sub_result_as_pickle(data, filename: str, results_path: str):
    """
    Input:
        - data: any
        - filename: str, e.g."externalized_preprocessed_data"
        - results_path: str, use 
            results_path = find_folders.get_monopolar_project_path(
                folder="results", sub=bids_id
                )

    picklefile will be written in the group_results_path:

    """

    data_path = os.path.join(results_path, f"{filename}.pickle")
    with open(data_path, "wb") as file:
        pickle.dump(data, file)

    print(f"{filename}.pickle", f"\nwritten in: {results_path}")

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


def assign_cluster(value=None):
    """
    This function takes an input float value and assigns a mathing cluster value between 1 and 3

        - value <= 0.4:         cluster 3
        - 0.4 < value <= 0.7:   cluster 2
        - 0.7 < value:          cluster 1

    """

    if value <= 0.4:
        return 3

    elif 0.4 < value <= 0.7:
        return 2

    elif 0.7 < value:
        return 1
