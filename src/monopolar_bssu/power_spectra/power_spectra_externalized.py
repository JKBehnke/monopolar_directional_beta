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
