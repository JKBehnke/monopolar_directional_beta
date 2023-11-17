"""" 

Reference: Binder et al. 2023 

Selecting a contact pair with maximal beta power 
    - extracting beta power from the Percept Survey Segment recordings
    - select the bipolar recording with maximal beta power from all segmented recordings


"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import os
import mne
import pickle

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..utils import io_externalized as io_externalized


directional_recordings = [
    "1A1B",
    "1B1C",
    "1A1C",
    "2A2B",
    "2B2C",
    "2A2C",
    "1A2A",
    "1B2B",
    "1C2C",
]
incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]

results_path = find_folders.get_local_path(folder="GroupResults")


def select_directional_contact_pair_from_bssu(fooof_version: str):
    """
    Load the bipolar beta average from FOOOFed BSSU data


    """

    ranked_directional_bssu_recordings = pd.DataFrame()
    best_2_contacts = {}

    # Load the bipolar beta values (FOOOF)
    bssu_beta_data = loadResults.load_fooof_beta_ranks(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        all_or_one_chan="beta_all",
        all_or_one_longterm_ses="one_longterm_session",
    )

    # only take the directional recordings
    bssu_beta_data = bssu_beta_data.loc[
        bssu_beta_data["bipolar_channel"].isin(directional_recordings)
    ]

    # drop columns that are not necessary
    bssu_beta_data.drop(
        columns=[
            "fooof_error",
            "fooof_r_sq",
            "fooof_exponent",
            "fooof_offset",
            "fooof_power_spectrum",
            "periodic_plus_aperiodic_power_log",
            "fooof_periodic_flat",
            "fooof_number_peaks",
            "alpha_peak_CF_power_bandWidth",
            "low_beta_peak_CF_power_bandWidth",
            "high_beta_peak_CF_power_bandWidth",
            "beta_peak_CF_power_bandWidth",
            "gamma_peak_CF_power_bandWidth",
        ],
        inplace=True,
    )

    # for each session and STN, rank the bipolar beta average from all directional recordings
    for ses in incl_sessions:
        ses_data = bssu_beta_data.loc[bssu_beta_data.session == ses]

        # list of STNs with data from this session
        stn_unique_ses = list(ses_data.subject_hemisphere.unique())

        for sub_hem in stn_unique_ses:
            sub_hem_data = ses_data.loc[ses_data.subject_hemisphere == sub_hem]

            # rank the beta average from all directional recordings of one STN in one session
            sub_hem_data_copy = sub_hem_data.copy()
            sub_hem_data_copy["beta_rank"] = sub_hem_data_copy["beta_average"].rank(
                ascending=False
            )

            # save in a dataframe
            ranked_directional_bssu_recordings = pd.concat(
                [ranked_directional_bssu_recordings, sub_hem_data_copy]
            )

            # now only select the bipolar recording with rank 1
            rank_1_directional_recording = sub_hem_data_copy.loc[
                sub_hem_data_copy.beta_rank == 1.0
            ]
            rank_1_directional_recording = (
                rank_1_directional_recording.bipolar_channel.values[0]
            )

            # get both contacts of the rank 1 directional recording
            contact_1 = rank_1_directional_recording[0:2]  # e.g. 1A
            contact_2 = rank_1_directional_recording[2:4]  # e.g. 1C

            selected_2_contacts = [contact_1, contact_2]

            # save the result of the best 2 contacts
            best_2_contacts[f"{ses}_{sub_hem}"] = [ses, sub_hem, selected_2_contacts]

    # save as dataframe
    results_dataframe = pd.DataFrame(best_2_contacts)
    results_dataframe.rename(
        index={0: "session", 1: "subject_hemisphere", 2: "selected_2_contacts"},
        inplace=True,
    )
    results_dataframe = results_dataframe.transpose()

    # save monopolar psd estimate Dataframes as pickle files
    results_filepath = os.path.join(
        results_path, f"best_2_contacts_from_directional_bssu_{fooof_version}.pickle"
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dataframe, file)

    print(
        f"best_2_contacts_from_directional_bssu_{fooof_version}.pickle",
        f"\nwritten in: {results_path}",
    )

    return ranked_directional_bssu_recordings, results_dataframe


###################### externalized BSSU data ######################


def select_directional_contact_pair_from_externalized_bssu(fooof_version: str):
    """
    Load the bipolar beta average from FOOOFed BSSU data


    """

    ranked_directional_bssu_recordings = pd.DataFrame()
    best_2_contacts = {}

    # Load the bipolar beta values (FOOOF)
    bssu_beta_data = io_externalized.load_externalized_pickle(
        filename="fooof_externalized_group_BSSU_only_high_pass_filtered",
        fooof_version=fooof_version,
        reference="bipolar_to_lowermost",
    )

    # rename column "contact" to "bipolar_channel"
    bssu_beta_data.rename(columns={"contact": "bipolar_channel"}, inplace=True)

    # only take the directional recordings
    bssu_beta_data = bssu_beta_data.loc[
        bssu_beta_data["bipolar_channel"].isin(directional_recordings)
    ]

    # drop columns that are not necessary
    bssu_beta_data.drop(
        columns=[
            "fooof_error",
            "fooof_r_sq",
            "fooof_exponent",
            "fooof_offset",
            "periodic_plus_aperiodic_power_log",
            "fooof_periodic_flat",
            "fooof_number_peaks",
            "alpha_peak_CF_power_bandWidth",
            "low_beta_peak_CF_power_bandWidth",
            "high_beta_peak_CF_power_bandWidth",
            "beta_peak_CF_power_bandWidth",
            "gamma_peak_CF_power_bandWidth",
        ],
        inplace=True,
    )

    # list of STNs with data from this session
    stn_unique = list(bssu_beta_data.subject_hemisphere.unique())

    for sub_hem in stn_unique:
        sub_hem_data = bssu_beta_data.loc[bssu_beta_data.subject_hemisphere == sub_hem]

        # rank the beta average from all directional recordings of one STN in one session
        sub_hem_data_copy = sub_hem_data.copy()
        sub_hem_data_copy["beta_average"] = sub_hem_data_copy["fooof_power_spectrum"]
        sub_hem_data_copy["beta_average"] = sub_hem_data_copy["beta_average"].apply(
            lambda row: np.mean(row[13:36])
        )
        sub_hem_data_copy["beta_rank"] = sub_hem_data_copy["beta_average"].rank(
            ascending=False
        )

        # save in a dataframe
        ranked_directional_bssu_recordings = pd.concat(
            [ranked_directional_bssu_recordings, sub_hem_data_copy]
        )

        # now only select the bipolar recording with rank 1
        rank_1_directional_recording = sub_hem_data_copy.loc[
            sub_hem_data_copy.beta_rank == 1.0
        ]
        rank_1_directional_recording = (
            rank_1_directional_recording.bipolar_channel.values[0]
        )

        # get both contacts of the rank 1 directional recording
        contact_1 = rank_1_directional_recording[0:2]  # e.g. 1A
        contact_2 = rank_1_directional_recording[2:4]  # e.g. 1C

        selected_2_contacts = [contact_1, contact_2]

        # save the result of the best 2 contacts
        best_2_contacts[sub_hem] = ["postop", sub_hem, selected_2_contacts]

    # save as dataframe
    results_dataframe = pd.DataFrame(best_2_contacts)
    results_dataframe.rename(
        index={0: "session", 1: "subject_hemisphere", 2: "selected_2_contacts"},
        inplace=True,
    )
    results_dataframe = results_dataframe.transpose()

    # save monopolar psd estimate Dataframes as pickle files
    results_filepath = os.path.join(
        results_path,
        f"best_2_contacts_from_directional_externalized_bssu_{fooof_version}.pickle",
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dataframe, file)

    print(
        f"best_2_contacts_from_directional_externalized_bssu_{fooof_version}.pickle",
        f"\nwritten in: {results_path}",
    )

    return ranked_directional_bssu_recordings, results_dataframe
