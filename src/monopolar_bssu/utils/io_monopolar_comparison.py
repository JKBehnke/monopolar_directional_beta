""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..externalized_lfp import feats_ssd as feats_ssd
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import loadResults as loadResults

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")


################## beta ranks of directional contacts from externalized LFP ##################
################## FOOOF ##################


def load_externalized_fooof_data(
    fooof_version: str, new_reference: str, reference=None
):
    """
    Input:
        - fooof_version: str "v1" or "v2"
        - reference: str "bipolar_to_lowermost" or "no"
        - new_reference: "no", "one_to_zero_two_to_three"

    """
    if new_reference == "one_to_zero_two_to_three":
        fname_extension = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        fname_extension = ""

    # only directional contacts
    # FOOOF version: only 1 Hz high-pass filtered
    externalized_fooof_beta_ranks = io_externalized.load_externalized_pickle(
        filename=f"fooof_externalized_beta_ranks_directional_contacts_{fname_extension}only_high_pass_filtered",
        fooof_version=fooof_version,
        reference=reference,
    )

    # add column with method name
    externalized_fooof_beta_ranks_copy = externalized_fooof_beta_ranks.copy()
    externalized_fooof_beta_ranks_copy["method"] = "externalized_fooof"
    externalized_fooof_beta_ranks_copy["session"] = "postop"
    externalized_fooof_beta_ranks_copy[
        "estimated_monopolar_beta_psd"
    ] = externalized_fooof_beta_ranks_copy["beta_average"]

    # drop columns
    externalized_fooof_beta_ranks_copy.drop(
        columns=[
            "subject",
            "hemisphere",
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
            "beta_average",
        ],
        inplace=True,
    )

    # drop rows of subject 052 Right, because directional contact 2C was used as common reference, so there is no data for contact 2C
    externalized_fooof_beta_ranks_copy.reset_index(drop=True, inplace=True)
    # externalized_fooof_beta_ranks_copy.drop(
    #     externalized_fooof_beta_ranks_copy[
    #         externalized_fooof_beta_ranks_copy["subject_hemisphere"] == "052_Right"
    #     ].index,
    #     inplace=True,
    # )
    externalized_fooof_beta_ranks_copy.drop(
        externalized_fooof_beta_ranks_copy[
            externalized_fooof_beta_ranks_copy["subject_hemisphere"] == "048_Right"
        ].index,
        inplace=True,
    )

    return externalized_fooof_beta_ranks_copy


def load_externalized_ssd_data(reference=None):
    """
    Input:
        - reference: str "bipolar_to_lowermost" or "no"

    """
    ################## SSD ##################
    externalized_SSD_beta_ranks = io_externalized.load_externalized_pickle(
        filename="SSD_directional_externalized_channels", reference=reference
    )

    # add column with method name
    externalized_SSD_beta_ranks_copy = externalized_SSD_beta_ranks.copy()
    externalized_SSD_beta_ranks_copy["method"] = "externalized_ssd"
    externalized_SSD_beta_ranks_copy["session"] = "postop"
    externalized_SSD_beta_ranks_copy[
        "estimated_monopolar_beta_psd"
    ] = externalized_SSD_beta_ranks_copy["ssd_pattern"]

    # drop columns
    externalized_SSD_beta_ranks_copy.drop(
        columns=[
            "ssd_filtered_timedomain",
        ],
        inplace=True,
    )

    return externalized_SSD_beta_ranks_copy


def load_externalized_bssu_fooof_data(fooof_version: str, reference=None):
    """ """

    externalized_bssu_data = io_externalized.load_externalized_pickle(
        filename="fooof_externalized_group_BSSU_only_high_pass_filtered",
        fooof_version=fooof_version,
        reference=reference,
    )

    # add column with method name
    externalized_bssu_data_copy = externalized_bssu_data.copy()
    externalized_bssu_data_copy["method"] = "externalized_bssu_fooof"

    return externalized_bssu_data_copy


def load_euclidean_method(fooof_version: str):
    """ """
    ################## method weighted by euclidean coordinates ##################
    # only directional contacts
    monopolar_fooof_euclidean_segmental = loadResults.load_fooof_monopolar_weighted_psd(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        segmental="yes",
        similarity_calculation="inverse_distance",
    )

    monopolar_fooof_euclidean_segmental = pd.concat(
        [
            monopolar_fooof_euclidean_segmental["postop_monopolar_Dataframe"],
            monopolar_fooof_euclidean_segmental["fu3m_monopolar_Dataframe"],
            monopolar_fooof_euclidean_segmental["fu12m_monopolar_Dataframe"],
            monopolar_fooof_euclidean_segmental["fu18or24m_monopolar_Dataframe"],
        ]
    )

    # add column with method name
    monopolar_fooof_euclidean_segmental_copy = (
        monopolar_fooof_euclidean_segmental.copy()
    )
    monopolar_fooof_euclidean_segmental_copy["method"] = "euclidean_directional"
    monopolar_fooof_euclidean_segmental_copy[
        "beta_rank"
    ] = monopolar_fooof_euclidean_segmental_copy["rank"]
    monopolar_fooof_euclidean_segmental_copy.drop(columns=["rank"], inplace=True)

    # columns: coord_z, coord_xy, session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank

    return monopolar_fooof_euclidean_segmental_copy


def load_euclidean_externalized_bssu(fooof_version: str):
    """ """

    data = loadResults.load_pickle_group_result(
        filename="fooof_externalized_BSSU_monoRef_only_segmental_weight_beta_psd_by_inverse_distance",
        fooof_version=fooof_version,
    )

    data = data["monopolar_Dataframe"]

    # add column with method name
    data_copy = data.copy()
    data_copy["method"] = "euclidean_directional_externalized_bssu"
    data_copy["beta_rank"] = data_copy["rank"]
    data_copy.drop(columns=["rank"], inplace=True)

    return data_copy


def load_JLB_method(fooof_version: str):
    """ """
    ################## method by JLB ##################
    # only directional contacts
    monopolar_fooof_JLB = loadResults.load_pickle_group_result(
        filename="MonoRef_JLB_fooof_beta", fooof_version=fooof_version
    )
    # columns: session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank

    # add column with method name
    monopolar_fooof_JLB_copy = monopolar_fooof_JLB.copy()
    monopolar_fooof_JLB_copy["method"] = "JLB_directional"
    monopolar_fooof_JLB_copy["beta_rank"] = monopolar_fooof_JLB_copy["rank"]
    monopolar_fooof_JLB_copy.drop(columns=["rank"], inplace=True)

    return monopolar_fooof_JLB_copy


def load_JLB_externalized_bssu(fooof_version: str):
    """ """

    data = loadResults.load_pickle_group_result(
        filename="MonoRef_JLB_fooof_externalized_BSSU_beta", fooof_version=fooof_version
    )

    # add column with method name
    data_copy = data.copy()
    data_copy["method"] = "JLB_directional_externalized_bssu"
    data_copy["beta_rank"] = data_copy["rank"]
    data_copy.drop(columns=["rank"], inplace=True)

    return data_copy


def load_best_bssu_method(fooof_version: str):
    """ """
    ################## method by Binder et al. - best directional Survey contact pair ##################
    best_bssu_contacts = loadResults.load_pickle_group_result(
        filename="best_2_contacts_from_directional_bssu", fooof_version=fooof_version
    )

    # add column with method name
    best_bssu_contacts_copy = best_bssu_contacts.copy()
    best_bssu_contacts_copy["method"] = "best_bssu_contacts"

    return best_bssu_contacts_copy


def load_best_externalized_bssu(fooof_version: str):
    """ """
    ################## method by Binder et al. - best directional Survey contact pair ##################
    best_bssu_contacts = loadResults.load_pickle_group_result(
        filename="best_2_contacts_from_directional_externalized_bssu",
        fooof_version=fooof_version,
    )

    # add column with method name
    best_bssu_contacts_copy = best_bssu_contacts.copy()
    best_bssu_contacts_copy["method"] = "best_bssu_contacts_externalized_bssu"

    return best_bssu_contacts_copy


def load_detec_strelow_beta_ranks(
    fooof_version: str, level_first_or_all_directional: str
):
    """
    Method from Strelow et al. weighting power by distance between contact pairs

    Parameters:
        - fooof_version: The version of FOOOF to use. Currently, only "v2" is supported.
        - level_first_or_all_directional: A string indicating whether to load the beta ranks for the "level_first" or "all_directional" approach.

    Returns:
        A pandas DataFrame containing the beta ranks from the Strelow et al. method, weighted by distance between contact pairs.
        The DataFrame includes columns for the beta ranks and a column indicating the method name.

    Notes:
        - The function relies on the 'loadResults.load_pickle_group_result' function to load the beta ranks.
        - The function assumes the existence of the pickle files 'fooof_detec_beta_levels_and_directions_ranks' and 'fooof_detec_beta_all_directional_ranks'.


    """
    if level_first_or_all_directional == "all_directional":
        detec_fooof_result = loadResults.load_pickle_group_result(
            filename="fooof_detec_beta_all_directional_ranks",
            fooof_version=fooof_version,
        )

    elif level_first_or_all_directional == "level_first":
        detec_fooof_result = loadResults.load_pickle_group_result(
            filename="fooof_detec_beta_levels_and_directions_ranks",
            fooof_version=fooof_version,
        )

        # only keep the directional contacts of the level rank 1
        detec_fooof_result = detec_fooof_result.loc[
            detec_fooof_result.level_or_direction == "direction"
        ]

    # add column with method name
    detec_fooof_result_copy = detec_fooof_result.copy()
    detec_fooof_result_copy["method"] = "detec_strelow_contacts"
    detec_fooof_result_copy = detec_fooof_result_copy.reset_index()
    detec_fooof_result_copy = detec_fooof_result_copy.drop(columns=["index"])

    return detec_fooof_result_copy


def load_detec_strelow_beta_ranks_externalized_bssu(
    fooof_version: str, level_first_or_all_directional: str
):
    """
    Method from Strelow et al. weighting power by distance between contact pairs

    Parameters:
        - fooof_version: The version of FOOOF to use. Currently, only "v2" is supported.
        - level_first_or_all_directional: A string indicating whether to load the beta ranks for the "level_first" or "all_directional" approach.

    Returns:
        A pandas DataFrame containing the beta ranks from the Strelow et al. method, weighted by distance between contact pairs.
        The DataFrame includes columns for the beta ranks and a column indicating the method name.

    Notes:
        - The function relies on the 'loadResults.load_pickle_group_result' function to load the beta ranks.
        - The function assumes the existence of the pickle files 'fooof_detec_beta_levels_and_directions_ranks' and 'fooof_detec_beta_all_directional_ranks'.


    """
    if level_first_or_all_directional == "all_directional":
        detec_fooof_result = loadResults.load_pickle_group_result(
            filename="fooof_detec_externalized_bssu_beta_all_directional_ranks",
            fooof_version=fooof_version,
        )

    elif level_first_or_all_directional == "level_first":
        detec_fooof_result = loadResults.load_pickle_group_result(
            filename="fooof_detec_externalized_bssu_beta_levels_and_directions_ranks",
            fooof_version=fooof_version,
        )

        # only keep the directional contacts of the level rank 1
        detec_fooof_result = detec_fooof_result.loc[
            detec_fooof_result.level_or_direction == "direction"
        ]

    # add column with method name
    detec_fooof_result_copy = detec_fooof_result.copy()
    detec_fooof_result_copy["method"] = "detec_strelow_contacts_externalized_bssu"
    detec_fooof_result_copy = detec_fooof_result_copy.reset_index()
    detec_fooof_result_copy = detec_fooof_result_copy.drop(columns=["index"])

    return detec_fooof_result_copy


def load_best_clinical_contacts():
    """
    Loading the Excel file BestClinicalStimulation.xlsx , sheet "BestContacts_one_longterm"
    """
    best_clinical_stimulation = loadResults.load_BestClinicalStimulation_excel()
    best_clinical_contacts = best_clinical_stimulation["BestContacts_one_longterm"]

    # add column with method name
    best_clinical_contacts_copy = best_clinical_contacts.copy()
    best_clinical_contacts_copy["method"] = "best_clinical_contacts"

    return best_clinical_contacts_copy


def save_result_excel(result_df: pd.DataFrame, filename: str, sheet_name: str):
    """
    Saves dataframe as Excel file

    Input:
        - result_df
        - filename
        - sheet_name

    """

    xlsx_filename = f"{filename}.xlsx"

    result_df.to_excel(
        os.path.join(GROUP_RESULTS_PATH, xlsx_filename),
        sheet_name=sheet_name,
        index=False,
    )

    print(
        "file: ",
        f"{xlsx_filename}",
        "\nwritten in: ",
        GROUP_RESULTS_PATH,
    )


def save_result_as_pickle(filename: str, data=None):
    """
    Input:
        - data: must be a pd.DataFrame() or dict
        - filename: str, e.g."externalized_preprocessed_data"

    picklefile will be written in the group_results_path:

    """

    group_data_path = os.path.join(GROUP_RESULTS_PATH, f"{filename}.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(data, file)

    print(f"{filename}.pickle", f"\nwritten in: {GROUP_RESULTS_PATH}")


def save_fig_png_and_svg(filename: str, figure=None):
    """
    Input:
        - path: str
        - filename: str
        - figure: must be a plt figure

    """

    figure.savefig(
        os.path.join(GROUP_FIGURES_PATH, f"{filename}.svg"),
        bbox_inches="tight",
        format="svg",
    )

    figure.savefig(
        os.path.join(GROUP_FIGURES_PATH, f"{filename}.png"),
        bbox_inches="tight",
    )

    print(
        f"Figures {filename}.svg and {filename}.png",
        f"\nwere written in: {GROUP_FIGURES_PATH}.",
    )
