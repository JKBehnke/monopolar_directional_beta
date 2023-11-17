""" Monopolar referencing by Strelow et al. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# utility functions
from ..utils import loadResults as loadResults
from ..utils import find_folders as find_folders
from ..utils import io_percept as io_percept
from ..utils import io_externalized as io_externalized
from ..utils import percept_lfp_preprocessing as percept_lfp_preprocessing
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing

GROUP_RESULTS_PATH = find_folders.get_local_path(folder="GroupResults")
INCL_SESSIONS = ["postop", "fu3m", "fu12m", "fu18or24m"]
DIRECTIONAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]
LEVELS = ["1", "2"]


DISTANCE_DICT = {
    "1": {
        "01": 2,
        "12": 2,
        "13": 4,
    },
    "2": {
        "02": 4,
        "12": 2,
        "23": 2,
    },
    "1A": {
        "1A1B": 1.126,  # calculation of distance between segments: 0.65 mm * sq_root(3)
        "1A1C": 1.126,
    },
    "1B": {
        "1A1B": 1.126,
        "1B1C": 1.126,
    },
    "1C": {
        "1A1C": 1.126,
        "1B1C": 1.126,
    },
    "2A": {
        "2A2B": 1.126,
        "2A2C": 1.126,
    },
    "2B": {
        "2A2B": 1.126,
        "2B2C": 1.126,
    },
    "2C": {
        "2B2C": 1.126,
        "2A2C": 1.126,
    },
}


def weight_power_by_distance(
    stn_data: pd.DataFrame, distance_dict: dict, spectra_column: str
):
    """

    Calculate the weighted average power of LFPs based on distance.

    Parameters:
        - stn_data: A pandas DataFrame containing LFP data.
        - distance_dict: A dictionary mapping channel names to their corresponding distances.

    Returns:
        The weighted average power of LFPs, calculated by summing the weighted power spectra of each channel
        and dividing it by the sum of all distances used.

    Notes:
        - The function assumes that the 'stn_data' DataFrame has a column named 'bipolar_channel' and "fooof_power_spectrum"
        - The 'distance_dict' should have channel names as keys and their corresponding distances as values.

    """

    # weight each LFP and add to a list
    weighted_power_list = []
    for channel in distance_dict:  # loops through keys e.g. ("1A1B", "1A1C")
        lfp_data = stn_data.loc[
            stn_data.bipolar_channel == channel
        ]  # select the channel of interest e.g. "1A1B"
        lfp_data = lfp_data[spectra_column].values[
            0
        ]  # get the FOOOF power spectrum from this channel
        # TODO: instead of only beta, weight the whole power spectrum!
        lfp_data = (
            lfp_data / distance_dict[channel]
        )  # weighted power, distance between 0 and 1 = 2 mm

        weighted_power_list.append(lfp_data)

    # sum of the 3 weighted power spectra
    sum_lfp_power = np.sum(weighted_power_list, axis=0)
    sum_distances = sum(distance_dict.values())  # sum of all distances used

    return (
        sum_lfp_power / sum_distances
    )  # weighted power of one level or one directional contact


def fooof_detec_weight_power(fooof_version: str):
    """

    Weight FOOOF power spectra using the DETEC method.

    Parameters:
        - fooof_version: The version of FOOOF to use. Currently, only "v2" is supported.

    Returns:
        A pandas DataFrame containing the weighted FOOOF power spectra for each subject, session, and contact.
        The DataFrame includes columns for 'subject_hemisphere', 'session', 'contact', and 'weighted_fooof_power_spectrum'.

    Notes:
        - The function relies on the 'loadResults.load_fooof_beta_ranks' function to load the FOOOF dataframe.
        - The function uses the 'weight_power_by_distance' function to weight the power spectra based on distance.
        - The 'DISTANCE_DICT' dictionary is used to provide the distance information for weighting.
        - The function assumes the existence of the 'INCL_SESSIONS' list.
        - The function calculates the beta average from the weighted FOOOF power spectra.


    """

    # result
    weighted_group_results = pd.DataFrame()

    ############# Load the FOOOF dataframe #############
    beta_average_DF = loadResults.load_fooof_beta_ranks(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        all_or_one_chan="beta_all",
        all_or_one_longterm_ses="one_longterm_session",
    )

    # loop over sessions
    for ses in INCL_SESSIONS:
        # check if session exists
        if ses not in beta_average_DF.session.values:
            continue

        session_Dataframe = beta_average_DF.loc[beta_average_DF.session == ses]
        # copying session_Dataframe to add new columns
        session_Dataframe_copy = session_Dataframe.copy()
        session_Dataframe_copy = session_Dataframe_copy.reset_index()
        session_Dataframe_copy = session_Dataframe_copy.drop(columns=["index"])

        stn_unique = list(session_Dataframe_copy.subject_hemisphere.unique())

        ##################### for every STN: get the relevant FOOOF power spectra and weight depending on distance between contact pair of bipolar LFP #####################
        for stn in stn_unique:
            stn_data = session_Dataframe_copy.loc[
                session_Dataframe_copy.subject_hemisphere == stn
            ]

            # weight the power for each level and directional contact
            weighted_power_dict = {
                level_or_direc: weight_power_by_distance(
                    stn_data=stn_data,
                    distance_dict=DISTANCE_DICT[level_or_direc],
                    spectra_column="fooof_power_spectrum",
                )
                for level_or_direc in DISTANCE_DICT
            }

            for cont in weighted_power_dict:
                stn_weighted_dict = {
                    "subject_hemisphere": [stn],
                    "session": [ses],
                    "contact": [cont],
                    "weighted_fooof_power_spectrum": [weighted_power_dict[cont]],
                }

                stn_weighted_results = pd.DataFrame(stn_weighted_dict)
                weighted_group_results = pd.concat(
                    [weighted_group_results, stn_weighted_results], ignore_index=True
                )

    # calculate beta average from weighted fooof power spectra
    weighted_group_results_copy = weighted_group_results.copy()
    weighted_group_results_copy[
        "estimated_monopolar_beta_psd"
    ] = weighted_group_results_copy["weighted_fooof_power_spectrum"]
    weighted_group_results_copy[
        "estimated_monopolar_beta_psd"
    ] = weighted_group_results_copy["estimated_monopolar_beta_psd"].apply(
        lambda row: np.mean(row[13:36])
    )

    return weighted_group_results_copy


def detec_rank_level_and_direction(fooof_version: str):
    """ """

    # result
    beta_level_dir_ranks = pd.DataFrame()
    beta_all_directional_ranks = pd.DataFrame()

    weighted_group_results = fooof_detec_weight_power(fooof_version=fooof_version)

    # loop over sessions
    for ses in INCL_SESSIONS:
        # check if session exists
        if ses not in weighted_group_results.session.values:
            continue

        session_Dataframe = weighted_group_results.loc[
            weighted_group_results.session == ses
        ]

        stn_unique = list(session_Dataframe.subject_hemisphere.unique())

        ##################### for every STN: first rank beta average of levels, then rank beta average of directions of level rank 1 #####################
        for stn in stn_unique:
            stn_data = session_Dataframe.loc[
                session_Dataframe.subject_hemisphere == stn
            ]

            ################# weighted power #################
            # rank beta average of 6 directional contacts
            all_directional_data = stn_data[
                stn_data["contact"].isin(DIRECTIONAL_CONTACTS)
            ]
            all_directional_data_copy = all_directional_data.copy()
            all_directional_data_copy["beta_rank"] = all_directional_data_copy[
                "estimated_monopolar_beta_psd"
            ].rank(ascending=False)

            # normalize to maximal beta
            max_value_dir = all_directional_data_copy[
                "estimated_monopolar_beta_psd"
            ].max()
            all_directional_data_copy["beta_relative_to_max"] = (
                all_directional_data_copy["estimated_monopolar_beta_psd"]
                / max_value_dir
            )

            # cluster values into 3 categories: <40%, 40-70% and >70%
            all_directional_data_copy["beta_cluster"] = all_directional_data_copy[
                "beta_relative_to_max"
            ].apply(percept_lfp_preprocessing.assign_cluster)

            # save
            beta_all_directional_ranks = pd.concat(
                [beta_all_directional_ranks, all_directional_data_copy],
                ignore_index=True,
            )

            ################# strategy 1st rank level, 2nd rank directions #################
            # 1st step: rank levels
            level_data = stn_data[stn_data["contact"].isin(LEVELS)]
            level_data_copy = level_data.copy()
            level_data_copy["beta_rank"] = level_data_copy[
                "estimated_monopolar_beta_psd"
            ].rank(ascending=False)
            level_data_copy["level_or_direction"] = "level"

            level_rank_1 = level_data_copy.loc[level_data_copy["beta_rank"] == 1.0]
            level_rank_1 = level_rank_1.contact.values[0]  # level 1 or 2

            # 2nd step: rank directions of the level with rank 1
            direction_contacts_level_rank_1 = [
                f"{level_rank_1}A",
                f"{level_rank_1}B",
                f"{level_rank_1}C",
            ]
            direction_data = stn_data[
                stn_data["contact"].isin(direction_contacts_level_rank_1)
            ]
            direction_data_copy = direction_data.copy()
            direction_data_copy["beta_rank"] = direction_data_copy[
                "estimated_monopolar_beta_psd"
            ].rank(ascending=False)
            direction_data_copy["level_or_direction"] = "direction"

            # save to group dataframe
            beta_level_dir_ranks = pd.concat(
                [beta_level_dir_ranks, level_data_copy, direction_data_copy],
                ignore_index=True,
            )

    # save dataframe
    io_percept.save_result_dataframe_as_pickle(
        data=beta_all_directional_ranks,
        filename=f"fooof_detec_beta_all_directional_ranks_{fooof_version}",
    )

    io_percept.save_result_dataframe_as_pickle(
        data=beta_level_dir_ranks,
        filename=f"fooof_detec_beta_levels_and_directions_ranks_{fooof_version}",
    )

    return beta_all_directional_ranks, beta_level_dir_ranks


################ externalized BSSU ################


def fooof_detec_weight_power_externalised_bssu(fooof_version: str, data_type: str):
    """

    Weight FOOOF power spectra using the DETEC method.

    Parameters:
        - fooof_version: The version of FOOOF to use. Currently, only "v2" is supported.
        - data_type: "fooof", "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"

    Returns:
        A pandas DataFrame containing the weighted FOOOF power spectra for each subject, session, and contact.
        The DataFrame includes columns for 'subject_hemisphere', 'session', 'contact', and 'weighted_fooof_power_spectrum'.

    Notes:
        - The function relies on the 'loadResults.load_fooof_beta_ranks' function to load the FOOOF dataframe.
        - The function uses the 'weight_power_by_distance' function to weight the power spectra based on distance.
        - The 'DISTANCE_DICT' dictionary is used to provide the distance information for weighting.
        - The function assumes the existence of the 'INCL_SESSIONS' list.
        - The function calculates the beta average from the weighted FOOOF power spectra.


    """

    # result
    weighted_group_results = pd.DataFrame()
    weighted_power_spectra = {}

    ############# Load the FOOOF dataframe #############
    # beta_average_DF = load_data.load_externalized_pickle(
    #     filename="fooof_externalized_group_BSSU_only_high_pass_filtered",
    #     fooof_version=fooof_version,
    #     reference="bipolar_to_lowermost",
    # )
    loaded_data = io_externalized.load_data_to_weight(data_type=data_type)
    externalized_data = loaded_data["loaded_data"]
    spectra_column = loaded_data["spectra"]

    # get frequencies for power plots
    if data_type != "fooof":
        frequencies = externalized_data["frequencies"].values[0]

    elif data_type == "fooof":
        frequencies = np.arange(2, 46)  # fooof model v2: 2-45 Hz

    # rename column "contact" to "bipolar_channel"
    # beta_average_DF = beta_average_DF.rename(columns={"contact": "bipolar_channel"})

    # copying session_Dataframe to add new columns
    Dataframe_copy = externalized_data.copy()

    stn_unique = list(Dataframe_copy.subject_hemisphere.unique())

    ##################### for every STN: get the relevant FOOOF power spectra and weight depending on distance between contact pair of bipolar LFP #####################
    for stn in stn_unique:
        stn_data = Dataframe_copy.loc[Dataframe_copy.subject_hemisphere == stn]

        # weight the power for each level and directional contact
        weighted_power_spectra_single_stn = {}

        weighted_power_dict = {
            level_or_direc: weight_power_by_distance(
                stn_data=stn_data,
                distance_dict=DISTANCE_DICT[level_or_direc],
                spectra_column=spectra_column,
            )
            for level_or_direc in DISTANCE_DICT
        }

        for cont in weighted_power_dict:
            stn_weighted_dict = {
                "subject_hemisphere": [stn],
                "session": ["postop"],
                "contact": [cont],
                "weighted_fooof_power_spectrum": [weighted_power_dict[cont]],
            }

            stn_weighted_results = pd.DataFrame(stn_weighted_dict)
            weighted_group_results = pd.concat(
                [weighted_group_results, stn_weighted_results], ignore_index=True
            )
            weighted_power_spectra_single_stn[cont] = weighted_power_dict[cont]

        weighted_power_spectra[stn] = {
            "weighted_power": weighted_power_spectra_single_stn,
            "frequencies": frequencies,
        }

    # save weighted power spectra
    io_externalized.save_result_dataframe_as_pickle(
        data=weighted_power_spectra,
        filename=f"detec_strelow_{data_type}_externalized_BSSU_weighted_power_spectra_{fooof_version}",
    )

    # calculate beta average from weighted fooof power spectra
    weighted_group_results_copy = weighted_group_results.copy()
    weighted_group_results_copy[
        "estimated_monopolar_beta_psd"
    ] = weighted_group_results_copy["weighted_fooof_power_spectrum"]
    weighted_group_results_copy[
        "estimated_monopolar_beta_psd"
    ] = weighted_group_results_copy["estimated_monopolar_beta_psd"].apply(
        lambda row: np.mean(row[13:36])
    )

    return weighted_group_results_copy


def detec_rank_level_and_direction_externalized_bssu(
    fooof_version: str, data_type: str
):
    """
    Input:
        - data_type: "fooof", "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"


    """

    # result
    beta_level_dir_ranks = pd.DataFrame()
    beta_all_directional_ranks = pd.DataFrame()

    weighted_group_results = fooof_detec_weight_power_externalised_bssu(
        fooof_version=fooof_version, data_type=data_type
    )

    stn_unique = list(weighted_group_results.subject_hemisphere.unique())

    ##################### for every STN: first rank beta average of levels, then rank beta average of directions of level rank 1 #####################
    for stn in stn_unique:
        stn_data = weighted_group_results.loc[
            weighted_group_results.subject_hemisphere == stn
        ]

        ################# weighted power #################
        # rank beta average of 6 directional contacts
        all_directional_data = stn_data[stn_data["contact"].isin(DIRECTIONAL_CONTACTS)]
        all_directional_data_copy = all_directional_data.copy()
        all_directional_data_copy["beta_rank"] = all_directional_data_copy[
            "estimated_monopolar_beta_psd"
        ].rank(ascending=False)

        # normalize to maximal beta
        max_value_dir = all_directional_data_copy["estimated_monopolar_beta_psd"].max()
        all_directional_data_copy["beta_relative_to_max"] = (
            all_directional_data_copy["estimated_monopolar_beta_psd"] / max_value_dir
        )

        # cluster values into 3 categories: <40%, 40-70% and >70%
        all_directional_data_copy["beta_cluster"] = all_directional_data_copy[
            "beta_relative_to_max"
        ].apply(externalized_lfp_preprocessing.assign_cluster)

        # save
        beta_all_directional_ranks = pd.concat(
            [beta_all_directional_ranks, all_directional_data_copy],
            ignore_index=True,
        )

        ################# strategy 1st rank level, 2nd rank directions #################
        # 1st step: rank levels
        level_data = stn_data[stn_data["contact"].isin(LEVELS)]
        level_data_copy = level_data.copy()
        level_data_copy["beta_rank"] = level_data_copy[
            "estimated_monopolar_beta_psd"
        ].rank(ascending=False)
        level_data_copy["level_or_direction"] = "level"

        level_rank_1 = level_data_copy.loc[level_data_copy["beta_rank"] == 1.0]
        level_rank_1 = level_rank_1.contact.values[0]  # level 1 or 2

        # 2nd step: rank directions of the level with rank 1
        direction_contacts_level_rank_1 = [
            f"{level_rank_1}A",
            f"{level_rank_1}B",
            f"{level_rank_1}C",
        ]
        direction_data = stn_data[
            stn_data["contact"].isin(direction_contacts_level_rank_1)
        ]
        direction_data_copy = direction_data.copy()
        direction_data_copy["beta_rank"] = direction_data_copy[
            "estimated_monopolar_beta_psd"
        ].rank(ascending=False)
        direction_data_copy["level_or_direction"] = "direction"

        # save to group dataframe
        beta_level_dir_ranks = pd.concat(
            [beta_level_dir_ranks, level_data_copy, direction_data_copy],
            ignore_index=True,
        )

    # save dataframe
    io_externalized.save_result_dataframe_as_pickle(
        data=beta_all_directional_ranks,
        filename=f"{data_type}_detec_externalized_bssu_beta_all_directional_ranks_{fooof_version}",
    )

    io_externalized.save_result_dataframe_as_pickle(
        data=beta_level_dir_ranks,
        filename=f"{data_type}_detec_externalized_bssu_beta_levels_and_directions_ranks_{fooof_version}",
    )

    return beta_all_directional_ranks, beta_level_dir_ranks
