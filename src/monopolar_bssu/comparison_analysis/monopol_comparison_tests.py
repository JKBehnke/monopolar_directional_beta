""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import loadResults as loadResults

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")



def correlation_tests_percept_methods(
    method_1: str,
    method_2: str,
    method_1_df: pd.DataFrame,
    method_2_df: pd.DataFrame,
    ses: str,
):
    """
    Requirement for methods: must have beta values for all directional contacts (n=6), so don't use best_bssu method here!

    For each session:
    for each subject hemisphere:

    perform 3 correlation tests between both methods:
        - estimated_beta_spearman
        - normalized_beta_pearson
        - cluster_beta_spearman

    return a dataframe with results

    """

    results_DF = pd.DataFrame()

    # find STNs with data from both methods
    stn_unique_method_1 = list(method_1_df.subject_hemisphere.unique())
    stn_unique_method_2 = list(method_2_df.subject_hemisphere.unique())

    stn_comparison_list = sorted(set(stn_unique_method_1) & set(stn_unique_method_2))
    comparison_df_method_1 = method_1_df.loc[
        method_1_df["subject_hemisphere"].isin(stn_comparison_list)
    ]
    comparison_df_method_2 = method_2_df.loc[
        method_2_df["subject_hemisphere"].isin(stn_comparison_list)
    ]

    comparison_df = pd.concat([comparison_df_method_1, comparison_df_method_2], axis=0)

    for sub_hem in stn_comparison_list:
        # only run, if sub_hem STN exists in both session Dataframes
        if sub_hem not in comparison_df.subject_hemisphere.values:
            print(f"{sub_hem} is not in the comparison Dataframe.")
            continue

        # only take one electrode at both sessions and get spearman correlation
        stn_comparison = comparison_df.loc[
            comparison_df["subject_hemisphere"] == sub_hem
        ]

        stn_method_1 = stn_comparison.loc[stn_comparison.method == method_1]
        stn_method_2 = stn_comparison.loc[stn_comparison.method == method_2]

        # Spearman correlation between estimated beta average
        # set a default value for the correlation tests

        spearman_statistic = np.nan
        spearman_pval = np.nan
        pearson_normalized_statistic = np.nan
        pearson_normalized_pval = np.nan
        spearman_cluster_statistic = np.nan
        spearman_cluster_pval = np.nan

        ####### CHECK IF THERE ARE NANS IN THE DATAFRAME ########
        if (
            np.isnan(stn_method_1["estimated_monopolar_beta_psd"].values).any()
            or np.isnan(stn_method_2["estimated_monopolar_beta_psd"].values).any()
        ):
            print(f"Sub-{sub_hem} has NaN values in the estimated beta average. NaN was exchanged by zero")
            
        elif (
            np.isnan(stn_method_1["beta_relative_to_max"].values).any()
            or np.isnan(stn_method_2["beta_relative_to_max"].values).any()
        ):
            print(f"Sub-{sub_hem} has NaN values in beta_relative_to_max. NaN was exchanged by zero")
            
        elif (
            np.isnan(stn_method_1["beta_cluster"].values).any()
            or np.isnan(stn_method_2["beta_cluster"].values).any()
        ):
            print(f"Sub-{sub_hem} has NaN values in beta_relative_to_max. NaN was exchanged by 3.")
            

        # # replace NaNs by values
        # stn_method_1.loc[stn_method_1['estimated_monopolar_beta_psd'].isna(), 'estimated_monopolar_beta_psd'] = 0
        # stn_method_2.loc[stn_method_2["estimated_monopolar_beta_psd"].isna(), "estimated_monopolar_beta_psd"] = 0

        # stn_method_1.loc[stn_method_1["beta_relative_to_max"].isna(), "beta_relative_to_max"] = 0
        # stn_method_2.loc[stn_method_2["beta_relative_to_max"].isna(), "beta_relative_to_max"] = 0

        # stn_method_1.loc[stn_method_1["beta_cluster"].isna(), "beta_cluster"] = 3
        # stn_method_2.loc[stn_method_2["beta_cluster"].isna(), "beta_cluster"] = 3

        else:  # correlation tests only work if there is no NaN value
            spearman_beta_stn = stats.spearmanr(
                stn_method_1["estimated_monopolar_beta_psd"].values,
                stn_method_2["estimated_monopolar_beta_psd"].values,
            )
            spearman_statistic = spearman_beta_stn.statistic
            spearman_pval = spearman_beta_stn.pvalue

            # Pearson correlation between normalized beta to maximum within each electrode
            pearson_normalized_beta_stn = stats.pearsonr(
                stn_method_1["beta_relative_to_max"].values,
                stn_method_2["beta_relative_to_max"].values,
            )
            pearson_normalized_statistic = pearson_normalized_beta_stn.statistic
            pearson_normalized_pval = pearson_normalized_beta_stn.pvalue

            spearman_beta_cluster_stn = stats.spearmanr(
                stn_method_1["beta_cluster"].values, stn_method_2["beta_cluster"].values
            )
            spearman_cluster_statistic = spearman_beta_cluster_stn.statistic
            spearman_cluster_pval = spearman_beta_cluster_stn.pvalue

        # contacts with beta rank 1 and 2
        no_rank_1 = "no"
        no_rank_2 = "no"
        ############## method 1: ##############
        rank1_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 1.0]
        # check if externalized has a rank 1 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank1_method_1.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 1 contact in the recording {method_1}.")
            no_rank_1 = "yes"
            rank1_method_1 = "none"
            #continue

        else:
            rank1_method_1 = rank1_method_1.contact.values[0]

        rank2_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 2.0]
        # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank2_method_1.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 2 contact in the recording {method_1}.")
            no_rank_2 = "yes"
            rank2_method_1 = "none"
            #continue

        else:
            rank2_method_1 = rank2_method_1.contact.values[0]

        rank_1_and_2_method_1 = [rank1_method_1, rank2_method_1]

        ############### method 2: ##############
        rank1_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 1.0]
        
        # check if externalized has a rank 1 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank1_method_2.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 1 contact in the recording {method_2}.")
            #continue
            no_rank_1 = "yes"
            rank1_method_2 = "none"

        else:
            rank1_method_2 = rank1_method_2.contact.values[0]

        rank2_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 2.0]
        # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank2_method_2.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 2 contact in the recording {method_2}.")
            #continue
            no_rank_2 = "yes"
            rank2_method_2 = "none"
        
        else:
            rank2_method_2 = rank2_method_2.contact.values[0]

        rank_1_and_2_method_2 = [rank1_method_2, rank2_method_2]

        # yes if contact with rank 1 is the same
        if no_rank_1 == "yes":
            compare_rank_1_contact = "no_rank_1"

        elif rank1_method_1 == rank1_method_2:
            compare_rank_1_contact = "same"

        else:
            compare_rank_1_contact = "different"

        # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
        if no_rank_1 == "yes" or no_rank_2 == "yes":
            both_contacts_matching = "no_rank_1_or_2"
        
        elif set(rank_1_and_2_method_1) == set(rank_1_and_2_method_2):
            both_contacts_matching = "yes"

        else:
            both_contacts_matching = "no"

        # check if at least one contact selected as beta rank 1 or 2 match for both methods
        if no_rank_1 == "yes" or no_rank_2 == "yes":
            compare_rank_1_and_2_contacts = "no_rank_1_or_2"

        if set(rank_1_and_2_method_1).intersection(set(rank_1_and_2_method_2)):
            compare_rank_1_and_2_contacts = "at_least_one_contact_match"

        else:
            compare_rank_1_and_2_contacts = "no_contacts_match"

        # store values in a dictionary
        spearman_result = {
            "method_1": [method_1],
            "method_2": [method_2],
            "session": [ses],
            "subject_hemisphere": [sub_hem],
            "estimated_beta_spearman_r": [spearman_statistic],
            "estimated_beta_spearman_pval": [spearman_pval],
            "normalized_beta_pearson_r": [pearson_normalized_statistic],
            "normalized_beta_pearson_pval": [pearson_normalized_pval],
            "cluster_beta_spearman_r": [spearman_cluster_statistic],
            "cluster_beta_spearman_pval": [spearman_cluster_pval],
            "contact_rank_1_method_1": [rank1_method_1],
            "contact_rank_1_method_2": [rank1_method_2],
            "contacts_rank_1_2_method_1": [rank_1_and_2_method_1],
            "contacts_rank_1_2_method_2": [rank_1_and_2_method_2],
            "compare_rank_1_contact": [compare_rank_1_contact],
            "compare_rank_1_and_2_contacts": [compare_rank_1_and_2_contacts],
            "both_contacts_matching": [both_contacts_matching],
        }
        results_single_DF = pd.DataFrame(spearman_result)

        results_DF = pd.concat([results_DF, results_single_DF], ignore_index=True)

    return results_DF


def rank_comparison_percept_methods(
    method_1: str,
    method_2: str,
    method_1_df: pd.DataFrame,
    method_2_df: pd.DataFrame,
    ses: str,
    bssu_version: str,
):
    """

    For each session:
    for each subject hemisphere:

    compare if rank 1 and 2 contacts both match or at least one contact match

    return a dataframe with results

    """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    comparison_result = pd.DataFrame()

    # find STNs with data from both methods and externalized
    stn_unique_method_1 = list(method_1_df.subject_hemisphere.unique())
    stn_unique_method_2 = list(method_2_df.subject_hemisphere.unique())

    stn_comparison_list = sorted(set(stn_unique_method_1) & set(stn_unique_method_2))
    comparison_df_method_1 = method_1_df.loc[
        method_1_df["subject_hemisphere"].isin(stn_comparison_list)
    ]
    comparison_df_method_2 = method_2_df.loc[
        method_2_df["subject_hemisphere"].isin(stn_comparison_list)
    ]

    comparison_df = pd.concat([comparison_df_method_1, comparison_df_method_2], axis=0)

    for sub_hem in stn_comparison_list:
        # only run, if sub_hem STN exists in both session Dataframes
        if sub_hem not in comparison_df.subject_hemisphere.values:
            print(f"{sub_hem} is not in the comparison Dataframe.")
            continue

        # only take one electrode at both sessions and get spearman correlation
        stn_comparison = comparison_df.loc[
            comparison_df["subject_hemisphere"] == sub_hem
        ]

        stn_method_1 = stn_comparison.loc[stn_comparison.method == method_1]
        stn_method_2 = stn_comparison.loc[stn_comparison.method == method_2]

        ######### METHOD 1 RANK CONTACTS 1 AND 2 #########
        if method_1 == f"best_bssu_contacts{external_extension}":
            rank1_method_1 = "none"
            rank2_method_1 = "none"
            rank_1_and_2_method_1 = stn_method_1.selected_2_contacts.values[0]

        else:
            # contacts with beta rank 1 and 2
            rank1_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 1.0]
            rank2_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 2.0]
            if (
                len(rank1_method_1.contact.values) == 0
                or len(rank2_method_1.contact.values) == 0
            ):
                print(f"no rank 1 contact in {sub_hem} {method_1}")
                continue

            else:
                rank1_method_1 = rank1_method_1.contact.values[0]
                rank2_method_1 = rank2_method_1.contact.values[0]

                rank_1_and_2_method_1 = [rank1_method_1, rank2_method_1]

        if method_2 == f"best_bssu_contacts{external_extension}":
            rank1_method_2 = "none"
            rank2_method_2 = "none"
            rank_1_and_2_method_2 = stn_method_2.selected_2_contacts.values[0]

        else:
            # contacts with beta rank 1 and 2
            rank1_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 1.0]
            rank2_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 2.0]

            if (
                len(rank1_method_2.contact.values) == 0
                or len(rank2_method_2.contact.values) == 0
            ):
                print(f"no rank 1 contact in {sub_hem} {method_2}")
                continue

            else:
                rank1_method_2 = rank1_method_2.contact.values[0]
                rank2_method_2 = rank2_method_2.contact.values[0]

                rank_1_and_2_method_2 = [rank1_method_2, rank2_method_2]

        # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
        if set(rank_1_and_2_method_1) == set(rank_1_and_2_method_2):
            both_contacts_matching = "yes"

        else:
            both_contacts_matching = "no"

        # check if at least one contact selected as beta rank 1 or 2 match for both methods
        if set(rank_1_and_2_method_1).intersection(set(rank_1_and_2_method_2)):
            at_least_1_contact_matching = "at_least_one_contact_match"

        else:
            at_least_1_contact_matching = "no_contacts_match"

        # check if rank 1 contacts are the same
        # if rank1_method_1 == rank1_method_2:
        #     rank_1_contacts_matching = "yes"
        
        # else: 
        #     rank_1_contacts_matching = "no"

        # store values in a dictionary
        comparison_result_dict = {
            "method_1": [method_1],
            "method_2": [method_2],
            "session": [ses],
            "subject_hemisphere": [sub_hem],
            "contact_rank_1_method_1": [rank1_method_1],
            "contact_rank_2_method_1": [rank2_method_1],
            "rank_1_and_2_method_1": [rank_1_and_2_method_1],
            "contact_rank_1_method_2": [rank1_method_2],
            "contact_rank_2_method_2": [rank2_method_2],
            "rank_1_and_2_method_2": [rank_1_and_2_method_2],
            # "bssu_best_contact_pair": [best_contact_pair],
            "both_contacts_matching": [both_contacts_matching],
            "at_least_1_contact_matching": [at_least_1_contact_matching],
            #"rank_1_contacts_matching": [rank_1_contacts_matching]
        }
        comparison_single_result = pd.DataFrame(comparison_result_dict)
        comparison_result = pd.concat(
            [comparison_result, comparison_single_result], ignore_index=True
        )

    return comparison_result


def get_sample_size_percept_methods(
    ses: str,
    ses_df: pd.DataFrame,
    method_1: str,
    method_2: str,
    rank_1_exists: str,
):
    """
    Input:
        - rank_1_exists: "yes" if you compare both monopolar estimation methods
                        "no" if you compare the best_bssu_method to the monopolar estimation methods

    from a comparison result dataframe
        - count how often rank 1 contacts are the same
        - count how often there is at least one matching contact in compare_rank_1_and_2_contact

    """
    # sample size
    ses_count = ses_df["subject_hemisphere"].count()

    if rank_1_exists == "yes":
        # count how often compare_rank_1_contact same
        same_rank_1 = ses_df.loc[ses_df.compare_rank_1_contact == "same"]
        same_rank_1 = same_rank_1["subject_hemisphere"].count()
        percentage_same_rank_1 = same_rank_1 / ses_count

        # count how often compare_rank_1_contact same
        both_contacts_matching = ses_df.loc[ses_df.both_contacts_matching == "yes"]
        both_contacts_matching = both_contacts_matching["subject_hemisphere"].count()
        percentage_both_contacts_matching = both_contacts_matching / ses_count

        # count how often there is at least one matching contact in compare_rank_1_and_2_contact
        at_least_1_same = ses_df.loc[
            ses_df.compare_rank_1_and_2_contacts == "at_least_one_contact_match"
        ]
        at_least_1_same = at_least_1_same["subject_hemisphere"].count()
        percentage_at_least_1_same = at_least_1_same / ses_count

        sample_size_dict = {
            "session": [ses],
            "method_1": [method_1],
            "method_2": [method_2],
            "sample_size": [ses_count],
            "same_rank_1": [same_rank_1],
            "percentage_same_rank_1": [percentage_same_rank_1],
            "at_least_1_contact_same": [at_least_1_same],
            "percentage_at_least_one_same_contact_rank_1_and_2": [
                percentage_at_least_1_same
            ],
            "both_contacts_matching": [both_contacts_matching],
            "percentage_both_contacts_matching": [percentage_both_contacts_matching],
        }
        sample_size_single_df = pd.DataFrame(sample_size_dict)

    elif rank_1_exists == "no":
        # count how often compare_rank_1_contact same
        both_contacts_matching = ses_df.loc[ses_df.both_contacts_matching == "yes"]
        both_contacts_matching = both_contacts_matching["subject_hemisphere"].count()
        percentage_both_contacts_matching = both_contacts_matching / ses_count

        # count how often there is at least one matching contact in compare_rank_1_and_2_contact
        at_least_1_same = ses_df.loc[
            ses_df.at_least_1_contact_matching == "at_least_one_contact_match"
        ]
        at_least_1_same = at_least_1_same["subject_hemisphere"].count()
        percentage_at_least_1_same = at_least_1_same / ses_count

        sample_size_dict = {
            "method_1": [method_1],
            "method_2": [method_2],
            "session": [ses],
            "sample_size": [ses_count],
            "both_contacts_matching": [both_contacts_matching],
            "percentage_both_contacts_matching": [percentage_both_contacts_matching],
            "at_least_1_contact_same": [at_least_1_same],
            "percentage_at_least_one_same_contact_rank_1_and_2": [
                percentage_at_least_1_same
            ],
        }

        sample_size_single_df = pd.DataFrame(sample_size_dict)

    return sample_size_single_df


def load_comparison_result_DF(
    method_comparison: str,
    comparison_file: str,
    clinical_session: str,
    percept_session: str,
    fooof_version: str,
    new_reference: str,
    bssu_version: str,
):
    """
    Input:
        - method_comparison
        - comparison_file: "rank" or "correlation"
        - clinical_session:
        - percept_session:

    """
    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    if comparison_file == "rank":
        filename = f"{comparison_file}_group_comparison_all_clinical_{clinical_session}_percept_{percept_session}{external_extension}_{ext_fooof_re_ref}{fooof_version}.pickle"

    elif comparison_file == "correlation":
        filename = f"{comparison_file}_group_comparison_all_externalized_percept_{percept_session}{external_extension}_{ext_fooof_re_ref}{fooof_version}.pickle"

    filepath = os.path.join(GROUP_RESULTS_PATH, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(
            file
        )  # data is a Dataframe with method_comparison as column "method_comparison"

    data = data.loc[data.method_comparison == method_comparison]

    return data


def get_comparison_matrix_for_heatmap_from_DF(
    value_to_plot: str,
    clinical_session: str,
    percept_session: str,
    rank_or_correlation: str,
    fooof_version: str,
    new_reference: str,
    bssu_version: str,
):
    """

    Creates a 5x5 comparison matrix of the input value
    value_to_plot must be a column name in the Excel sample size file loaded with load_sample_size_result()

    Input:
        - value_to_plot: e.g. "percentage_at_least_one_same_contact_rank_1_and_2", "percentage_both_contacts_matching"
        - clinical_session: "fu3m", "fu12m", "fu18or24m"
        - percept_session: "postop", "fu3m", "fu12m", "fu18or24m"
        - rank_or_correlation: "rank", "correlation"


    """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    def populate_matrix(matrix, dict, list_of_methods):
        for i in range(len(list_of_methods)):
            for j in range(i, len(list_of_methods)):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    method1 = list_of_methods[i]
                    method2 = list_of_methods[j]
                    key1 = f"{method1}_{method2}"
                    key2 = f"{method2}_{method1}"
                    if key1 in dict:
                        matrix[i, j] = dict[key1]
                        matrix[j, i] = dict[key1]
                    elif key2 in dict:
                        matrix[i, j] = dict[key2]
                        matrix[j, i] = dict[key2]
        return matrix

    comparison_dict = {}
    sample_size = {}

    rank_comparison = [
        "percentage_at_least_one_same_contact_rank_1_and_2",
        "percentage_both_contacts_matching",
    ]
    correlation_comparison = ["estimated_beta_spearman", "normalized_beta_pearson"]

    if rank_or_correlation == "rank":
        method_comparisons = [
            f"euclidean_directional{external_extension}_JLB_directional{external_extension}",
            f"euclidean_directional{external_extension}_best_bssu_contacts{external_extension}",
            f"euclidean_directional{external_extension}_detec_strelow_contacts{external_extension}",
            f"JLB_directional{external_extension}_best_bssu_contacts{external_extension}",
            f"JLB_directional{external_extension}_detec_strelow_contacts{external_extension}",
            f"detec_strelow_contacts{external_extension}_best_bssu_contacts{external_extension}",
            f"externalized_fooof_detec_strelow_contacts{external_extension}",
            f"externalized_ssd_detec_strelow_contacts{external_extension}",
            f"JLB_directional{external_extension}_externalized_fooof",
            f"JLB_directional{external_extension}_externalized_ssd",
            f"euclidean_directional{external_extension}_externalized_fooof",
            f"euclidean_directional{external_extension}_externalized_ssd",
            f"externalized_fooof_best_bssu_contacts{external_extension}",
            f"externalized_ssd_best_bssu_contacts{external_extension}",
            "externalized_fooof_externalized_ssd",
            "best_clinical_contacts_externalized_ssd",
            "best_clinical_contacts_externalized_fooof",
            f"best_clinical_contacts_JLB_directional{external_extension}",
            f"best_clinical_contacts_euclidean_directional{external_extension}",
            f"best_clinical_contacts_best_bssu_contacts{external_extension}",
            f"best_clinical_contacts_detec_strelow_contacts{external_extension}",
        ]

        list_of_methods = [
            "externalized_ssd",
            "externalized_fooof",
            f"JLB_directional{external_extension}",
            f"euclidean_directional{external_extension}",
            f"best_bssu_contacts{external_extension}",
            f"detec_strelow_contacts{external_extension}",
            "best_clinical_contacts",
        ]

        # Initialize an empty 7x7 matrix
        comparison_matrix = np.zeros((7, 7))
        sample_size_matrix = np.zeros((7, 7))

    elif rank_or_correlation == "correlation":
        method_comparisons = [
            f"euclidean_directional{external_extension}_JLB_directional{external_extension}",
            f"euclidean_directional{external_extension}_detec_strelow_contacts{external_extension}",
            f"JLB_directional{external_extension}_detec_strelow_contacts{external_extension}",
            f"detec_strelow_contacts{external_extension}_externalized_fooof",
            f"detec_strelow_contacts{external_extension}_externalized_ssd",
            f"JLB_directional{external_extension}_externalized_fooof",
            f"JLB_directional{external_extension}_externalized_ssd",
            f"euclidean_directional{external_extension}_externalized_fooof",
            f"euclidean_directional{external_extension}_externalized_ssd",
            "externalized_fooof_externalized_ssd",
        ]

        list_of_methods = [
            "externalized_ssd",
            "externalized_fooof",
            f"JLB_directional{external_extension}",
            f"euclidean_directional{external_extension}",
            f"detec_strelow_contacts{external_extension}",
        ]

        # Initialize an empty 5x5 matrix
        comparison_matrix = np.zeros((5, 5))
        sample_size_matrix = np.zeros((5, 5))

    # create dictionary with method comparisons as keys and the percentage of at least 1 same rank 1 or 2 contact as value
    for comp in method_comparisons:
        # load the percentage_at_least_one_same_contact_rank_1_and_2
        # from each comparison of methods
        comparison_df = load_comparison_result_DF(
            method_comparison=comp,
            comparison_file=rank_or_correlation,
            clinical_session=clinical_session,
            percept_session=percept_session,
            fooof_version=fooof_version,
            new_reference=new_reference,
            bssu_version=bssu_version,
        )

        # for correlation first select the rows with relevant values
        if rank_or_correlation == "correlation":
            comparison_df = comparison_df.loc[
                comparison_df.correlation == value_to_plot
            ]  # only row with specific correlation: spearman or pearson
            comparison_dict[comp] = comparison_df["percentage_significant"].values[0]

        elif rank_or_correlation == "rank":
            comparison_dict[comp] = comparison_df[value_to_plot].values[0]

        sample_size[comp] = comparison_df.sample_size.values[0]

    # Populate the matrix with comparison values
    comparison_matrix = populate_matrix(
        comparison_matrix, comparison_dict, list_of_methods
    )
    sample_size_matrix = populate_matrix(
        sample_size_matrix, sample_size, list_of_methods
    )

    # Now, comparison_matrix contains the nicely structured comparison values
    return {
        "comparison_matrix": comparison_matrix,
        "comparison_dict": comparison_dict,
        "sample_size": sample_size,
        "list_of_methods": list_of_methods,
        "sample_size_matrix": sample_size_matrix,
    }
