""" Comparisons between monopolar estimation methods """


import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import io_monopolar_comparison as io_monopolar_comparison
from ..utils import loadResults as loadResults
from ..comparison_analysis import monopol_comparison_tests as monopol_comparison_tests


GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

INCL_SESSIONS = ["postop", "fu3m", "fu12m", "fu18or24m"]
SEGMENTAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]

# no beta either in externalized or percept -> exclude for comparison externalized vs percept
# EXCLUDED_NO_BETA_EXT_OR_PERCEPT = [
#     "028_Right",
#     "028_Left",
#     "029_Right",
#     "029_Left",
#     "030_Left",
#     "032_Right",
#     "032_Left",
#     "048_Right",
#     "048_Left",
#     "049_Right",
#     "049_Left",
#     "052_Left",
#     "056_Left",
#     "061_Right",
#     "071_Right",
#     "072_Left",
#     "075_Right",
# ]


# only hemispheres are included, that have beta power in the externalized recording!!!!
EXCLUDED_NO_BETA_EXT_OR_PERCEPT = [
    "028_Right",
    "028_Left", # unsure about 029_Left w artifact in raw time series, 029_Right
    "030_Left",
    "032_Right",
    "032_Left",
    "048_Right",
    "048_Left",
    "049_Left",
    #"052_Left",
    "061_Right",  # NaNs, don't know why because nice beta
    #"069_Left", # zu viele Bewegungsartifakte, too short for 20sec analysis
    "071_Right",
    "072_Right",
    #"072_Left",
    "075_Right", # beta there initially but gone after re-ref
    "077_Left",
]


# no beta only in externalized -> exclude for comparison externalized vs externalized
EXCLUDED_NO_BETA_EXT = [
    "028_Right",
    "028_Left", # unsure about 029_Left w artifact in raw time series, 029_Right
    "030_Left",
    "032_Right",
    "032_Left",
    "048_Right",
    "048_Left",
    "049_Left",
    #"052_Left",
    "061_Right",  # NaNs, don't know why because nice beta
    #"069_Left", # zu viele Bewegungsartifakte, too short for 20sec analysis
    "071_Right",
    "072_Right",
    #"072_Left",
    "075_Right", # beta there initially but gone after re-ref
    "077_Left",
]

# no beta only in percept postop -> exclude for comparison percept postop vs percept postop
EXCLUDED_NO_BETA_PERCEPT = [
    # "028_Left",
    # "029_Right",
    # "029_Left",
    # "032_Right",
    # "048_Right",
    # "048_Left",
    # "049_Left",
    # "056_Left",
    # "061_Right",
    # "071_Right",
    # "075_Right",
]


def exclude_subjects(df: pd.DataFrame, exclude_list: list):
    """
    Exclude subjects from Dataframe

    Input:
        - df: Dataframe with subject_hemisphere column
        - exclude_list: list of subject_hemispheres to exclude: EXCLUDED_NO_BETA_EXT_OR_PERCEPT, EXCLUDED_NO_BETA_EXT or EXCLUDED_NO_BETA_PERCEPT
    """

    df = df[~((df.subject_hemisphere.isin(exclude_list)) & (df.session == "postop"))]

    return df


def correlation_monopol_fooof_beta_methods(
    method_1: str, method_2: str, fooof_version: str, bssu_version: str
):
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    watch out: for detec_strelow_contacts here we use the file containing estimated monopolar beta values of all directional contacts,
    for ranks not optimal, because we don't go first rank levels, second rank directions of this level

    Input: define methods to compare
        - method_1: "JLB_directional", "euclidean_directional", "detec_strelow_contacts"
        - method_2: "JLB_directional", "euclidean_directional", "detec_strelow_contacts"
        - fooof_version: "v1", "v2"
    """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    # results
    results_DF = pd.DataFrame()
    sample_size_df = pd.DataFrame()
    corr_ses_df = pd.DataFrame()

    # get data from method 1
    if method_1 == f"JLB_directional{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_JLB_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_1_data = io_monopolar_comparison.load_JLB_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_1 == f"euclidean_directional{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_euclidean_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_1_data = io_monopolar_comparison.load_euclidean_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_1 == f"detec_strelow_contacts{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_detec_strelow_beta_ranks(
                fooof_version=fooof_version,
                level_first_or_all_directional="all_directional",
            )

        elif bssu_version == "externalized":
            method_1_data = (
                io_monopolar_comparison.load_detec_strelow_beta_ranks_externalized_bssu(
                    fooof_version=fooof_version,
                    level_first_or_all_directional="all_directional",
                )
            )

        print("Strelow method: all directional contact values")
        # monopolar beta average for all directional contacts

    # get data from method 2
    if method_2 == f"JLB_directional{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_JLB_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_2_data = io_monopolar_comparison.load_JLB_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_2 == f"euclidean_directional{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_euclidean_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_2_data = io_monopolar_comparison.load_euclidean_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_2 == f"detec_strelow_contacts{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_detec_strelow_beta_ranks(
                fooof_version=fooof_version,
                level_first_or_all_directional="all_directional",
            )

        elif bssu_version == "externalized":
            method_2_data = (
                io_monopolar_comparison.load_detec_strelow_beta_ranks_externalized_bssu(
                    fooof_version=fooof_version,
                    level_first_or_all_directional="all_directional",
                )
            )
        print("Strelow method: all directional contact values")
        # monopolar beta average for all directional contacts

    # exclude subject hemispheres without beta
    if bssu_version == "percept":
        method_1_data = exclude_subjects(
            df=method_1_data, exclude_list=EXCLUDED_NO_BETA_PERCEPT
        )
        method_2_data = exclude_subjects(
            df=method_2_data, exclude_list=EXCLUDED_NO_BETA_PERCEPT
        )

    elif bssu_version == "externalized":
        method_1_data = exclude_subjects(
            df=method_1_data, exclude_list=EXCLUDED_NO_BETA_EXT 
        )
        method_2_data = exclude_subjects(
            df=method_2_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )

    # Perform 3 versions of correlation tests for every session separately and within each STN
    if bssu_version == "percept":
        for ses in INCL_SESSIONS:
            method_1_session = method_1_data.loc[method_1_data.session == ses]
            method_2_session = method_2_data.loc[method_2_data.session == ses]

            ses_result_df = monopol_comparison_tests.correlation_tests_percept_methods(
                method_1=method_1,
                method_2=method_2,
                method_1_df=method_1_session,
                method_2_df=method_2_session,
                ses=ses,
            )
            results_DF = pd.concat([results_DF, ses_result_df], ignore_index=True)

    elif bssu_version == "externalized":
        ses_result_df = monopol_comparison_tests.correlation_tests_percept_methods(
            method_1=method_1,
            method_2=method_2,
            method_1_df=method_1_data,
            method_2_df=method_2_data,
            ses="postop",
        )
        results_DF = pd.concat([results_DF, ses_result_df], ignore_index=True)

    # save Dataframe to Excel
    results_DF_copy = results_DF.copy()

    correlation_results = [
        "estimated_beta_spearman",
        "normalized_beta_pearson",
        "cluster_beta_spearman",
    ]

    for corr in correlation_results:
        # add new column: significant yes, no
        significant_correlation = results_DF_copy[f"{corr}_pval"] < 0.05
        results_DF_copy[f"significant_{corr}"] = [
            "yes" if cond else "no" for cond in significant_correlation
        ]

    # save as Excel
    io_monopolar_comparison.save_result_excel(
        result_df=results_DF_copy,
        filename=f"fooof_monopol_beta_correlations_per_stn_{method_1}_{method_2}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    # get sample size
    if bssu_version == "percept":
        for ses in INCL_SESSIONS:
            ses_df = results_DF_copy.loc[results_DF_copy.session == ses]
            ses_count = ses_df["session"].count()

            sample_size_single_df = (
                monopol_comparison_tests.get_sample_size_percept_methods(
                    ses=ses,
                    ses_df=ses_df,
                    method_1=method_1,
                    method_2=method_2,
                    rank_1_exists="yes",
                )
            )
            sample_size_df = pd.concat(
                [sample_size_df, sample_size_single_df], ignore_index=True
            )

            # correlation results per correlation test
            for corr in correlation_results:
                corr_mean = ses_df[f"{corr}_r"].mean()
                corr_median = ses_df[f"{corr}_r"].median()
                corr_std = np.std(ses_df[f"{corr}_r"])

                # calculate how often significant?
                significant_count = ses_df.loc[ses_df[f"significant_{corr}"] == "yes"]
                significant_count = significant_count["session"].count()
                percentage_significant = significant_count / ses_count

                corr_ses_result = {
                    "session": [ses],
                    "method_1": [method_1],
                    "method_2": [method_2],
                    "sample_size": [ses_count],
                    "correlation": [corr],
                    "corr_mean": [corr_mean],
                    "corr_median": [corr_median],
                    "corr_std": [corr_std],
                    "significant_count": [significant_count],
                    "percentage_significant": [percentage_significant],
                }
                corr_ses_single_df = pd.DataFrame(corr_ses_result)

                corr_ses_df = pd.concat(
                    [corr_ses_df, corr_ses_single_df], ignore_index=True
                )

    elif bssu_version == "externalized":
        ses_count = results_DF_copy["session"].count()

        sample_size_single_df = (
            monopol_comparison_tests.get_sample_size_percept_methods(
                ses="postop",
                ses_df=results_DF_copy,
                method_1=method_1,
                method_2=method_2,
                rank_1_exists="yes",
            )
        )
        sample_size_df = pd.concat(
            [sample_size_df, sample_size_single_df], ignore_index=True
        )

        # correlation results per correlation test
        for corr in correlation_results:
            corr_mean = results_DF_copy[f"{corr}_r"].mean()
            corr_median = results_DF_copy[f"{corr}_r"].median()
            corr_std = np.std(results_DF_copy[f"{corr}_r"])

            # calculate how often significant?
            significant_count = results_DF_copy.loc[
                results_DF_copy[f"significant_{corr}"] == "yes"
            ]
            significant_count = significant_count["session"].count()
            percentage_significant = significant_count / ses_count

            corr_ses_result = {
                "session": ["postop"],
                "method_1": [method_1],
                "method_2": [method_2],
                "sample_size": [ses_count],
                "correlation": [corr],
                "corr_mean": [corr_mean],
                "corr_median": [corr_median],
                "corr_std": [corr_std],
                "significant_count": [significant_count],
                "percentage_significant": [percentage_significant],
            }
            corr_ses_single_df = pd.DataFrame(corr_ses_result)

            corr_ses_df = pd.concat(
                [corr_ses_df, corr_ses_single_df], ignore_index=True
            )

    # # save session Dataframes as Excel files
    # helpers.save_result_excel(
    #     result_df=sample_size_df,
    #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{method_1}_{method_2}_{fooof_version}",
    #     sheet_name="fooof_monopol_beta_correlations",
    # )

    # helpers.save_result_excel(
    #     result_df=corr_ses_df,
    #     filename=f"fooof_monopol_beta_correlations_corr_ses_df_{method_1}_{method_2}_{fooof_version}",
    #     sheet_name="fooof_monopol_beta_correlations",
    # )

    return {
        "single_stn_results": results_DF_copy,
        "rank": sample_size_df,
        "correlation": corr_ses_df,
    }


def compare_method_to_best_bssu_contact_pair(fooof_version: str, bssu_version: str):
    """
    No need for this function, consider to delete it

    Comparing the selected best BSSU contact pair to the two directional contacts selected with method Euclidean and JLB and detec

    Input:
        - fooof_version: "v1", "v2"

    """

    sample_size_df = pd.DataFrame()
    comparison_result = pd.DataFrame()

    # get data
    JLB_method = io_monopolar_comparison.load_JLB_method(fooof_version=fooof_version)
    Euclidean_method = io_monopolar_comparison.load_euclidean_method(
        fooof_version=fooof_version
    )
    detec_method = io_monopolar_comparison.load_detec_strelow_beta_ranks(
        fooof_version=fooof_version, level_first_or_all_directional="level_first"
    )  # only directional contact ranks of level rank 1

    best_bssu_contact_data = io_monopolar_comparison.load_best_bssu_method(
        fooof_version=fooof_version
    )

    # exclude subject hemispheres without beta
    JLB_method = exclude_subjects(df=JLB_method, exclude_list=EXCLUDED_NO_BETA_PERCEPT)
    Euclidean_method = exclude_subjects(
        df=Euclidean_method, exclude_list=EXCLUDED_NO_BETA_PERCEPT
    )
    detec_method = exclude_subjects(
        df=detec_method, exclude_list=EXCLUDED_NO_BETA_PERCEPT
    )
    best_bssu_contact_data = exclude_subjects(
        df=best_bssu_contact_data, exclude_list=EXCLUDED_NO_BETA_PERCEPT
    )

    three_methods = [1, 2, 3]

    for run in three_methods:
        if run == 1:
            method_data = JLB_method
            method = "JLB_directional"

        elif run == 2:
            method_data = Euclidean_method
            method = "euclidean_directional"

        elif run == 3:
            method_data = detec_method
            method = "detec_strelow_contacts"

        for ses in INCL_SESSIONS:
            ses_data_method = method_data.loc[method_data.session == ses]
            ses_data_best_bssu = best_bssu_contact_data.loc[
                best_bssu_contact_data.session == ses
            ]

            comparison_single_result = (
                monopol_comparison_tests.rank_comparison_percept_methods(
                    method_1=method,
                    method_2="best_bssu_contacts",
                    method_1_df=ses_data_method,
                    method_2_df=ses_data_best_bssu,
                    ses=ses,
                    bssu_version=bssu_version,
                )
            )

            comparison_result = pd.concat(
                [comparison_result, comparison_single_result], ignore_index=True
            )

        # # save as Excel
        # helpers.save_result_excel(
        #     result_df=comparison_result,
        #     filename=f"fooof_monopol_best_contacts_per_stn_{method}_best_bssu_contacts_{fooof_version}",
        #     sheet_name="fooof_monopol_best_contacts",
        # )

        results_DF_copy = comparison_result.copy()

        for ses in INCL_SESSIONS:
            ses_result = results_DF_copy.loc[results_DF_copy.session == ses]

            # get sample size
            sample_size_single_df = (
                monopol_comparison_tests.get_sample_size_percept_methods(
                    ses=ses,
                    ses_df=ses_result,
                    rank_1_exists="no",
                    method_1=method,
                    method_2="best_bssu_contacts",
                )
            )

            # sample_size_single_df = pd.DataFrame(sample_size_dict)
            sample_size_df = pd.concat([sample_size_df, sample_size_single_df])

        # # save session Dataframes as Excel files
        # helpers.save_result_excel(
        #     result_df=sample_size_df,
        #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{method}_best_bssu_contacts_{fooof_version}",
        #     sheet_name="fooof_monopol_beta_correlations",
        # )

    return {
        "single_stn_results": results_DF_copy,
        "rank": sample_size_df,
    }


def rank_comparison_percept_methods(
    method_1: str, method_2: str, fooof_version: str, bssu_version: str
):
    """
    Comparing the ranks of all percept methods to each other (NO correlation)

    Input:
        - fooof_version: "v1", "v2"

    """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    sample_size_df = pd.DataFrame()
    comparison_result = pd.DataFrame()

    # get data from method 1
    if method_1 == f"JLB_directional{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_JLB_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_1_data = io_monopolar_comparison.load_JLB_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_1 == f"euclidean_directional{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_euclidean_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_1_data = io_monopolar_comparison.load_euclidean_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_1 == f"detec_strelow_contacts{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_detec_strelow_beta_ranks(
                fooof_version=fooof_version,
                level_first_or_all_directional="level_first",
            )

        elif bssu_version == "externalized":
            method_1_data = (
                io_monopolar_comparison.load_detec_strelow_beta_ranks_externalized_bssu(
                    fooof_version=fooof_version,
                    level_first_or_all_directional="level_first",
                )
            )

        print("Strelow method: level first, 3 directional ranks of level rank 1")

    elif method_1 == f"best_bssu_contacts{external_extension}":
        if bssu_version == "percept":
            method_1_data = io_monopolar_comparison.load_best_bssu_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_1_data = io_monopolar_comparison.load_best_externalized_bssu(
                fooof_version=fooof_version
            )

    # get data from method 2
    if method_2 == f"JLB_directional{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_JLB_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_2_data = io_monopolar_comparison.load_JLB_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_2 == f"euclidean_directional{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_euclidean_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_2_data = io_monopolar_comparison.load_euclidean_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method_2 == f"detec_strelow_contacts{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_detec_strelow_beta_ranks(
                fooof_version=fooof_version,
                level_first_or_all_directional="level_first",
            )

        elif bssu_version == "externalized":
            method_2_data = (
                io_monopolar_comparison.load_detec_strelow_beta_ranks_externalized_bssu(
                    fooof_version=fooof_version,
                    level_first_or_all_directional="level_first",
                )
            )

        print("Strelow method: level first, 3 directional ranks of level rank 1")

    elif method_2 == f"best_bssu_contacts{external_extension}":
        if bssu_version == "percept":
            method_2_data = io_monopolar_comparison.load_best_bssu_method(
                fooof_version=fooof_version
            )

        elif bssu_version == "externalized":
            method_2_data = io_monopolar_comparison.load_best_externalized_bssu(
                fooof_version=fooof_version
            )

    # exclude subject hemispheres without beta
    if bssu_version == "percept":
        method_1_data = exclude_subjects(
            df=method_1_data, exclude_list=EXCLUDED_NO_BETA_PERCEPT
        )
        method_2_data = exclude_subjects(
            df=method_2_data, exclude_list=EXCLUDED_NO_BETA_PERCEPT
        )

    elif bssu_version == "externalized":
        method_1_data = exclude_subjects(
            df=method_1_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )
        method_2_data = exclude_subjects(
            df=method_2_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )

    # Perform 3 versions of correlation tests for every session separately and within each STN

    if bssu_version == "percept":
        for ses in INCL_SESSIONS:
            method_1_session = method_1_data.loc[method_1_data.session == ses]
            method_2_session = method_2_data.loc[method_2_data.session == ses]

            ses_result_df = monopol_comparison_tests.rank_comparison_percept_methods(
                method_1=method_1,
                method_2=method_2,
                method_1_df=method_1_session,
                method_2_df=method_2_session,
                ses=ses,
                bssu_version=bssu_version,
            )

            comparison_result = pd.concat(
                [comparison_result, ses_result_df], ignore_index=True
            )

    elif bssu_version == "externalized":
        ses_result_df = monopol_comparison_tests.rank_comparison_percept_methods(
            method_1=method_1,
            method_2=method_2,
            method_1_df=method_1_data,
            method_2_df=method_2_data,
            ses="postop",
            bssu_version=bssu_version,
        )

        comparison_result = pd.concat(
            [comparison_result, ses_result_df], ignore_index=True
        )

    # save as Excel
    io_monopolar_comparison.save_result_excel(
        result_df=comparison_result,
        filename=f"fooof_monopol_best_contacts_per_stn_{method_1}_{method_2}{external_extension}_{fooof_version}",
        sheet_name="fooof_monopol_best_contacts",
    )

    results_DF_copy = comparison_result.copy()

    if bssu_version == "percept":
        for ses in INCL_SESSIONS:
            ses_result = results_DF_copy.loc[results_DF_copy.session == ses]

            # get sample size
            sample_size_single_df = (
                monopol_comparison_tests.get_sample_size_percept_methods(
                    ses=ses,
                    ses_df=ses_result,
                    rank_1_exists="no",
                    method_1=method_1,
                    method_2=method_2,
                )
            )

            # sample_size_single_df = pd.DataFrame(sample_size_dict)
            sample_size_df = pd.concat([sample_size_df, sample_size_single_df])

    elif bssu_version == "externalized":
        # get sample size
        sample_size_single_df = (
            monopol_comparison_tests.get_sample_size_percept_methods(
                ses="postop",
                ses_df=results_DF_copy,
                rank_1_exists="no",
                method_1=method_1,
                method_2=method_2,
            )
        )

        # sample_size_single_df = pd.DataFrame(sample_size_dict)
        sample_size_df = pd.concat([sample_size_df, sample_size_single_df])

    # # save session Dataframes as Excel files
    # helpers.save_result_excel(
    #     result_df=sample_size_df,
    #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{method_1}_{method_2}_{fooof_version}",
    #     sheet_name="fooof_monopol_beta_correlations",
    # )

    return {
        "single_stn_results": results_DF_copy,
        "rank": sample_size_df,
    }


def percept_vs_externalized(
    method: str,
    percept_session: str,
    fooof_version: str,
    strelow_level_first: str,
    externalized_version: str,
    new_reference: str,
    bssu_version: str,
    reference=None,
):
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - method: "JLB_directional", "euclidean_directional", "detec_strelow_contacts", "best_bssu_contacts"
        - percept_session: "fu3m", "fu12m", "fu18or24m"
        - fooof_version: "v1", "v2"
        - strelow_level_first: "level_first" if you want to use this method for ranking, or "all_directional" if you want the beta average of all directional contacts
        - externalized_version: "externalized_fooof", "externalized_ssd"
        - bssu_version: "externalized", "percept"
        - reference: str "bipolar_to_lowermost" or "no"
    """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    if strelow_level_first == "level_first":
        methods_for_spearman = [
            f"JLB_directional{external_extension}",
            f"euclidean_directional{external_extension}",
        ]
        methods_without_spearman = [
            f"best_bssu_contacts{external_extension}",
            f"detec_strelow_contacts{external_extension}",
        ]

    elif strelow_level_first == "all_directional":
        methods_for_spearman = [
            f"JLB_directional{external_extension}",
            f"euclidean_directional{external_extension}",
            f"detec_strelow_contacts{external_extension}",
        ]
        methods_without_spearman = [f"best_bssu_contacts{external_extension}"]

    # get only postop data from method
    if method == f"JLB_directional{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_JLB_method(
                fooof_version=fooof_version
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = io_monopolar_comparison.load_JLB_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method == f"euclidean_directional{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_euclidean_method(
                fooof_version=fooof_version
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = io_monopolar_comparison.load_euclidean_externalized_bssu(
                fooof_version=fooof_version
            )

    elif method == f"detec_strelow_contacts{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_detec_strelow_beta_ranks(
                fooof_version=fooof_version,
                level_first_or_all_directional=strelow_level_first,
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = (
                io_monopolar_comparison.load_detec_strelow_beta_ranks_externalized_bssu(
                    fooof_version=fooof_version,
                    level_first_or_all_directional=strelow_level_first,
                )
            )

    elif method == f"best_bssu_contacts{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_best_bssu_method(
                fooof_version=fooof_version
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = io_monopolar_comparison.load_best_externalized_bssu(
                fooof_version=fooof_version
            )

    # get data from externalized LFP
    if externalized_version == "externalized_fooof":
        externalized_data = io_monopolar_comparison.load_externalized_fooof_data(
            fooof_version=fooof_version,
            new_reference=new_reference,
            reference=reference,
        )

    elif externalized_version == "externalized_ssd":
        externalized_data = io_monopolar_comparison.load_externalized_ssd_data(
            reference=reference
        )

    # exclude subject hemispheres without beta
    if bssu_version == "percept":
        method_data = exclude_subjects(
            df=method_data, exclude_list=EXCLUDED_NO_BETA_EXT_OR_PERCEPT
        )
        externalized_data = exclude_subjects(
            df=externalized_data, exclude_list=EXCLUDED_NO_BETA_EXT_OR_PERCEPT
        )

    elif bssu_version == "externalized":
        method_data = exclude_subjects(
            df=method_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )
        externalized_data = exclude_subjects(
            df=externalized_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )

    # Perform comparison for every session separately and within each STN
    if method in methods_for_spearman:
        result_df = monopol_comparison_tests.correlation_tests_percept_methods(
            method_1=method,
            method_2=externalized_version,
            method_1_df=method_data,
            method_2_df=externalized_data,
            ses="postop",
        )

        # save Dataframe to Excel
        results_DF_copy = result_df.copy()

        correlation_results = [
            "estimated_beta_spearman",
            "normalized_beta_pearson",
            "cluster_beta_spearman",
        ]

        for corr in correlation_results:
            # add new column: significant yes, no
            significant_correlation = results_DF_copy[f"{corr}_pval"] < 0.05
            results_DF_copy[f"significant_{corr}"] = [
                "yes" if cond else "no" for cond in significant_correlation
            ]

        # save as Excel
        io_monopolar_comparison.save_result_excel(
            result_df=results_DF_copy,
            filename=f"fooof_monopol_beta_correlations_per_stn_{method}_{externalized_version}{ext_fooof_re_ref}{reference_name}_{fooof_version}_{percept_session}{external_extension}",
            sheet_name="fooof_monopol_beta_correlations",
        )

        # get sample size
        count = results_DF_copy["subject_hemisphere"].count()

        sample_size_df = monopol_comparison_tests.get_sample_size_percept_methods(
            ses="postop",
            ses_df=results_DF_copy,
            method_1=method,
            method_2=externalized_version,
            rank_1_exists="yes",
        )

        # correlation results per correlation test
        corr_ses_df = pd.DataFrame()

        for corr in correlation_results:
            corr_mean = results_DF_copy[f"{corr}_r"].mean()
            corr_median = results_DF_copy[f"{corr}_r"].median()
            corr_std = np.std(results_DF_copy[f"{corr}_r"])

            # calculate how often significant?
            significant_count = results_DF_copy.loc[
                results_DF_copy[f"significant_{corr}"] == "yes"
            ]
            significant_count = significant_count["session"].count()
            percentage_significant = significant_count / count

            corr_ses_result = {
                "session": ["postop"],
                "percept_session": [percept_session],
                "method_1": [method],
                "method_2": [externalized_version],
                "sample_size": [count],
                "correlation": [corr],
                "corr_mean": [corr_mean],
                "corr_median": [corr_median],
                "corr_std": [corr_std],
                "significant_count": [significant_count],
                "percentage_significant": [percentage_significant],
            }
            corr_ses_single_df = pd.DataFrame(corr_ses_result)

            corr_ses_df = pd.concat(
                [corr_ses_df, corr_ses_single_df], ignore_index=True
            )

        # # save session Dataframes as Excel files
        # helpers.save_result_excel(
        #     result_df=sample_size_df,
        #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{method}_{externalized_version}{reference_name}_{fooof_version}_{percept_session}",
        #     sheet_name="fooof_monopol_beta_correlations",
        # )

        # helpers.save_result_excel(
        #     result_df=corr_ses_df,
        #     filename=f"fooof_monopol_beta_correlations_corr_ses_df_{method}_{externalized_version}{reference_name}_{fooof_version}_{percept_session}",
        #     sheet_name="fooof_monopol_beta_correlations",
        # )

    elif method in methods_without_spearman:
        corr_ses_df = pd.DataFrame()

        result_df = monopol_comparison_tests.rank_comparison_percept_methods(
            method_1=externalized_version,
            method_2=method,
            method_1_df=externalized_data,
            method_2_df=method_data,
            ses="postop",
            bssu_version=bssu_version,
        )

        # save as Excel
        io_monopolar_comparison.save_result_excel(
            result_df=result_df,
            filename=f"fooof_monopol_beta_correlations_per_stn_{method}_{externalized_version}{ext_fooof_re_ref}{reference_name}_{fooof_version}_{percept_session}{external_extension}",
            sheet_name="fooof_monopol_best_contacts",
        )

        results_DF_copy = result_df.copy()

        # get sample size
        if method == f"best_bssu_contacts{external_extension}":
            rank_1_exists = "no"
        elif method == f"detec_strelow_contacts{external_extension}":
            rank_1_exists = "no"

        sample_size_df = monopol_comparison_tests.get_sample_size_percept_methods(
            ses="postop",
            ses_df=results_DF_copy,
            rank_1_exists=rank_1_exists,
            method_1=externalized_version,
            method_2=method,
        )

        # # save session Dataframes as Excel files
        # helpers.save_result_excel(
        #     result_df=sample_size_df,
        #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{method}_{externalized_version}{reference_name}_{fooof_version}_{percept_session}",
        #     sheet_name="fooof_monopol_beta_correlations",
        # )

    return {
        "single_stn_results": results_DF_copy,
        "rank": sample_size_df,
        "correlation": corr_ses_df,
    }


def externalized_versions_comparison(
    externalized_version_1: str,
    externalized_version_2: str,
    fooof_version: str,
    new_reference: str,
    reference=None,
):
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - externalized_version_1: "externalized_ssd", "externalized_fooof"
        - externalized_version_2: "externalized_ssd", "externalized_fooof"
        - fooof_version: "v1", "v2"
        - reference: "bipolar_to_lowermost" or "no"
        - new_reference: "no", "one_to_zero_two_to_three"
    """

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    # results
    # spearman_result = {}

    # sample_size_dict = {}

    # get data from externalized LFP version 1
    if externalized_version_1 == "externalized_fooof":
        externalized_data_1 = io_monopolar_comparison.load_externalized_fooof_data(
            fooof_version=fooof_version,
            new_reference=new_reference,
            reference=reference,
        )

    elif externalized_version_1 == "externalized_ssd":
        externalized_data_1 = io_monopolar_comparison.load_externalized_ssd_data(
            reference=reference
        )

    # get data from externalized LFP version 2
    if externalized_version_2 == "externalized_fooof":
        externalized_data_2 = io_monopolar_comparison.load_externalized_fooof_data(
            fooof_version=fooof_version,
            new_reference=new_reference,
            reference=reference,
        )

    elif externalized_version_2 == "externalized_ssd":
        externalized_data_2 = io_monopolar_comparison.load_externalized_ssd_data(
            reference=reference
        )

    # exclude subject hemispheres without beta
    externalized_data_1 = exclude_subjects(
        df=externalized_data_1, exclude_list=EXCLUDED_NO_BETA_EXT
    )
    externalized_data_2 = exclude_subjects(
        df=externalized_data_2, exclude_list=EXCLUDED_NO_BETA_EXT
    )

    # Perform spearman correlation for every session separately and within each STN
    result_df = monopol_comparison_tests.correlation_tests_percept_methods(
        method_1=externalized_version_1,
        method_2=externalized_version_2,
        method_1_df=externalized_data_1,
        method_2_df=externalized_data_2,
        ses="postop",
    )

    # save Dataframe to Excel
    results_DF_copy = result_df.copy()

    correlation_results = [
        "estimated_beta_spearman",
        "normalized_beta_pearson",
        "cluster_beta_spearman",
    ]

    for corr in correlation_results:
        # add new column: significant yes, no
        significant_correlation = results_DF_copy[f"{corr}_pval"] < 0.05
        results_DF_copy[f"significant_{corr}"] = [
            "yes" if cond else "no" for cond in significant_correlation
        ]

    # save as Excel
    io_monopolar_comparison.save_result_excel(
        result_df=results_DF_copy,
        filename=f"fooof_monopol_beta_correlations_per_stn_{externalized_version_1}_{externalized_version_2}{ext_fooof_re_ref}{reference_name}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    # get sample size
    count = results_DF_copy["subject_hemisphere"].count()

    sample_size_df = monopol_comparison_tests.get_sample_size_percept_methods(
        ses="postop",
        ses_df=results_DF_copy,
        rank_1_exists="yes",
        method_1=externalized_version_1,
        method_2=externalized_version_2,
    )

    # correlation results per correlation test
    corr_ses_df = pd.DataFrame()

    for corr in correlation_results:
        corr_mean = results_DF_copy[f"{corr}_r"].mean()
        corr_median = results_DF_copy[f"{corr}_r"].median()
        corr_std = np.std(results_DF_copy[f"{corr}_r"])

        # calculate how often significant?
        significant_count = results_DF_copy.loc[
            results_DF_copy[f"significant_{corr}"] == "yes"
        ]
        significant_count = significant_count["session"].count()
        percentage_significant = significant_count / count

        corr_ses_result = {
            "session": ["postop"],
            "method_1": [externalized_version_1],
            "method_2": [externalized_version_2],
            "sample_size": [count],
            "correlation": [corr],
            "corr_mean": [corr_mean],
            "corr_median": [corr_median],
            "corr_std": [f"{corr_std: .4f}"],
            "significant_count": [significant_count],
            "percentage_significant": [percentage_significant],
        }
        corr_ses_single_df = pd.DataFrame(corr_ses_result)

        corr_ses_df = pd.concat([corr_ses_df, corr_ses_single_df], ignore_index=True)

    # # save session Dataframes as Excel files
    # helpers.save_result_excel(
    #     result_df=sample_size_df,
    #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}",
    #     sheet_name="fooof_monopol_beta_correlations",
    # )

    # helpers.save_result_excel(
    #     result_df=corr_ses_df,
    #     filename=f"fooof_monopol_beta_correlations_corr_ses_df_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}",
    #     sheet_name="fooof_monopol_beta_correlations",
    # )

    return {
        "single_stn_results": results_DF_copy,
        "rank": sample_size_df,
        "correlation": corr_ses_df,
    }


############## TODO: ISSUE HERE ADD BSSU_VERSION ##############


def methods_vs_best_clinical_contacts(
    clinical_session: str,
    percept_session: str,
    method: str,
    rank_or_rel_above_70: str,
    fooof_version: str,
    bssu_version: str,
    new_reference: str,
    reference=None,
):
    """
    Comparing if the selected 2 best contacts of each method match with the best clinical contacts
        - at least one of the 2 contacts is clinically used
        - both selected contacts are clinically used

    Input
        - session_of_interest: "fu3m", "fu12m", fu18or24m"
        - percept_session: "fu3m", "fu12m", fu18or24m"
        - method: "externalized_ssd", "externalized_fooof", "JLB_directional", "euclidean_directional", "best_bssu_contacts", "detec_strelow_contacts"
        - rank_or_rel_above_70: "rank" if you want to compare to ranked 1 and 2, OR "rel_above_70" if you want to compare to rel contacts above 70
            "rank" does NOT work with best_bssu_contacts as method
        - fooof_version: "v2"
        - new_reference: "no", "one_to_zero_two_to_three"
        - bssu_version: "externalized", "percept"
        - reference: "bipolar_to_lowermost"


    """
    if bssu_version == "percept":
        external_extension = ""
        exclude = EXCLUDED_NO_BETA_PERCEPT

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"
        exclude = EXCLUDED_NO_BETA_EXT

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    results_DF = pd.DataFrame()

    best_clinical_contacts = io_monopolar_comparison.load_best_clinical_contacts()
    best_clinical_contacts_ses = best_clinical_contacts.loc[
        best_clinical_contacts.session == clinical_session
    ]

    # get data from the method
    if method == "externalized_fooof":
        method_data = io_monopolar_comparison.load_externalized_fooof_data(
            fooof_version=fooof_version,
            new_reference=new_reference,
            reference=reference,
        )
        method_data = exclude_subjects(
            df=method_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )

    elif method == "externalized_ssd":
        method_data = io_monopolar_comparison.load_externalized_ssd_data(
            reference=reference
        )
        method_data = exclude_subjects(
            df=method_data, exclude_list=EXCLUDED_NO_BETA_EXT
        )

    elif method == f"JLB_directional{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_JLB_method(
                fooof_version=fooof_version
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = io_monopolar_comparison.load_JLB_externalized_bssu(
                fooof_version=fooof_version
            )

        method_data = exclude_subjects(df=method_data, exclude_list=exclude)

    elif method == f"euclidean_directional{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_euclidean_method(
                fooof_version=fooof_version
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = io_monopolar_comparison.load_euclidean_externalized_bssu(
                fooof_version=fooof_version
            )

        method_data = exclude_subjects(df=method_data, exclude_list=exclude)

    elif method == f"best_bssu_contacts{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_best_bssu_method(
                fooof_version=fooof_version
            )
            method_data = method_data.loc[method_data.session == percept_session]

        elif bssu_version == "externalized":
            method_data = io_monopolar_comparison.load_best_externalized_bssu(
                fooof_version=fooof_version
            )

        method_data = exclude_subjects(df=method_data, exclude_list=exclude)

    elif method == f"detec_strelow_contacts{external_extension}":
        if bssu_version == "percept":
            method_data = io_monopolar_comparison.load_detec_strelow_beta_ranks(
                fooof_version=fooof_version,
                level_first_or_all_directional="level_first",
            )

        elif bssu_version == "externalized":
            method_data = (
                io_monopolar_comparison.load_detec_strelow_beta_ranks_externalized_bssu(
                    fooof_version=fooof_version,
                    level_first_or_all_directional="level_first",
                )
            )

        method_data = exclude_subjects(df=method_data, exclude_list=exclude)
        print("Strelow method: level first, 3 directional ranks of level rank 1")

    # select the session of the method data
    percept_methods = [
        f"JLB_directional{external_extension}",
        f"euclidean_directional{external_extension}",
        f"best_bssu_contacts{external_extension}",
        f"detec_strelow_contacts{external_extension}",
    ]

    if method in percept_methods:
        method_session = percept_session
        ses_add_filename = f"_{percept_session}"
    else:
        method_session = "postop"
        ses_add_filename = ""

    method_data = method_data.loc[method_data.session == method_session]

    # check which subject_hemispheres are in both tables
    stn_unique_clinical_contacts = list(
        best_clinical_contacts_ses.subject_hemisphere.unique()
    )
    stn_unique_method = list(method_data.subject_hemisphere.unique())

    stn_comparison_list = list(
        set(stn_unique_clinical_contacts) & set(stn_unique_method)
    )
    stn_comparison_list.sort()

    comparison_df_clinical_contacts = best_clinical_contacts_ses.loc[
        best_clinical_contacts_ses["subject_hemisphere"].isin(stn_comparison_list)
    ]
    comparison_df_method = method_data.loc[
        method_data["subject_hemisphere"].isin(stn_comparison_list)
    ]

    comparison_df = pd.concat(
        [comparison_df_clinical_contacts, comparison_df_method], axis=0
    )

    # for each subject_hemisphere compare the clinically active contacts to the selected contacts with the methods
    for sub_hem in stn_comparison_list:
        # only run, if sub_hem STN exists in both session Dataframes
        if sub_hem not in comparison_df.subject_hemisphere.values:
            print(f"{sub_hem} is not in the comparison Dataframe.")
            continue

        # only take one electrode at both sessions and get spearman correlation
        stn_comparison = comparison_df.loc[
            comparison_df["subject_hemisphere"] == sub_hem
        ]

        stn_clinical_contacts = stn_comparison.loc[
            stn_comparison.method == "best_clinical_contacts"
        ]
        stn_method = stn_comparison.loc[stn_comparison.method == method]

        # clinical contacts
        clinical_contact_selection = stn_clinical_contacts.CathodalContact.values[0]
        clinical_contact_selection = str(clinical_contact_selection).split(
            "_"
        )  # list of clinical contacts
        number_clinical_contacts = len(clinical_contact_selection)

        # from method: get contact selection in 2 versions: beta_rank_1_and_2 OR beta relative_to_max_above_70
        if method == f"best_bssu_contacts{external_extension}":
            method_rank_1_and_2 = stn_method["selected_2_contacts"].values[0]

            method_rel_to_max_above_70 = "n.a."
            number_contacts_rel_above_70 = "n.a."
            compare_rel_above_70_contacts = "n.a."
            common_rel_above_70_at_least_2 = "n.a."
            method_rank_1 = "n.a."

        elif method == f"detec_strelow_contacts{external_extension}":
            method_rank_1_and_2 = stn_method[stn_method["beta_rank"].isin([1.0, 2.0])]
            if len(method_rank_1_and_2.contact.values) == 0:
                print(
                    f"Sub-{sub_hem} has no rank 1 or rank 2 contacts with method {method}."
                )
                continue

            method_rank_1_and_2 = method_rank_1_and_2[
                "contact"
            ].tolist()  # list of contacts ranked 1 or 2

            method_rel_to_max_above_70 = "n.a."
            number_contacts_rel_above_70 = "n.a."
            compare_rel_above_70_contacts = "n.a."
            common_rel_above_70_at_least_2 = "n.a."

            ######################## rank 1 contact
            method_rank_1 = stn_method[stn_method["beta_rank"] == 1.0]
            method_rank_1 = method_rank_1.contact.values[0]

        else:
            method_rank_1_and_2 = stn_method[stn_method["beta_rank"].isin([1.0, 2.0])]
            if len(method_rank_1_and_2.contact.values) == 0:
                print(
                    f"Sub-{sub_hem} has no rank 1 or rank 2 contacts with method {method}."
                )
                continue

            ######################## rank 1 contact
            method_rank_1 = stn_method[stn_method["beta_rank"] == 1.0]
            method_rank_1 = method_rank_1.contact.values[0]
            
            method_rank_1_and_2 = method_rank_1_and_2[
                "contact"
            ].tolist()  # list of contacts ranked 1 or 2

            method_rel_to_max_above_70 = stn_method[
                stn_method["beta_relative_to_max"] >= 0.7
            ]
            method_rel_to_max_above_70 = method_rel_to_max_above_70[
                "contact"
            ].tolist()  # list of contacts with rel. beta above 0.7
            number_contacts_rel_above_70 = len(method_rel_to_max_above_70)

            # check if at least one contact matches the clinical contacts: relative above 0.7 method
            if set(clinical_contact_selection).intersection(
                set(method_rel_to_max_above_70)
            ):
                compare_rel_above_70_contacts = "at_least_one_contact_match"

            else:
                compare_rel_above_70_contacts = "no_contacts_match"

            # check if at least 2 contacts match the clinical contacts: relative above 0.7 method
            common_elements_rel_above_70 = set(clinical_contact_selection).intersection(
                set(method_rel_to_max_above_70)
            )
            if len(common_elements_rel_above_70) >= 2:
                common_rel_above_70_at_least_2 = "yes"
            else:
                common_rel_above_70_at_least_2 = "no"

        # check if at least one contact matches the clinical contacts: rank method
        if set(clinical_contact_selection).intersection(set(method_rank_1_and_2)):
            compare_rank_1_and_2_contacts = "at_least_one_contact_match"

        else:
            compare_rank_1_and_2_contacts = "no_contacts_match"

        
        # check if at least 2 contacts match the clinical contacts: rank method
        common_elements_rank_1_and_2 = set(clinical_contact_selection).intersection(
            set(method_rank_1_and_2)
        )
        if len(common_elements_rank_1_and_2) >= 2:
            common_rank_1_and_2_at_least_2 = "yes"
        else:
            common_rank_1_and_2_at_least_2 = "no"
        
        ######################## rank 1 only ########################
        # check if only rank 1 contact matches the clinical contacts
        if method_rank_1 in clinical_contact_selection:
            beta_rank_1_contact_active = "yes"
        else:
            beta_rank_1_contact_active = "no"

        # store values in a dictionary
        results_dict = {
            "method_1": ["best_clinical_contacts"],
            "method_2": [method],
            "session": [method_session],
            "session_clinical": [clinical_session],
            "subject_hemisphere": [sub_hem],
            "clinical_contact_selection": [clinical_contact_selection],
            "number_clinical_contacts": [number_clinical_contacts],
            "contacts_rank_1_2": [method_rank_1_and_2],
            "contacts_rel_to_max_above_70": [method_rel_to_max_above_70],
            "number_contacts_rel_above_70": [number_contacts_rel_above_70],
            "compare_rank_contacts": [compare_rank_1_and_2_contacts],
            "compare_rel_above_70_contacts": [compare_rel_above_70_contacts],
            "both_contacts_matching_rank": [common_rank_1_and_2_at_least_2],
            "both_contacts_matching_rel_above_70": [common_rel_above_70_at_least_2],
            "beta_rank_1_contact_active": [beta_rank_1_contact_active]
        }
        results_single_DF = pd.DataFrame(results_dict)
        results_DF = pd.concat([results_DF, results_single_DF], ignore_index=True)

    ################# sample size result #################
    sub_hem_count = results_DF["subject_hemisphere"].count()

    # count how often at least 1 contact is matching
    at_least_1_same = results_DF.loc[
        results_DF[f"compare_{rank_or_rel_above_70}_contacts"]
        == "at_least_one_contact_match"
    ]
    at_least_1_same = at_least_1_same["subject_hemisphere"].count()
    percentage_at_least_1_same = at_least_1_same / sub_hem_count

    # count how often at least 2 contacts are matching
    both_contacts_matching = results_DF.loc[
        results_DF[f"both_contacts_matching_{rank_or_rel_above_70}"] == "yes"
    ]
    both_contacts_matching = both_contacts_matching["subject_hemisphere"].count()
    percentage_both_contacts_matching = both_contacts_matching / sub_hem_count

    # count how often beta rank 1 contact was active
    rank_1_contact_active = results_DF.loc[
        results_DF["beta_rank_1_contact_active"] == "yes"
    ]
    rank_1_contact_active = rank_1_contact_active["subject_hemisphere"].count()
    percentage_rank_1_contact_active = rank_1_contact_active / sub_hem_count

    sample_size_dict = {
        "method_1": ["best_clinical_contacts"],
        "method_2": [method],
        "session": [method_session],
        "session_clinical": [clinical_session],
        "rank_or_rel": [rank_or_rel_above_70],
        "sample_size": [sub_hem_count],
        "both_contacts_matching": [both_contacts_matching],
        "percentage_both_contacts_matching": [percentage_both_contacts_matching],
        "at_least_1_contact_same": [at_least_1_same],
        "percentage_at_least_one_same_contact_rank_1_and_2": [percentage_at_least_1_same],
        "same_rank_1": [rank_1_contact_active],
        "percentage_same_rank_1": [percentage_rank_1_contact_active]
    }

    sample_size_result = pd.DataFrame(sample_size_dict)

    # save tables as Excel
    io_monopolar_comparison.save_result_excel(
        result_df=results_DF,
        filename=f"fooof_monopol{external_extension}_beta_correlations_per_stn_{clinical_session}_best_clinical_contacts_{method}{ext_fooof_re_ref}{reference_name}_{fooof_version}{ses_add_filename}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    # helpers.save_result_excel(
    #     result_df=sample_size_result,
    #     filename=f"fooof_monopol_beta_correlations_sample_size_df_{clinical_session}_best_clinical_contacts_{method}{reference_name}_{fooof_version}{ses_add_filename}",
    #     sheet_name="fooof_monopol_beta_correlations",
    # )

    return {"single_stn_results": results_DF, "rank": sample_size_result}


def group_rank_comparison_externalized_percept_clinical(
    clinical_session: str,
    percept_session: str,
    fooof_version: str,
    bssu_version: str,
    new_reference: str,
):  # sourcery skip: use-itertools-product
    """ """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    list_of_methods = [
        "externalized_ssd",
        "externalized_fooof",
        f"JLB_directional{external_extension}",
        f"euclidean_directional{external_extension}",
        f"best_bssu_contacts{external_extension}",
        f"detec_strelow_contacts{external_extension}",
    ]

    rank_comparison_group = {}

    percept_methods = [
        f"euclidean_directional{external_extension}",
        f"JLB_directional{external_extension}",
        f"detec_strelow_contacts{external_extension}",
        f"best_bssu_contacts{external_extension}",
    ]
    externalized_methods = ["externalized_fooof", "externalized_ssd"]

    # all percept methods vs. each of the percept methods
    for percept_m in percept_methods:
        for vs_percept_m in percept_methods:
            percept_vs_percept = rank_comparison_percept_methods(
                method_1=percept_m,
                method_2=vs_percept_m,
                fooof_version=fooof_version,
                bssu_version=bssu_version,
            )

            percept_vs_percept = percept_vs_percept["rank"]
            percept_vs_percept = percept_vs_percept.loc[
                percept_vs_percept.session == percept_session
            ]

            rank_comparison_group[f"{percept_m}_{vs_percept_m}"] = percept_vs_percept

    # percept methods vs. externalized FOOOF
    for percept_m in percept_methods:
        for ext_m in externalized_methods:
            percept_vs_externalized_fooof = percept_vs_externalized(
                method=percept_m,
                percept_session=percept_session,
                strelow_level_first= "all_directional", #"level_first",
                externalized_version=ext_m,
                fooof_version="v2",
                new_reference=new_reference,
                bssu_version=bssu_version,
                reference="bipolar_to_lowermost",
            )

            rank_comparison_group[
                f"{percept_m}_{ext_m}"
            ] = percept_vs_externalized_fooof["rank"]

    # compare externalized with each other
    for ext_1 in externalized_methods:
        for ext_2 in externalized_methods:
            ext_1_vs_ext_2 = externalized_versions_comparison(
                externalized_version_1=ext_1,
                externalized_version_2=ext_2,
                new_reference=new_reference,
                fooof_version="v2",
                reference="bipolar_to_lowermost",
            )

            rank_comparison_group[f"{ext_1}_{ext_2}"] = ext_1_vs_ext_2["rank"]

    # compare all methods to clinical contacts
    for m in list_of_methods:
        result_m_vs_clinical = methods_vs_best_clinical_contacts(
            clinical_session=clinical_session,
            percept_session=percept_session,
            method=m,
            rank_or_rel_above_70="rank",
            fooof_version="v2",
            new_reference=new_reference,
            bssu_version=bssu_version,
            reference="bipolar_to_lowermost",
        )

        rank_comparison_group[f"best_clinical_contacts_{m}"] = result_m_vs_clinical[
            "rank"
        ]

    rank_comparison_group = pd.concat(rank_comparison_group.values(), ignore_index=True)
    rank_comparison_group_copy = rank_comparison_group.copy()
    rank_comparison_group_copy["method_comparison"] = (
        rank_comparison_group_copy["method_1"]
        + "_"
        + rank_comparison_group_copy["method_2"]
    )

    # save session Dataframes as Excel files
    io_monopolar_comparison.save_result_as_pickle(
        filename=f"rank_group_comparison_all_clinical_{clinical_session}_percept_{percept_session}{external_extension}_{ext_fooof_re_ref}{fooof_version}",
        data=rank_comparison_group_copy,
    )

    io_monopolar_comparison.save_result_excel(
        filename=f"rank_group_comparison_all_clinical_{clinical_session}_percept_{percept_session}{external_extension}_{ext_fooof_re_ref}{fooof_version}",
        result_df=rank_comparison_group_copy,
        sheet_name="rank",
    )

    return rank_comparison_group_copy


def group_correlation_comparison_externalized_percept_clinical(
    percept_session: str,
    fooof_version: str,
    new_reference: str,
    bssu_version: str,
):  # sourcery skip: use-itertools-product
    """
    Watch out: detec strelow method -> here there is no 2 step procedure of level, then direction used! Instead all directional weighted beta values
    """

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    list_of_methods = [
        "externalized_ssd",
        "externalized_fooof",
        f"JLB_directional{external_extension}",
        f"euclidean_directional{external_extension}",
        f"detec_strelow_contacts{external_extension}",
    ]

    correlation_comparison_group = {}

    percept_methods = [
        f"euclidean_directional{external_extension}",
        f"JLB_directional{external_extension}",
        f"detec_strelow_contacts{external_extension}",
    ]
    externalized_methods = ["externalized_fooof", "externalized_ssd"]

    # all percept methods vs. each of the percept methods
    for percept_m in percept_methods:
        for vs_percept_m in percept_methods:
            percept_vs_percept = correlation_monopol_fooof_beta_methods(
                method_1=percept_m,
                method_2=vs_percept_m,
                fooof_version=fooof_version,
                bssu_version=bssu_version,
            )
            percept_vs_percept = percept_vs_percept["correlation"]
            percept_vs_percept = percept_vs_percept.loc[
                percept_vs_percept.session == percept_session
            ]

            correlation_comparison_group[
                f"{percept_m}_{vs_percept_m}"
            ] = percept_vs_percept

    # percept methods vs. externalized FOOOF
    for percept_m in percept_methods:
        for ext_m in externalized_methods:
            percept_vs_externalized_fooof = percept_vs_externalized(
                method=percept_m,
                percept_session=percept_session,
                strelow_level_first="all_directional",
                externalized_version=ext_m,
                fooof_version="v2",
                bssu_version=bssu_version,
                new_reference=new_reference,
                reference="bipolar_to_lowermost",
            )

            correlation_comparison_group[
                f"{percept_m}_{ext_m}"
            ] = percept_vs_externalized_fooof["correlation"]

    # compare externalized with each other
    for ext_1 in externalized_methods:
        for ext_2 in externalized_methods:
            ext_1_vs_ext_2 = externalized_versions_comparison(
                externalized_version_1=ext_1,
                externalized_version_2=ext_2,
                new_reference=new_reference,
                fooof_version="v2",
                reference="bipolar_to_lowermost",
            )

            correlation_comparison_group[f"{ext_1}_{ext_2}"] = ext_1_vs_ext_2[
                "correlation"
            ]

    correlation_comparison_group = pd.concat(
        correlation_comparison_group.values(), ignore_index=True
    )
    correlation_comparison_group = correlation_comparison_group.copy()
    correlation_comparison_group["method_comparison"] = (
        correlation_comparison_group["method_1"]
        + "_"
        + correlation_comparison_group["method_2"]
    )

    # save session Dataframes as Excel files
    io_monopolar_comparison.save_result_as_pickle(
        filename=f"correlation_group_comparison_all_externalized_percept_{percept_session}{external_extension}_{ext_fooof_re_ref}{fooof_version}",
        data=correlation_comparison_group,
    )

    io_monopolar_comparison.save_result_excel(
        filename=f"correlation_group_comparison_all_externalized_percept_{percept_session}{external_extension}_{ext_fooof_re_ref}{fooof_version}",
        result_df=correlation_comparison_group,
        sheet_name="correlation",
    )

    return correlation_comparison_group

