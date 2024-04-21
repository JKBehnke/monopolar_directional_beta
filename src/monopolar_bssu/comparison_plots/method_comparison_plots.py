""" Plots of comparisons between monopolar estimation methods """


import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, mannwhitneyu
from itertools import combinations
from statannotations.Annotator import Annotator

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import io_monopolar_comparison as io_monopolar_comparison
from ..utils import loadResults as loadResults
from ..utils import sub_recordings_dict as sub_recordings
from ..comparison_analysis import monopol_comparison_tests as monopol_comparison_tests
from ..comparison_analysis import monopol_method_comparison as monopol_method_comparison

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

BSSU_METHODS = [
    "euclidean_directional",
    "JLB_directional",
    "detec_strelow_contacts"
]

EXTERNALIZED_METHODS = [
    "externalized_fooof",
    "externalized_ssd"
] 


# Heatmap of all comparisons
def heatmap_method_comparison(
    value_to_plot: str,
    clinical_session: str,
    percept_session: str,
    rank_or_correlation: str,
    fooof_version: str,
    new_reference: str,
    bssu_version: str,
):
    """
    methods: "externalized_ssd", "externalized_fooof", "JLB_directional", "euclidean_directional", "best_bssu_contacts", "detec_strelow_contacts", "best_clinical_contacts"

    Input:
        - value_to_plot: str e.g.
            "percentage_at_least_one_same_contact_rank_1_and_2" must be a column in the sample size result Excel file
            "percentage_both_contacts_matching"
            "estimated_beta_spearman",
            "normalized_beta_pearson",
            "cluster_beta_spearman"

        - clinical_session (only relevant when bssu_version == "percept")

        - percept_session

        - rank_or_correlation: "rank" or "correlation"

        - bssu_version: "percept" or "externalized"

    """
    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    if new_reference == "one_to_zero_two_to_three":
        ext_fooof_re_ref = "one_to_zero_two_to_three_"

    elif new_reference == "no":
        ext_fooof_re_ref = ""

    # load the comparison matrix for the value to plot
    loaded_comparison_matrix = (
        monopol_comparison_tests.get_comparison_matrix_for_heatmap_from_DF(
            value_to_plot=value_to_plot,
            clinical_session=clinical_session,
            percept_session=percept_session,
            rank_or_correlation=rank_or_correlation,
            fooof_version=fooof_version,
            new_reference=new_reference,
            bssu_version=bssu_version,
        )
    )

    comparison_matrix = loaded_comparison_matrix["comparison_matrix"]
    comparison_dict = loaded_comparison_matrix["comparison_dict"]
    sample_size = loaded_comparison_matrix["sample_size"]
    sample_size_matrix = loaded_comparison_matrix["sample_size_matrix"]
    list_of_methods = loaded_comparison_matrix["list_of_methods"]

    # Create a heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(comparison_matrix, cmap="coolwarm", interpolation="nearest")

    # # Define custom color map
    # colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]  # Adjust color points as needed
    # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    # # Example data
    # comparison_matrix = np.random.rand(10, 10)  # Replace with your data

    cbar = fig.colorbar(heatmap)
    # heatmap.set_clim(vmin=0, vmax=1)
    # cbar.set_label(f"{value_to_plot}")

    ax.set_xticks(range(len(list_of_methods)))
    ax.set_yticks(range(len(list_of_methods)))
    ax.set_xticklabels(list_of_methods, rotation=90)
    ax.set_yticklabels(list_of_methods)
    ax.grid(False)

    title_str = {
        "percentage_at_least_one_same_contact_rank_1_and_2": "Selecting 2 contacts from 6 directional contacts:"
        + "\nhow many hemispheres with at least 1 matching contact [%]?",
        "percentage_both_contacts_matching": "Selecting 2 contacts from 6 directional contacts:"
        + "\nhow many hemispheres with with both contacts matching [%]?",
        "estimated_beta_spearman": "Spearman correlation of 6 directional values per hemisphere"
        + "\nhow many hemispheres with significant correlation [%]?",
        "normalized_beta_pearson": "Pearson correlation of 6 directional normalized values per hemisphere"
        + "\nhow many hemispheres with significant correlation [%]?",
    }

    ax.set_title(title_str[value_to_plot])

    # Add the values to the heatmap cells
    # for i in range(len(list_of_methods)):
    #     for j in range(len(list_of_methods)):
    #         ax.text(j, i, f"{comparison_matrix[i][j]:.2f}", ha='center', va='center', color='black', fontsize=10)

    for i in range(len(list_of_methods)):
        for j in range(len(list_of_methods)):
            value = f"{comparison_matrix[i][j]:.2f}"
            sample_size_value = f"n={int(sample_size_matrix[i][j])}"
            text_for_cell = f"{value}\n{sample_size_value}"
            ax.text(
                j,
                i,
                text_for_cell,
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    io_monopolar_comparison.save_fig_png_and_svg(
        filename=f"heatmap_method_comparison_{value_to_plot}_clinical_{clinical_session}_percept_{percept_session}{external_extension}_{rank_or_correlation}_{ext_fooof_re_ref}{fooof_version}",
        figure=fig,
    )

    return comparison_matrix, comparison_dict, sample_size


def barplot_group_correlation(
    percept_session: str,
    fooof_version: str,
    bssu_version: str,
    new_reference: str,
    method_to_compare: str,
    list_of_methods: list
):
    """
    This function plots a barplot of correlation or rank comparison results of 
    - a list of methods
    - to one method

    Input:
        - percept_session: "postop", only relevant, if bssu_version == "percept"
        - fooof_version: "v2"
        - bssu_version: "externalized" or "percept"
        - new_reference: "one_to_zero_two_to_three"
        - method_to_compare: "externalized_fooof" 
        - list_of_methods: ["euclidean_directional", "JLB_directional", "detec_strelow_contacts"

    """

    result_dataframe = pd.DataFrame()

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    comparison_dataframe = monopol_method_comparison.group_correlation_comparison_externalized_percept_clinical(
        percept_session=percept_session,
        fooof_version=fooof_version,
        bssu_version=bssu_version,
        new_reference=new_reference
    )

    # select only the data of interest
    if method_to_compare in BSSU_METHODS:
        method_to_compare_data = comparison_dataframe.loc[comparison_dataframe.method_2 == f"{method_to_compare}{external_extension}"]

    elif method_to_compare in EXTERNALIZED_METHODS:
        method_to_compare_data = comparison_dataframe.loc[comparison_dataframe.method_2 == method_to_compare]


    # result_dict = {
    #     "percentage_significant_y": [],
    #     "methods_x": []
    # }

    # select the data of methods in the list
    for m, method in enumerate(list_of_methods):

        if method in BSSU_METHODS:
            m_name = f"{method}{external_extension}"
        
        else:
            m_name = method
        
        m_data = method_to_compare_data.loc[method_to_compare_data.method_1 == m_name]
        spearman_data = m_data.loc[m_data.correlation == "estimated_beta_spearman"]
        percentage_significant = spearman_data.percentage_significant.values[0]

        result_dict = {
            "percentage_significant_y": [percentage_significant],
            "methods_x": [m]
        }
        
        result_df_single = pd.DataFrame(result_dict)
        result_dataframe = pd.concat([result_dataframe, result_df_single], ignore_index=True)
        
    # plot a barplot
    sns.barplot(result_dataframe, x="methods_x", y="percentage_significant_y")

    return result_dataframe

    

def bssu_vs_externalized_correlation(
    percept_session: str,
    bssu_version: str,
    externalized_version: str,
    list_of_methods: list
):
    """
    This function correlates all methods ins list_of_methods to the externalied version and compares the correlation of each method with each other
    - fisher-transformed correlation
    - t-test
    
    This function plots a barplot of correlation or rank comparison results of 
    - a list of methods
    - to one method

    Input:
        - percept_session: "postop", only relevant, if bssu_version == "percept"
        - fooof_version: "v2"
        - bssu_version: "externalized" or "percept"
        - new_reference: "one_to_zero_two_to_three"
        - method_to_compare: "externalized_fooof" 
        - list_of_methods: ["euclidean_directional_externalized_bssu", "JLB_directional_externalized_bssu", "detec_strelow_contacts_externalized_bssu"]

    """

    organized_data = {} # nan free and fisher transformed (np.arctanh)
    spearman_original_data = {}
    sub_hemispheres = {}
    data_description = pd.DataFrame()
    fisher_transformed_data_description = pd.DataFrame()
    mwu_all_results = pd.DataFrame()

    pairs = list(combinations(list_of_methods, 2))
    
    for m, method in enumerate(list_of_methods):

        loaded_data = monopol_method_comparison.percept_vs_externalized(
            method=method,
            percept_session=percept_session,
            externalized_version=externalized_version,
            fooof_version="v2",
            bssu_version=bssu_version,
            reference="bipolar_to_lowermost", 
            new_reference="one_to_zero_two_to_three",
            strelow_level_first="all_directional"
        )

        method_data = loaded_data["single_stn_results"]
        estimated_beta_spearman = method_data["estimated_beta_spearman_r"]
        sub_hem = method_data["subject_hemisphere"]
        
        # check for NaN values
        nan_indices = estimated_beta_spearman.isna()
        nan_free_estimated_beta_spearman = estimated_beta_spearman[~nan_indices]
        nan_free_sub_hem = sub_hem[~nan_indices]

        # fisher_transformation of correlation coefficients
        fisher_transformed = np.arctanh(nan_free_estimated_beta_spearman)

        # in case of correlation coefficients of ±1 the fisher transformation results in an inf value (division by zero)
        # exclude inf from the dataset
        inf_indices = np.isinf(fisher_transformed)
        fisher_transformed = fisher_transformed[~inf_indices]
        nan_free_sub_hem = nan_free_sub_hem[~inf_indices]
        #fisher_transformed = fisher_transformed[np.isfinite(fisher_transformed)]

        # save in dictionary
        organized_data[method] = fisher_transformed
        sub_hemispheres[method] = nan_free_sub_hem
        spearman_original_data[method] = nan_free_estimated_beta_spearman

        # get statistics
        stats_df = io_externalized.get_statistics(data_info=method, data=nan_free_estimated_beta_spearman)
        data_description = pd.concat([data_description, stats_df], ignore_index=True)

        fisher_stats = io_externalized.get_statistics(data_info=method, data=fisher_transformed)
        fisher_transformed_data_description = pd.concat([fisher_transformed_data_description, fisher_stats], ignore_index=True)
    
    for pair in pairs:
        group1_spearman = pair[0]
        group2_spearman = pair[1]

        # perform a mann-whithney-u test (non-parametric, different sample sizes)
        statistic, p_value = mannwhitneyu(organized_data[group1_spearman], organized_data[group2_spearman], alternative="two-sided")

        mwu_result = {
            "method_1": [group1_spearman],
            "method_2": [group2_spearman],
            "statistic": [statistic],
            "p_value": [p_value],
            "significant": [p_value < 0.05]
        }

        mwu_pair = pd.DataFrame(mwu_result)
        mwu_all_results = pd.concat([mwu_all_results, mwu_pair], ignore_index=True)
        
        
    # plot a barplot
    #sns.barplot(result_dataframe, x="methods_x", y="percentage_significant_y")

    return {
        "spearman_data_description": data_description,
        "fisher_transformed_data_description": fisher_transformed_data_description,
        "mwu_result": mwu_all_results,
        "organized_data": organized_data,
        "sub_hem_data": sub_hemispheres,
        "spearman_original_data": spearman_original_data
    }


def bssu_vs_externalized_plot(
        percept_session: str,
        bssu_version: str,
        externalized_version: str,
        list_of_methods: list,
        spearman_or_fisher_transformed: str
):
    """
    Input: 
        - spearman_or_fisher_transformed: "spearman" or "fisher_transformed"
    """

    data_organized_to_plot = pd.DataFrame()

    data_loaded = bssu_vs_externalized_correlation(
        percept_session=percept_session,
        bssu_version=bssu_version,
        externalized_version=externalized_version,
        list_of_methods=list_of_methods
    )

    if spearman_or_fisher_transformed == "spearman":
        data_all_methods = data_loaded["spearman_original_data"] 
        title_str = "Spearman_correlation"

    elif spearman_or_fisher_transformed == "fisher_transformed":
        data_all_methods = data_loaded["organized_data"] 
        title_str = "Spearman_correlation_fisher_transformed"

    # concatenate together in one dataframe
    for m, method in enumerate(list_of_methods):

        m_data = data_all_methods[method].values
        method_x = [m+1]*len(m_data)

        data_to_plot = pd.DataFrame({"data_y": m_data, "method_x": method_x})
        data_organized_to_plot = pd.concat([data_organized_to_plot, data_to_plot], ignore_index=True)
    

    # plot a violinplot
    fig = plt.figure() 
    ax = fig.add_subplot()

    sns.violinplot(
        data=data_organized_to_plot,
        x="method_x",
        y="data_y",
        palette="coolwarm",
        inner="box",
        ax=ax,
    )

    # statistical test:
    pairs = list(combinations(np.arange(1, len(list_of_methods)+1), 2))

    annotator = Annotator(ax, pairs, data=data_organized_to_plot, x='method_x', y="data_y")
    annotator.configure(test='Mann-Whitney', text_format='star')  # or t-test_ind ??
    annotator.apply_and_annotate()

    sns.stripplot(
        data=data_organized_to_plot,
        x="method_x",
        y="data_y",
        ax=ax,
        size=8, # 6
        color="black",
        alpha=0.3,  # Transparency of dots
    )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    plt.title(f"Beta Power {title_str} to {externalized_version} 01, 23 beta power, \n6 directional contacts per lead")
    plt.ylabel(f"{title_str} coefficient")
    plt.xlabel("method to estimate pseudo-monopolar beta power")
    plt.xticks(range(len(list_of_methods)), ["Euclidean", "JLB", "Detec"])

    fig.tight_layout()

    io_externalized.save_fig_png_and_svg(path=GROUP_FIGURES_PATH,
                                         filename=f"{title_str}_beta_power_of_{bssu_version}_to_{externalized_version}_only_directional",
                                         figure=fig)


    
    return data_organized_to_plot


        
################################### Monopolar estimation methods in comparison to each other 


def bssu_vs_bssu_method_correlation(
    bssu_version: str,
    list_of_methods: list,
    percept_session: str,
):
    """
    This function correlates all methods ins list_of_methods to the externalied version and compares the correlation of each method with each other
    - fisher-transformed correlation
    - t-test

    Input:
        - percept_session: "postop", only relevant, if bssu_version == "percept"
        - fooof_version: "v2"
        - bssu_version: "externalized" or "percept"
        - new_reference: "one_to_zero_two_to_three"
        - list_of_methods: ["euclidean_directional_externalized_bssu", "JLB_directional_externalized_bssu", "detec_strelow_contacts_externalized_bssu"]

    """

    organized_data = {} # nan free and fisher transformed (np.arctanh)
    spearman_original_data = {}
    sub_hemispheres = {}
    data_description = pd.DataFrame()
    fisher_transformed_data_description = pd.DataFrame()
    mwu_all_results = pd.DataFrame()

    pairs = list(combinations(list_of_methods, 2))
    
    for method_1 in list_of_methods:
        for method_2 in list_of_methods:

            if method_1 == method_2:
                continue

            loaded_data = monopol_method_comparison.correlation_monopol_fooof_beta_methods(
                method_1=method_1,
                method_2=method_2,
                fooof_version="v2",
                bssu_version=bssu_version,
            )

            method_data = loaded_data["single_stn_results"]

            # if bssu_version is "percept" make sure to only keep the correct session
            if bssu_version == "percept":
                method_data = method_data.loc[method_data["session"] == percept_session]

            estimated_beta_spearman = method_data["estimated_beta_spearman_r"]
            sub_hem = method_data["subject_hemisphere"]
            
            # check for NaN values
            nan_indices = estimated_beta_spearman.isna()
            nan_free_estimated_beta_spearman = estimated_beta_spearman[~nan_indices]
            nan_free_sub_hem = sub_hem[~nan_indices]

            # fisher_transformation of correlation coefficients
            fisher_transformed = np.arctanh(nan_free_estimated_beta_spearman)

            # in case of correlation coefficients of ±1 the fisher transformation results in an inf value (division by zero)
            # exclude inf from the dataset
            inf_indices = np.isinf(fisher_transformed)
            fisher_transformed = fisher_transformed[~inf_indices]
            nan_free_sub_hem = nan_free_sub_hem[~inf_indices]
            #fisher_transformed = fisher_transformed[np.isfinite(fisher_transformed)]

            # save in dictionary
            organized_data[f"{method_1}_{method_2}"] = fisher_transformed
            sub_hemispheres[f"{method_1}_{method_2}"] = nan_free_sub_hem
            spearman_original_data[f"{method_1}_{method_2}"] = nan_free_estimated_beta_spearman

            # get statistics
            stats_df = io_externalized.get_statistics(data_info=f"{method_1}_{method_2}", data=nan_free_estimated_beta_spearman)
            data_description = pd.concat([data_description, stats_df], ignore_index=True)

            fisher_stats = io_externalized.get_statistics(data_info=f"{method_1}_{method_2}", data=fisher_transformed)
            fisher_transformed_data_description = pd.concat([fisher_transformed_data_description, fisher_stats], ignore_index=True)
    
    for pair in pairs:
        group1_spearman = pair[0]
        group2_spearman = pair[1]
        comparison_1 = f"{group1_spearman}_{group2_spearman}"

        if group1_spearman == group2_spearman:
            continue

        for pair2 in pairs:
            group21_spearman = pair2[0]
            group22_spearman = pair2[1]
            comparison_2 = f"{group21_spearman}_{group22_spearman}"

            if group21_spearman == group22_spearman:
                continue

            # perform a mann-whithney-u test (non-parametric, different sample sizes)
            statistic, p_value = mannwhitneyu(organized_data[comparison_1], organized_data[comparison_2], alternative="two-sided")

            mwu_result = {
                "method_1": [comparison_1],
                "method_2": [comparison_2],
                "statistic": [statistic],
                "p_value": [p_value],
                "significant": [p_value < 0.05]
            }

            mwu_pair = pd.DataFrame(mwu_result)
            mwu_all_results = pd.concat([mwu_all_results, mwu_pair], ignore_index=True)
            
        
    # plot a barplot
    #sns.barplot(result_dataframe, x="methods_x", y="percentage_significant_y")

    return {
        "spearman_data_description": data_description,
        "fisher_transformed_data_description": fisher_transformed_data_description,
        "mwu_result": mwu_all_results,
        "organized_data": organized_data,
        "sub_hem_data": sub_hemispheres,
        "spearman_original_data": spearman_original_data
    }


def bssu_vs_bssu_method_correlation_plot(
        percept_session: str,
        bssu_version: str,
        list_of_methods: list,
        spearman_or_fisher_transformed: str
):
    """
    Input: 
        - spearman_or_fisher_transformed: "spearman" or "fisher_transformed"
    """

    data_organized_to_plot = pd.DataFrame()

    data_loaded = bssu_vs_bssu_method_correlation(
        percept_session=percept_session,
        bssu_version=bssu_version,
        list_of_methods=list_of_methods
    )

    pairs = list(combinations(list_of_methods, 2))

    if spearman_or_fisher_transformed == "spearman":
        data_all_methods = data_loaded["spearman_original_data"] 
        title_str = "Spearman_correlation"

    elif spearman_or_fisher_transformed == "fisher_transformed":
        data_all_methods = data_loaded["organized_data"] 
        title_str = "Spearman_correlation_fisher_transformed"

    # list of all method comparisons
    comparisons = []
    for pair in pairs:
        group1_spearman = pair[0]
        group2_spearman = pair[1]

        if group1_spearman == group2_spearman:
            continue

        comparisons.append(f"{group1_spearman}_{group2_spearman}")

    for m, comp in enumerate(comparisons):

        m_data = data_all_methods[comp].values
        method_x = [m+1]*len(m_data)

        data_to_plot = pd.DataFrame({"data_y": m_data, "method_x": method_x})
        data_organized_to_plot = pd.concat([data_organized_to_plot, data_to_plot], ignore_index=True)
    

    # plot a violinplot
    fig = plt.figure(figsize=[10,12]) 
    ax = fig.add_subplot()

    sns.violinplot(
        data=data_organized_to_plot,
        x="method_x",
        y="data_y",
        palette="coolwarm",
        inner="box",
        ax=ax,
    )

    # statistical test:
    pairs = list(combinations(np.arange(1, len(list_of_methods)+1), 2))

    annotator = Annotator(ax, pairs, data=data_organized_to_plot, x='method_x', y="data_y")
    annotator.configure(test='Mann-Whitney', text_format='star')  # or t-test_ind ??
    annotator.apply_and_annotate()

    sns.stripplot(
        data=data_organized_to_plot,
        x="method_x",
        y="data_y",
        ax=ax,
        size=8, # 6
        color="black",
        alpha=0.3,  # Transparency of dots
    )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    plt.title(f"Beta Power {title_str}, {bssu_version} method comparisons \n6 directional contacts per lead")
    plt.ylabel(f"{title_str} coefficient")
    plt.xlabel("methods to estimate pseudo-monopolar beta power")
    plt.xticks(range(len(comparisons)), comparisons, rotation=45)

    fig.tight_layout()

    io_externalized.save_fig_png_and_svg(path=GROUP_FIGURES_PATH,
                                         filename=f"{title_str}_beta_power_of_{bssu_version}_comparisons_only_directional_{percept_session}",
                                         figure=fig)


    
    return data_organized_to_plot


