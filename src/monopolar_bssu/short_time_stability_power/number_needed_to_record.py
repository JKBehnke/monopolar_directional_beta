""" Calculating the proportion of patients showing a 'stable' channel rank (of beta rank 1 channels only)"""

import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from fooof.plts.spectra import plot_spectrum
from scipy.stats import shapiro
from itertools import combinations
from scipy.stats import wilcoxon, mannwhitneyu
from statannotations.Annotator import Annotator

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import io_monopolar_comparison as io_monopolar_comparison
from ..utils import loadResults as loadResults
from ..utils import sub_recordings_dict as sub_recordings_dict
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..externalized_lfp import feats_ssd as feats_ssd
from ..short_time_stability_power import externalized_short_chunks as short_windows

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

# patient_metadata = load_data.load_patient_metadata_externalized()
PATIENT_METADATA = io_externalized.load_excel_data(filename="patient_metadata")
HEMISPHERES = ["Right", "Left"]
DIRECTIONAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]
BSSU_CHANNELS = [
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

BSSU_RING_CHANNELS = [
    "01",
    "12",
    "23",
]

BSSU_SEGM_INTER_CHANNELS = [
    "1A2A",
    "1B2B",
    "1C2C",
]

BSSU_SEGM_INTRA_CHANNELS = [
    "1A1B",
    "1A1C",
    "1B1C",
    "2A2B",
    "2A2C",
    "2B2C",
]

BSSU_SEGM_CHANNELS = [
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

SECONDS_LIST = [35, 30, 25, 20, 15, 10, 5]

SPECTRA_LIST = [
    "power_spectrum_2_min",
    "power_spectrum_20_sec_1",
    "power_spectrum_20_sec_2",
    "power_spectrum_20_sec_3",
    "power_spectrum_20_sec_4",
    "power_spectrum_20_sec_5",
]

FREQ_BAND_LIST = ["beta", "high_beta", "low_beta"]


# load the data
def load_ranks(
    incl_sub: list,
    sec_per_epoch: int,
    freq_band: str,
    channel_group: str,
    rank_of_interest: int,
):
    """
    Input:
        - freq_band: str "beta", "high_beta", "low_beta"
        - channel_group: str "all", "ring", "segm_inter", "segm_intra", "segm"
        - incl_sub: ["all"] or list of sub
        - sec_per_epoch: e.g. 20


    """

    data = short_windows.get_ranks_short_sec_power_channels(
        incl_sub=incl_sub,
        filtered="notch_and_band_pass_filtered",
        sec_per_epoch=sec_per_epoch,
        freq_band=freq_band,
        channel_group=channel_group,
        rank_of_interest=rank_of_interest,
    )

    data = data["only_one_2min_rank_data"]

    short_window_ranks = (
        {}
    )  # keys: sub_hem_chan, values: list with beta ranks for multiple windows

    # merge together columns "subject", "hemisphere", "channel" to get unique patient ids
    data["patient_id"] = (
        data["subject"] + "_" + data["hemisphere"] + "_" + data["channel"]
    )
    patient_ids = data["patient_id"].unique()

    for patient_id in patient_ids:
        patient_data = data[data["patient_id"] == patient_id]

        # get number of epochs
        number_epochs = patient_data["number_of_epochs"].values[0]

        measurements = (
            []
        )  # fill this list with the number of epochs x short sec power averages
        for i in range(1, number_epochs + 1):
            column_name = (
                f"rank_sec_{i}"  # power average within the freq band of interest
            )

            measurements.append(patient_data[column_name].values[0])

        short_window_ranks[patient_id] = measurements

    # create a Dataframe from the dictionary with rows=sub_hem_chan and columns=windows
    return pd.DataFrame.from_dict(short_window_ranks, orient="index")


def most_frequent_integer(lst):
    """
    From a list with integers, return the most frequent integer(s) as a list.
    If more than one integer has the same frequency, all integers are returned.
    """
    count_dict = {}
    for num in lst:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1
    max_count = max(count_dict.values())
    return [
        num for num, count in count_dict.items() if count == max_count
    ]  # returns a list of most frequent integers


def stability_distribution_plot(
    incl_sub: list,
    sec_per_epoch: int,
    freq_band: str,
    channel_group: str,
    rank_of_interest: int,
    num_windows: int,
    num_iterations: int,
):
    """
    This function iterates (num_iterations times) and picks a certain number of windows (num_windows) randomly per channel.
    From the selected windows it checks if the most common rank is rank 1 !(NOT the rank_of_interest).

    """

    np.random.seed(42)

    data = load_ranks(
        incl_sub=incl_sub,
        sec_per_epoch=sec_per_epoch,
        freq_band=freq_band,
        channel_group=channel_group,
        rank_of_interest=rank_of_interest,
    )

    stable_proportions = []  # list of proportions (n=1000)

    for iteration in range(num_iterations):
        stable_counts = 0

        for sub_hem_chan in data.itertuples(
            index=False
        ):  # .itertuples iteratese over DataFrame rows as named tuples
            # with index=False only the columns are accessed

            # will randomly select 3 elements from sub_hem_chan
            # Nan values excluded
            exclude_nans = np.array(sub_hem_chan)[~np.isnan(sub_hem_chan)]
            selected_windows = np.random.choice(
                exclude_nans, size=num_windows, replace=False
            )
            # replace=False means elements cannot be selected more than once

            # stability is True if the most common rank is the rank_of_interest
            most_common_rank = most_frequent_integer(
                list(selected_windows)
            )  # most common rank(s) in the selected windows
            if len(most_common_rank) > 1:
                stability = False
            else:
                stability = most_common_rank[0] == 1  # rank_of_interest

            # stability is True only if 80% of windows show beta rank==rank_of_interest
            # proportion_stable = np.sum(selected_windows == rank_of_interest) / num_windows
            # stability = proportion_stable >= stability_threshold # True if proportion_stable is bigger or equal the stability_threshold
            stable_counts += stability  # in Python True equals 1, False equals 0, so if stability is True stable_counts is increamented by 1

        # across patients
        proportion = stable_counts / len(
            data
        )  # proportion of patients with stable counts per iteration
        stable_proportions.append(proportion)  # list of proportions (n=1000)

    fig = plt.figure(figsize=(8, 6))
    plt.hist(stable_proportions, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Proportion of Patients per Iteration")
    plt.ylabel("Frequency")
    plt.xlim(0.3, 1.0)
    plt.title(
        f"Distribution of Patient Proportion with Stable Rank {rank_of_interest}"
        + f"\nfor {num_windows} Windows of {sec_per_epoch}s"
    )

    fig.tight_layout()

    io_monopolar_comparison.save_fig_png_and_svg(
        filename=f"distribution_stable_rank_{rank_of_interest}_"
        + f"number_{sec_per_epoch}s_windows_{num_windows}_"
        + f"{channel_group}_group_{freq_band}",
        figure=fig,
    )

    # give a summary of the distribution: count of all values
    value_counts = Counter(stable_proportions)

    for value, count in value_counts.items():
        print(f"Value {value} appears {count} times")

    return stable_proportions, value_counts


def normal_distrbution_test(data):
    """
    Input:
        - data = list of proportions (n=1000)

    Outcome:
        - p value: if p-val < .05 this means the data is not normally distributed
    """
    stat, p = shapiro(data)

    # interpret
    alpha = 0.05
    if p > alpha:
        distribution = "normal_distribution"
    else:
        distribution = "not_normal_distribution"

    return distribution, p


def calculate_p_val(stable_proportions: list, goal: float):
    """

    Because our data is not normally distributed, we use the percentile method to calculate the confidence interval.

    Input:
        - stable_proportions = list with 1000 proportions of patients with stable rank
        (from stability_distribution_plot() )

        - goal = float e.g. 0.8
        if you want to test whether 80% of patients showing stable rank is within the confidence interval

    Outcome:

        - p value:  if p-val < .05 this means 80 % is within the confidence interval, so we can conclude 80 % of patients show "stable" ranks
                    if p-val significant, then 80 % lies outside of the confidence interval
    """

    mean_proportion = np.mean(stable_proportions)
    ci_lower = np.percentile(stable_proportions, 2.5)  # 2.5th percentile
    ci_upper = np.percentile(
        stable_proportions, 97.5
    )  # 97.5th percentile used to construct a 95% CI
    p_value = np.sum(np.array(stable_proportions) >= goal) / len(
        stable_proportions
    )  # fraction of iterations meeting or exceeding the goal
    # low p-value <.05 means Few iterations meet or exceed the goal, indicating that achieving the goal is rare and significantly difficult.

    return mean_proportion, ci_lower, ci_upper, p_value


def main_rank_stability_test(
    incl_sub: list,
    sec_per_epoch: int,
    freq_band: str,
    channel_group: str,
    rank_of_interest: int,
    num_windows_lowest: int,
    num_windows_highest: int,
    num_iterations: int,
    goal_stability: float,
):
    """
    Input:
        - incl_sub: ["all"]
        - rank_of_interest: 1
        - num_windows_lowest: 2 (e.g. if you want to test the stability of 2 repetitions of 20sec windows)
        - num_windows_highest: 8
        - num_iterations: 1000
        - goal_stability: 0.8, testing if 80 % of patients would be within the confidence interval of the distribution
        - freq_band: str "beta", "high_beta", "low_beta"
        - channel_group: str "all", "ring", "segm_inter", "segm_intra", "segm"
        - incl_sub: ["all"] or list of sub
        - sec_per_epoch: e.g. 20

    """

    results = []
    results_report = pd.DataFrame()

    value_counts_results = {}

    for num in range(num_windows_lowest, num_windows_highest + 1):

        # calculate and plot the distribution
        stable_proportions, value_counts = stability_distribution_plot(
            incl_sub=incl_sub,
            sec_per_epoch=sec_per_epoch,
            freq_band=freq_band,
            channel_group=channel_group,
            rank_of_interest=rank_of_interest,
            num_windows=num,
            num_iterations=num_iterations,
        )

        value_counts_results[num] = value_counts

        mean_proportion, ci_lower, ci_upper, p_value = calculate_p_val(
            stable_proportions=stable_proportions, goal=goal_stability
        )

        distribution, p = normal_distrbution_test(stable_proportions)

        results.append((num, mean_proportion, ci_lower, ci_upper, p_value))
        result_summary = {
            "number_windows": [num],
            "sec_per_epoch": [sec_per_epoch],
            "mean_proportion": [mean_proportion],
            "ci_lower": [ci_lower],
            "ci_upper": [ci_upper],
            "p_value": [p_value],
            "channel_group": [channel_group],
            "freq_band": [freq_band],
            "distribution": [distribution],
        }

        results_dataframe = pd.DataFrame(result_summary)
        results_report = pd.concat(
            [results_report, results_dataframe], ignore_index=True
        )

    # Print the results
    io_monopolar_comparison.save_result_excel(
        result_df=results_report,
        filename=f"across_num_windows_ci_patient_proportions_stable_rank_{rank_of_interest}_"
        + f"{sec_per_epoch}s_windows_"
        + f"{channel_group}_group_{freq_band}",
        sheet_name="result",
    )

    # Plot the results
    window_sizes, mean_props, ci_lowers, ci_uppers, p_values = zip(*results)

    yerr = np.array(
        [mean_props - np.array(ci_lowers), np.array(ci_uppers) - mean_props]
    )

    results_report["mean-ci_lower"] = yerr[0]
    results_report["ci_upper-mean"] = yerr[1]

    if np.any(yerr < 0):
        print("yerr contains negative values:", yerr)
        yerr = np.abs(yerr)

    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(window_sizes, mean_props, yerr=yerr, fmt="o", capsize=5)
    plt.axhline(y=goal_stability, color="r", linestyle="-")
    plt.xlabel("Number of Windows")
    plt.ylabel("Mean and CI of Patient Proportion with Stable Rank")
    plt.ylim(0.3, 1.0)
    plt.title("Patient Proportions with Stable Ranks Across Number of Windows")

    fig.tight_layout()

    io_monopolar_comparison.save_fig_png_and_svg(
        filename=f"across_num_windows_ci_patient_proportions_stable_rank_{rank_of_interest}_"
        + f"{sec_per_epoch}s_windows_"
        + f"{channel_group}_group_{freq_band}",
        figure=fig,
    )

    return results_report, value_counts_results
