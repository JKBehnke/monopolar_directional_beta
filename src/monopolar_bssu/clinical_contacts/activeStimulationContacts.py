""" Best clinical stimulation contacts longitudinal change in levels and directions """


import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import scipy
import pingouin as pg
from itertools import combinations
from statannotations.Annotator import Annotator


######### PRIVATE PACKAGES #########
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults

results_path = find_folders.get_local_path(folder="GroupResults")
figures_path = find_folders.get_local_path(folder="GroupFigures")


def correlateActiveClinicalContacts_monopolarPSDRanks(
    incl_sub: list, freqBand: str, rank_or_psd: str, singleContacts_or_average: str
):
    """
    Using the monopolar rank results from the monoRef_weightPsdAverageByCoordinateDistance.py

    Input:
        - incl_sub: list, e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - freqBand: str, e.g. "beta", "lowBeta", "highBeta"
        - rank_or_psd: str, e.g. "rank", "rawPsd"
        - singleContacts_or_average: str, e.g. "singleContacts", "averageContacts"
            -> choose if you want to use the average of active contacts vs. average of inactive contacts to have the same sample size in both groups





    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    hemispheres = ["Right", "Left"]
    sessions = ["fu3m", "fu12m", "fu18m"]

    ##################### LOAD RANKS OR PSD from monoRef_weightPsdAverageByCoordinateDistance.py #####################

    data_weightedByCoordinates = {}
    keys_weightedByCoordinates = {}
    session_weightedByCoordinate_Dataframe = {}

    for sub in incl_sub:
        for hem in hemispheres:
            data_weightedByCoordinates[f"{sub}_{hem}"] = loadResults.load_monoRef_weightedPsdCoordinateDistance_pickle(
                sub=sub, hemisphere=hem, freqBand=freqBand, normalization="rawPsd", filterSignal="band-pass"
            )

            # first check, which sessions exist
            keys_weightedByCoordinates[f"{sub}_{hem}"] = data_weightedByCoordinates[f"{sub}_{hem}"].keys()

            for ses in sessions:
                # first check, if session exists in keys
                if f"{ses}_monopolar_Dataframe" in keys_weightedByCoordinates[f"{sub}_{hem}"]:
                    print(f"{sub}_{hem}_{ses}")

                else:
                    continue

                # get the dataframe per session
                session_weightedByCoordinates = data_weightedByCoordinates[f"{sub}_{hem}"][f"{ses}_monopolar_Dataframe"]

                # choose only directional contacts and Ring contacts 0, 3 and rank again only the chosen contacts
                session_weightedByCoordinates = session_weightedByCoordinates.loc[
                    ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
                ]
                session_weightedByCoordinates["Rank8contacts"] = session_weightedByCoordinates[
                    "averaged_monopolar_PSD_beta"
                ].rank(ascending=False)
                session_weightedByCoordinates_copy = session_weightedByCoordinates.copy()

                # add column subject_hemisphere_monoChannel
                session_weightedByCoordinates_copy["monopolarChannels"] = np.array(
                    ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
                )
                # session_weightedByCoordinates_copy["subject_hemisphere_monoChannel"] = session_weightedByCoordinates_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)
                session_weightedByCoordinates_copy.drop(["rank", "coord_z", "coord_xy"], axis=1, inplace=True)

                # save the Dataframe per sub_hem_ses combination
                session_weightedByCoordinate_Dataframe[f"{sub}_{hem}_{ses}"] = session_weightedByCoordinates_copy

    ##### Concatenate all Dataframes together to one
    sub_hem_ses_keys = list(session_weightedByCoordinate_Dataframe.keys())

    MonoBeta8Ranks_DF = pd.DataFrame()

    # loop through all Dataframes and concatenate together
    for sub_hem_ses in sub_hem_ses_keys:
        single_Dataframe = session_weightedByCoordinate_Dataframe[sub_hem_ses]

        # concatenate all DF together
        MonoBeta8Ranks_DF = pd.concat([MonoBeta8Ranks_DF, single_Dataframe], ignore_index=True)

    ##################### LOAD CLINICAL STIMULATION PARAMETERS #####################

    bestClinicalStim_file = loadResults.load_BestClinicalStimulation_excel()

    # get sheet with best clinical contacts
    BestClinicalContacts = bestClinicalStim_file["BestClinicalContacts"]

    ##################### FILTER THE MONOBETA8RANKS_DF: clinically ACTIVE contacts #####################
    activeMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():
        activeContacts = str(BestClinicalContacts.CathodalContact.values[idx])  # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        activeContacts_list = activeContacts.split("_")  # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = MonoBeta8Ranks_DF.loc[
            (MonoBeta8Ranks_DF["subject_hemisphere"] == sub_hem)
            & (MonoBeta8Ranks_DF["session"] == session)
            & (MonoBeta8Ranks_DF["monopolarChannels"].isin(activeContacts_list))
        ]

        # concatenate single rows to new Dataframe
        activeMonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "active"
    activeMonoBeta8Ranks["clinicalUse"] = "active"

    ##################### FILTER THE MONOBETA8RANKS_DF: clinically INACTIVE contacts #####################
    inactiveMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():
        inactiveContacts = str(BestClinicalContacts.InactiveContacts.values[idx])  # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        inactiveContacts_list = inactiveContacts.split("_")  # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = MonoBeta8Ranks_DF.loc[
            (MonoBeta8Ranks_DF["subject_hemisphere"] == sub_hem)
            & (MonoBeta8Ranks_DF["session"] == session)
            & (MonoBeta8Ranks_DF["monopolarChannels"].isin(inactiveContacts_list))
        ]

        # concatenate single rows to new Dataframe
        inactiveMonoBeta8Ranks = pd.concat([inactiveMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "non_active"
    inactiveMonoBeta8Ranks["clinicalUse"] = "inactive"

    ##################### CONCATENATE BOTH DATAFRAMES: CLINICALLY ACTIVE and INACTIVE CONTACTS #####################
    active_and_inactive_MonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, inactiveMonoBeta8Ranks], ignore_index=True)

    ##################### CHOOSE BETWEEN USING EACH SINGLE CONTACT OR AVERAGE OF ACTIVE OR INACTIVE CONTACTS #####################

    if singleContacts_or_average == "singleContacts":
        data_MonoBeta8Ranks = active_and_inactive_MonoBeta8Ranks

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg(
            '_'.join, axis=1
        )

    # average for each STN: ranks and psd of active or inactive contacts
    elif singleContacts_or_average == "averageContacts":
        STN_average_activeVsInactiveContacts = (
            {}
        )  # store averages of ranks and psd values of active vs. inactive contacts per STN

        STN_unique = list(
            active_and_inactive_MonoBeta8Ranks["subject_hemisphere"].unique()
        )  # list of all existing STNs

        for STN in STN_unique:
            # get dataframe only of one STN
            STN_dataframe = active_and_inactive_MonoBeta8Ranks.loc[
                (active_and_inactive_MonoBeta8Ranks.subject_hemisphere == STN)
            ]

            # get all existing sessions per STN
            sessions_unique = list(STN_dataframe["session"].unique())

            for ses in sessions_unique:
                STN_session_dataframe = STN_dataframe.loc[(STN_dataframe.session == ses)]

                # get average of active contacts
                STN_session_active = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "active")]
                MEANrank_active = STN_session_active["Rank8contacts"].values.mean()
                MEANpsd_active = STN_session_active["averaged_monopolar_PSD_beta"].values.mean()

                # get average of inactive contacts
                STN_session_inactive = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "inactive")]
                MEANrank_inactive = STN_session_inactive["Rank8contacts"].values.mean()
                MEANpsd_inactive = STN_session_inactive["averaged_monopolar_PSD_beta"].values.mean()

                # store MEAN values in dictionary
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_active"] = [
                    STN,
                    ses,
                    MEANrank_active,
                    MEANpsd_active,
                    "active",
                ]
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_inactive"] = [
                    STN,
                    ses,
                    MEANrank_inactive,
                    MEANpsd_inactive,
                    "inactive",
                ]

        # transform the dictionary to Dataframe
        STN_average_activeVsInactive_DF = pd.DataFrame(STN_average_activeVsInactiveContacts)
        STN_average_activeVsInactive_DF.rename(
            index={0: "subject_hemisphere", 1: "session", 2: "MEAN_beta_rank", 3: "MEAN_beta_psd", 4: "clinicalUse"},
            inplace=True,
        )
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.transpose()
        # important to transform datatype of columns MEAN_beta_rank and MEAN_beta_psd to float (otherwise Error when plotting with seaborn)
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEAN_beta_rank": float})
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEAN_beta_psd": float})

        # use the dataframe with MEAN values for plotting
        data_MonoBeta8Ranks = STN_average_activeVsInactive_DF

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg(
            '_'.join, axis=1
        )

    ##################### PERFORM MANN-WHITNEY TEST  #####################

    ses_clinicalUse = [
        "fu3m_active",
        "fu3m_inactive",
        "fu12m_active",
        "fu12m_inactive",
        "fu18m_active",
        "fu18m_inactive",
    ]
    ses_clinicalUse_wilcoxon = [
        ("fu3m_active", "fu3m_inactive"),
        ("fu12m_active", "fu12m_inactive"),
        ("fu18m_active", "fu18m_inactive"),
    ]
    pairs = list(combinations(ses_clinicalUse, 2))
    all_results_mwu = []
    describe_arrays = {}

    # pair = tuple e.g. fu3m_active, fu3m_inactive
    # for pair in pairs:
    for s_c_wcx in ses_clinicalUse_wilcoxon:
        firstInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[0])]
        secondInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[1])]

        if rank_or_psd == "rank":
            if singleContacts_or_average == "singleContacts":
                firstInPair = np.array(firstInPair.Rank8contacts.values)
                secondInPair = np.array(secondInPair.Rank8contacts.values)

            elif singleContacts_or_average == "averageContacts":
                firstInPair = np.array(firstInPair.MEAN_beta_rank.values)
                secondInPair = np.array(secondInPair.MEAN_beta_rank.values)

        elif rank_or_psd == "rawPsd":
            if singleContacts_or_average == "singleContacts":
                firstInPair = np.array(firstInPair.averaged_monopolar_PSD_beta.values)
                secondInPair = np.array(secondInPair.averaged_monopolar_PSD_beta.values)

            elif singleContacts_or_average == "averageContacts":
                firstInPair = np.array(firstInPair.MEAN_beta_psd.values)
                secondInPair = np.array(secondInPair.MEAN_beta_psd.values)

        # Perform Mann-Whitney Test
        results_mwu = pg.wilcoxon(
            firstInPair, secondInPair
        )  # pair is always a tuple, comparing first and second component of this tuple
        results_mwu[f'comparison_{rank_or_psd}_{singleContacts_or_average}'] = '_'.join(
            s_c_wcx
        )  # new column "comparison" with the pair being compared e.g. fu3m_active and fu3m_inactive

        all_results_mwu.append(results_mwu)

    significance_results = pd.concat(all_results_mwu)

    ##################### GET STATISTICAL IMPORTANT FEATURES #####################
    # describe all 6 groups
    for s_c in ses_clinicalUse:
        # get array of each group
        group = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c)]

        if rank_or_psd == "rank":
            if singleContacts_or_average == "singleContacts":
                group = np.array(group.Rank8contacts.values)

            elif singleContacts_or_average == "averageContacts":
                group = np.array(group.MEAN_beta_rank.values)

        elif rank_or_psd == "rawPsd":
            if singleContacts_or_average == "singleContacts":
                group = np.array(group.averaged_monopolar_PSD_beta.values)

            elif singleContacts_or_average == "averageContacts":
                group = np.array(group.MEAN_beta_psd.values)

        description = scipy.stats.describe(group)

        describe_arrays[f"{s_c}_{rank_or_psd}_{singleContacts_or_average}"] = description

    description_results = pd.DataFrame(describe_arrays)
    description_results.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_results = description_results.transpose()

    ##################### STORE RESULTS IN DICTIONARY AND SAVE #####################

    results_dictionary = {"significance_results": significance_results, "description_results": description_results}

    # save as pickle
    results_filepath = os.path.join(
        results_path,
        f"ClinicalActiveVsNonactiveContacts_statistics_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.pickle",
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dictionary, file)

    ##################### PLOT VIOLINPLOT OF RANKS OR RAWPSD OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    if rank_or_psd == "rank":
        if singleContacts_or_average == "singleContacts":
            y_values = "Rank8contacts"
            y_label = "Beta rank of contact"
            title = "Beta rank of clinically active vs. inactive stimulation contacts"
            y_lim = 10, -1

        elif singleContacts_or_average == "averageContacts":
            y_values = "MEAN_beta_rank"
            y_label = "Mean Beta rank of contact group"
            title = "Mean Beta rank of clinically active vs. inactive stimulation contacts"
            y_lim = 10, -1

    elif rank_or_psd == "rawPsd":
        if singleContacts_or_average == "singleContacts":
            y_values = "averaged_monopolar_PSD_beta"
            y_label = "Beta PSD of contact [uV^2/Hz]"
            title = "Beta PSD of clinically active vs. inactive stimulation contacts"
            y_lim = -0.3, 1.8

        elif singleContacts_or_average == "averageContacts":
            y_values = "MEAN_beta_psd"
            y_label = "Mean Beta PSD of contact group [uV^2/Hz]"
            title = "Mean Beta PSD of clinically active vs. inactive stimulation contacts"
            y_lim = -0.3, 1.8

    fig = plt.figure()
    ax = fig.add_subplot()

    # sns.violinplot(data=data_MonoBeta8Ranks, x="session_clinicalUse", y=y_values, hue="clinicalUse", palette="Set2", inner="box", ax=ax)
    sns.violinplot(
        data=data_MonoBeta8Ranks, x="session", y=y_values, hue="clinicalUse", palette="Set3", inner="box", ax=ax
    )

    # statistical test
    # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
    # pairs = list(combinations(ses_clinicalUse, 2))

    # annotator = Annotator(ax, pairs, data=active_and_inactive_MonoBeta8Ranks, x='session_clinicalUse', y=y_values)
    # annotator.configure(test='Wilcoxon', text_format='star')
    # annotator.apply_and_annotate()

    sns.stripplot(
        data=data_MonoBeta8Ranks,
        x="session",
        y=y_values,
        hue="clinicalUse",
        ax=ax,
        size=6,
        color="black",
        alpha=0.4,  # Transparency of dots
        dodge=True,  # datapoints of groups active, inactive are plotted next to each other
    )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))

    fig.savefig(
        figures_path + f"\\ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.png",
        bbox_inches="tight",
    )

    # save Dataframe with data
    ClinicalActiveVsNonactiveContacts_filepath = os.path.join(
        results_path, f"ClinicalActiveVsNonactiveContacts_{freqBand}_{singleContacts_or_average}.pickle"
    )
    with open(ClinicalActiveVsNonactiveContacts_filepath, "wb") as file:
        pickle.dump(data_MonoBeta8Ranks, file)

    # active_and_inactive_MonoBeta8Ranks.to_json(os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.json"))

    print(
        "new files: ",
        f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.pickle",
        f"\nand: ClinicalActiveVsNonactiveContacts_statistics_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.pickle",
        "\nwritten in in: ",
        results_path,
        f"\nnew figure: ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.png",
        "\nwritten in: ",
        figures_path,
    )

    return {
        "data_MonoBeta8Ranks": data_MonoBeta8Ranks,
        "description_results": description_results,
        "significance_results": significance_results,
    }


def correlateActiveClinicalContacts_monopolPSDrelToRank1(
    filterSignal: str, normalization: str, freqBand: str, singleContacts_or_average: str
):
    """

    Input:
        - filterSignal:str, e.g. "band-pass"
        - normalization:str, e.g. "rawPsd"
        - freqBand:str, e.g. "beta"
        - singleContacts_or_average:str, e.g. "singleContacts", "averageContacts"


    1) Load the Dataframe from the file: "GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle"

    2) Load the

    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    hemispheres = ["Right", "Left"]
    sessions = ["fu3m", "fu12m", "fu18m"]
    contacts = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]

    ##################### LOAD RANKS OR PSD from monoRef_weightPsdAverageByCoordinateDistance.py #####################

    weightedByCoordinate_Dataframe = pd.DataFrame()  # concat all Dataframes from all sub, hem, sessions

    data_weightedByCoordinates = loadResults.load_GroupMonoRef_weightedPsdCoordinateDistance_pickle(
        filterSignal=filterSignal, normalization=normalization, freqBand=freqBand
    )

    # first check, which STNs and sessions exist in data
    sub_hem_keys = list(data_weightedByCoordinates.subject_hemisphere.unique())

    for STN in sub_hem_keys:
        # select only one STN
        STN_data = data_weightedByCoordinates[data_weightedByCoordinates.subject_hemisphere == STN]

        for ses in sessions:
            # first check, if session exists in STN data
            if ses not in STN_data.session.values:
                continue

            # get the dataframe per session
            STN_session_data = STN_data[STN_data.session == ses]

            # choose only directional contacts and Ring contacts 0, 3 and rank again only the chosen contacts
            STN_session_data = STN_session_data[STN_session_data["contact"].isin(contacts)]
            STN_session_data["Rank8contacts"] = STN_session_data["averaged_monopolar_PSD_beta"].rank(ascending=False)
            STN_session_data_copy = STN_session_data.copy()
            STN_session_data_copy.drop(["rank"], axis=1, inplace=True)

            # calculate the relative PSD to the highest PSD of the 8 remaining contacts
            beta_rank_1 = STN_session_data_copy[
                STN_session_data_copy["Rank8contacts"] == 1.0
            ]  # taking the row containing 1.0 in rank
            beta_rank_1 = beta_rank_1[f"averaged_monopolar_PSD_{freqBand}"].values[
                0
            ]  # just taking psdAverage of rank 1.0

            STN_session_data_copy[f"relativePSD_to_{freqBand}_Rank1from8"] = STN_session_data_copy.apply(
                lambda row: row[f"averaged_monopolar_PSD_{freqBand}"] / beta_rank_1, axis=1
            )  # in each row add to new value psd/beta_rank1
            STN_session_data_copy.drop([f"relativePSD_to_{freqBand}_Rank1"], axis=1, inplace=True)
            # session_weightedByCoordinates_copy["subject_hemisphere_monoChannel"] = session_weightedByCoordinates_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)

            weightedByCoordinate_Dataframe = pd.concat(
                [weightedByCoordinate_Dataframe, STN_session_data_copy], ignore_index=True
            )

    ##################### LOAD CLINICAL STIMULATION PARAMETERS #####################

    bestClinicalStim_file = loadResults.load_BestClinicalStimulation_excel()

    # get sheet with best clinical contacts
    BestClinicalContacts = bestClinicalStim_file["BestClinicalContacts"]

    ##################### FILTER THE MONOBETA8RANKS_DF: clinically ACTIVE contacts #####################
    activeMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():
        activeContacts = str(BestClinicalContacts.CathodalContact.values[idx])  # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        activeContacts_list = activeContacts.split("_")  # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = weightedByCoordinate_Dataframe.loc[
            (weightedByCoordinate_Dataframe["subject_hemisphere"] == sub_hem)
            & (weightedByCoordinate_Dataframe["session"] == session)
            & (weightedByCoordinate_Dataframe["contact"].isin(activeContacts_list))
        ]

        # concatenate single rows to new Dataframe
        activeMonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "active"
    activeMonoBeta8Ranks["clinicalUse"] = "active"

    ##################### FILTER THE MONOBETA8RANKS_DF: clinically INACTIVE contacts #####################
    inactiveMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():
        inactiveContacts = str(BestClinicalContacts.InactiveContacts.values[idx])  # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        inactiveContacts_list = inactiveContacts.split("_")  # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = weightedByCoordinate_Dataframe.loc[
            (weightedByCoordinate_Dataframe["subject_hemisphere"] == sub_hem)
            & (weightedByCoordinate_Dataframe["session"] == session)
            & (weightedByCoordinate_Dataframe["contact"].isin(inactiveContacts_list))
        ]

        # concatenate single rows to new Dataframe
        inactiveMonoBeta8Ranks = pd.concat([inactiveMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "non_active"
    inactiveMonoBeta8Ranks["clinicalUse"] = "inactive"

    ##################### CONCATENATE BOTH DATAFRAMES: CLINICALLY ACTIVE and INACTIVE CONTACTS #####################
    active_and_inactive_MonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, inactiveMonoBeta8Ranks], ignore_index=True)

    ##################### CHOOSE BETWEEN USING EACH SINGLE CONTACT OR AVERAGE OF ACTIVE OR INACTIVE CONTACTS #####################

    if singleContacts_or_average == "singleContacts":
        data_MonoBeta8Ranks = active_and_inactive_MonoBeta8Ranks

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg(
            '_'.join, axis=1
        )

    # average for each STN: ranks and psd of active or inactive contacts
    elif singleContacts_or_average == "averageContacts":
        STN_average_activeVsInactiveContacts = (
            {}
        )  # store averages of ranks and psd values of active vs. inactive contacts per STN

        STN_unique = list(
            active_and_inactive_MonoBeta8Ranks["subject_hemisphere"].unique()
        )  # list of all existing STNs

        for STN in STN_unique:
            # get dataframe only of one STN
            STN_dataframe = active_and_inactive_MonoBeta8Ranks.loc[
                (active_and_inactive_MonoBeta8Ranks.subject_hemisphere == STN)
            ]

            # get all existing sessions per STN
            sessions_unique = list(STN_dataframe["session"].unique())

            for ses in sessions_unique:
                STN_session_dataframe = STN_dataframe.loc[(STN_dataframe.session == ses)]

                # get average of active contacts
                STN_session_active = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "active")]
                MEANrank_active = STN_session_active["Rank8contacts"].values.mean()
                MEANrelToRank1psd_active = STN_session_active["relativePSD_to_beta_Rank1from8"].values.mean()

                # get average of inactive contacts
                STN_session_inactive = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "inactive")]
                MEANrank_inactive = STN_session_inactive["Rank8contacts"].values.mean()
                MEANrelToRank1psd_inactive = STN_session_inactive["relativePSD_to_beta_Rank1from8"].values.mean()

                # store MEAN values in dictionary
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_active"] = [
                    STN,
                    ses,
                    MEANrank_active,
                    MEANrelToRank1psd_active,
                    "active",
                ]
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_inactive"] = [
                    STN,
                    ses,
                    MEANrank_inactive,
                    MEANrelToRank1psd_inactive,
                    "inactive",
                ]

        # transform the dictionary to Dataframe
        STN_average_activeVsInactive_DF = pd.DataFrame(STN_average_activeVsInactiveContacts)
        STN_average_activeVsInactive_DF.rename(
            index={
                0: "subject_hemisphere",
                1: "session",
                2: "MEAN_beta_rank",
                3: "MEANrelToRank1psd",
                4: "clinicalUse",
            },
            inplace=True,
        )
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.transpose()
        # important to transform datatype of columns MEAN_beta_rank and MEAN_beta_psd to float (otherwise Error when plotting with seaborn)
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEAN_beta_rank": float})
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEANrelToRank1psd": float})

        # use the dataframe with MEAN values for plotting
        data_MonoBeta8Ranks = STN_average_activeVsInactive_DF

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg(
            '_'.join, axis=1
        )

    ##################### PERFORM Wilcoxon signed rank TEST  #####################

    ses_clinicalUse = [
        "fu3m_active",
        "fu3m_inactive",
        "fu12m_active",
        "fu12m_inactive",
        "fu18m_active",
        "fu18m_inactive",
    ]
    ses_clinicalUse_wilcoxon = [
        ("fu3m_active", "fu3m_inactive"),
        ("fu12m_active", "fu12m_inactive"),
        ("fu18m_active", "fu18m_inactive"),
    ]
    pairs = list(combinations(ses_clinicalUse, 2))
    all_results_wilcoxon = []
    describe_arrays = {}

    # pair = tuple e.g. fu3m_active, fu3m_inactive
    # for pair in pairs:
    for s_c_wcx in ses_clinicalUse_wilcoxon:
        firstInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[0])]
        secondInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[1])]

        if singleContacts_or_average == "singleContacts":
            firstInPair = np.array(firstInPair.averaged_monopolar_PSD_beta.values)
            secondInPair = np.array(secondInPair.averaged_monopolar_PSD_beta.values)

        elif singleContacts_or_average == "averageContacts":
            firstInPair = np.array(firstInPair.MEANrelToRank1psd.values)
            secondInPair = np.array(secondInPair.MEANrelToRank1psd.values)

        # Perform Wilcoxon signed rank Test
        results_wcx = pg.wilcoxon(
            firstInPair, secondInPair
        )  # pair is always a tuple, comparing first and second component of this tuple
        results_wcx[f'comparison_relativePsdToRank1_{singleContacts_or_average}'] = '_'.join(
            s_c_wcx
        )  # new column "comparison" with the pair being compared e.g. fu3m_active and fu3m_inactive

        all_results_wilcoxon.append(results_wcx)

    significance_results = pd.concat(all_results_wilcoxon)

    ##################### GET STATISTICAL IMPORTANT FEATURES #####################
    # describe all 6 groups
    for s_c in ses_clinicalUse:
        # get array of each group
        group = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c)]

        if singleContacts_or_average == "singleContacts":
            group = np.array(group.averaged_monopolar_PSD_beta.values)

        elif singleContacts_or_average == "averageContacts":
            group = np.array(group.MEANrelToRank1psd.values)

        description = scipy.stats.describe(group)

        describe_arrays[f"{s_c}_{singleContacts_or_average}"] = description

    description_results = pd.DataFrame(describe_arrays)
    description_results.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_results = description_results.transpose()

    ##################### STORE RESULTS IN DICTIONARY AND SAVE #####################

    results_dictionary = {"significance_results": significance_results, "description_results": description_results}

    # save as pickle
    results_filepath = os.path.join(
        results_path,
        f"ClinicalActiveVsNonactiveContacts_statistics_relativeToRank1_psd_{freqBand}_{singleContacts_or_average}.pickle",
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dictionary, file)

    ##################### PLOT VIOLINPLOT OF relative PSD to rank 1 OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    if singleContacts_or_average == "singleContacts":
        y_values = "averaged_monopolar_PSD_beta"
        y_label = "relative Beta PSD [% of rank 1]"
        title = "relative Beta PSD of clinically active vs. inactive stimulation contacts"
        y_lim = 0, 1.25

    elif singleContacts_or_average == "averageContacts":
        y_values = "MEANrelToRank1psd"
        y_label = "Mean rel. Beta PSD [% of rank 1]"
        title = "relative Mean Beta PSD of clinically active vs. inactive stimulation contacts"
        y_lim = 0, 1.25

    fig = plt.figure()
    ax = fig.add_subplot()

    # sns.violinplot(data=data_MonoBeta8Ranks, x="session_clinicalUse", y=y_values, hue="clinicalUse", palette="Set2", inner="box", ax=ax)
    sns.violinplot(
        data=data_MonoBeta8Ranks, x="session", y=y_values, hue="clinicalUse", palette="Set3", inner="box", ax=ax
    )

    # statistical test
    # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
    # pairs = list(combinations(ses_clinicalUse, 2))

    # annotator = Annotator(ax, pairs, data=active_and_inactive_MonoBeta8Ranks, x='session_clinicalUse', y=y_values)
    # annotator.configure(test='Wilcoxon', text_format='star')
    # annotator.apply_and_annotate()

    sns.stripplot(
        data=data_MonoBeta8Ranks,
        x="session",
        y=y_values,
        hue="clinicalUse",
        ax=ax,
        size=6,
        color="black",
        alpha=0.4,  # Transparency of dots
        dodge=True,  # datapoints of groups active, inactive are plotted next to each other
    )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))

    fig.savefig(
        figures_path
        + f"\\ClinicalActiveVsNonactiveContacts_relativeToRank1_{normalization}_{freqBand}_{singleContacts_or_average}.png",
        bbox_inches="tight",
    )

    # save Dataframe with data
    ClinicalActiveVsNonactiveContacts_filepath = os.path.join(
        results_path,
        f"ClinicalActiveVsNonactiveContacts_relativeToRank1_psd_{freqBand}_{singleContacts_or_average}.pickle",
    )
    with open(ClinicalActiveVsNonactiveContacts_filepath, "wb") as file:
        pickle.dump(data_MonoBeta8Ranks, file)

    # active_and_inactive_MonoBeta8Ranks.to_json(os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_relativeToRank1_psd_{freqBand}.json"))

    print(
        "new files: ",
        f"ClinicalActiveVsNonactiveContacts_relativeToRank1_psd_{freqBand}_{singleContacts_or_average}.pickle",
        f"\nand: ClinicalActiveVsNonactiveContacts_statistics_relativeToRank1_psd_{freqBand}_{singleContacts_or_average}.pickle",
        "\nwritten in in: ",
        results_path,
        f"\nnew figure: ClinicalActiveVsNonactiveContacts_relativeToRank1_psd_{freqBand}__{singleContacts_or_average}.png",
        "\nwritten in: ",
        figures_path,
    )

    return {
        "weightedByCoordinate_Dataframe": weightedByCoordinate_Dataframe,
        "data_MonoBeta8Ranks": data_MonoBeta8Ranks,
        "results_dictionary": results_dictionary,
    }


def active_vs_inactive_contacts_monopolPSD(
    filterSignal: str, normalization: str, freqBand: str, singleContacts_or_average: str
):
    """

    Input:
        - filterSignal:str, e.g. "band-pass"
        - normalization:str, e.g. "rawPsd"
        - freqBand:str, e.g. "beta"
        - singleContacts_or_average:str, e.g. "singleContacts", "averageContacts"


    1) Load the Dataframe from "monopol_rel_psd_from0To8_{freqBand}_{normalization}_{signalFilter}.pickle"
        -

    2) Load the

    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    ##################### LOAD RANKS OR PSD from monoRef_weightPsdAverageByCoordinateDistance.py #####################

    data_monopol_rel_0_to_1 = loadResults.load_monopol_rel_psd_from0To8_pickle(
        filterSignal=filterSignal, normalization=normalization, freqBand=freqBand
    )

    ##################### CHOOSE BETWEEN USING EACH SINGLE CONTACT OR AVERAGE OF ACTIVE OR INACTIVE CONTACTS #####################

    if singleContacts_or_average == "singleContacts":
        data_MonoBeta8Ranks = data_monopol_rel_0_to_1

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg(
            '_'.join, axis=1
        )

    # average for each STN: ranks and psd of active or inactive contacts
    elif singleContacts_or_average == "averageContacts":
        STN_average_activeVsInactiveContacts = (
            {}
        )  # store averages of ranks and psd values of active vs. inactive contacts per STN

        STN_unique = list(data_monopol_rel_0_to_1["subject_hemisphere"].unique())  # list of all existing STNs

        for STN in STN_unique:
            # get dataframe only of one STN
            STN_dataframe = data_monopol_rel_0_to_1.loc[(data_monopol_rel_0_to_1.subject_hemisphere == STN)]

            # get all existing sessions per STN
            sessions_unique = list(STN_dataframe["session"].unique())

            for ses in sessions_unique:
                STN_session_dataframe = STN_dataframe.loc[(STN_dataframe.session == ses)]

                # get average of active contacts
                STN_session_active = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "active")]
                MEANrank_active = STN_session_active["Rank8contacts"].values.mean()
                MEANrelToRank1psd_active = STN_session_active[f"relativePSD_{freqBand}_from_0_to_1"].values.mean()

                # get average of inactive contacts
                STN_session_inactive = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "inactive")]
                MEANrank_inactive = STN_session_inactive["Rank8contacts"].values.mean()
                MEANrelToRank1psd_inactive = STN_session_inactive[f"relativePSD_{freqBand}_from_0_to_1"].values.mean()

                # store MEAN values in dictionary
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_active"] = [
                    STN,
                    ses,
                    MEANrank_active,
                    MEANrelToRank1psd_active,
                    "active",
                ]
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_inactive"] = [
                    STN,
                    ses,
                    MEANrank_inactive,
                    MEANrelToRank1psd_inactive,
                    "inactive",
                ]

        # transform the dictionary to Dataframe
        STN_average_activeVsInactive_DF = pd.DataFrame(STN_average_activeVsInactiveContacts)
        STN_average_activeVsInactive_DF.rename(
            index={
                0: "subject_hemisphere",
                1: "session",
                2: "MEAN_beta_rank",
                3: "MEANrelToRank1psd",
                4: "clinicalUse",
            },
            inplace=True,
        )
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.transpose()
        # important to transform datatype of columns MEAN_beta_rank and MEAN_beta_psd to float (otherwise Error when plotting with seaborn)
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEAN_beta_rank": float})
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEANrelToRank1psd": float})

        # use the dataframe with MEAN values for plotting
        data_MonoBeta8Ranks = STN_average_activeVsInactive_DF

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg(
            '_'.join, axis=1
        )

    ##################### PERFORM Wilcoxon signed rank TEST  #####################

    ses_clinicalUse = [
        "fu3m_active",
        "fu3m_inactive",
        "fu12m_active",
        "fu12m_inactive",
        "fu18m_active",
        "fu18m_inactive",
    ]
    ses_clinicalUse_wilcoxon = [
        ("fu3m_active", "fu3m_inactive"),
        ("fu12m_active", "fu12m_inactive"),
        ("fu18m_active", "fu18m_inactive"),
    ]
    # pairs = list(combinations(ses_clinicalUse, 2))
    all_results_wilcoxon = []
    describe_arrays = {}

    # pair = tuple e.g. fu3m_active, fu3m_inactive
    # for pair in pairs:
    for s_c_wcx in ses_clinicalUse_wilcoxon:
        firstInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[0])]
        secondInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[1])]

        if singleContacts_or_average == "singleContacts":
            firstInPair = np.array(firstInPair.averaged_monopolar_PSD_beta.values)
            secondInPair = np.array(secondInPair.averaged_monopolar_PSD_beta.values)

        elif singleContacts_or_average == "averageContacts":
            firstInPair = np.array(firstInPair.MEANrelToRank1psd.values)
            secondInPair = np.array(secondInPair.MEANrelToRank1psd.values)

        # Perform Wilcoxon signed rank Test
        results_wcx = pg.wilcoxon(
            firstInPair, secondInPair
        )  # pair is always a tuple, comparing first and second component of this tuple
        results_wcx[f'comparison_relativePsdToRank1_{singleContacts_or_average}'] = '_'.join(
            s_c_wcx
        )  # new column "comparison" with the pair being compared e.g. fu3m_active and fu3m_inactive

        all_results_wilcoxon.append(results_wcx)

    significance_results = pd.concat(all_results_wilcoxon)

    ##################### GET STATISTICAL IMPORTANT FEATURES #####################
    # describe all 6 groups
    for s_c in ses_clinicalUse:
        # get array of each group
        group = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c)]

        if singleContacts_or_average == "singleContacts":
            group = np.array(group.averaged_monopolar_PSD_beta.values)

        elif singleContacts_or_average == "averageContacts":
            group = np.array(group.MEANrelToRank1psd.values)

        description = scipy.stats.describe(group)

        describe_arrays[f"{s_c}_{singleContacts_or_average}"] = description

    description_results = pd.DataFrame(describe_arrays)
    description_results.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_results = description_results.transpose()

    ##################### STORE RESULTS IN DICTIONARY AND SAVE #####################

    results_dictionary = {"significance_results": significance_results, "description_results": description_results}

    # save as pickle
    results_filepath = os.path.join(
        results_path,
        f"ClinicalActiveVsNonactiveContacts_statistics_rel_psd_0_to_1_{freqBand}_{singleContacts_or_average}.pickle",
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dictionary, file)

    ##################### PLOT VIOLINPLOT OF relative PSD to rank 1 OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    if singleContacts_or_average == "singleContacts":
        y_values = "relativePSD_beta_from_0_to_1"
        y_label = "relative Beta PSD (range from 0 to 1)"
        title = "relative Beta PSD of clinically active vs. inactive stimulation contacts"
        y_lim = -0.3, 1.25

    elif singleContacts_or_average == "averageContacts":
        y_values = "MEANrelToRank1psd"
        y_label = "Mean rel. Beta PSD (range from 0 to 1)"
        title = "relative Mean Beta PSD of clinically active vs. inactive stimulation contacts"
        y_lim = -0.3, 1.25

    fig = plt.figure()
    ax = fig.add_subplot()

    # sns.violinplot(data=data_MonoBeta8Ranks, x="session_clinicalUse", y=y_values, hue="clinicalUse", palette="Set2", inner="box", ax=ax)
    sns.violinplot(
        data=data_MonoBeta8Ranks, x="session", y=y_values, hue="clinicalUse", palette="Set3", inner="box", ax=ax
    )

    # statistical test
    # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
    # pairs = list(combinations(ses_clinicalUse, 2))

    # annotator = Annotator(ax, pairs, data=active_and_inactive_MonoBeta8Ranks, x='session_clinicalUse', y=y_values)
    # annotator.configure(test='Wilcoxon', text_format='star')
    # annotator.apply_and_annotate()

    sns.stripplot(
        data=data_MonoBeta8Ranks,
        x="session",
        y=y_values,
        hue="clinicalUse",
        ax=ax,
        size=6,
        # color="black",
        alpha=0.4,  # Transparency of dots
        dodge=True,  # datapoints of groups active, inactive are plotted next to each other
    )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))

    fig.savefig(
        figures_path
        + f"\\ClinicalActiveVsNonactiveContacts_rel_psd_0_to_1_{normalization}_{freqBand}_{singleContacts_or_average}.png",
        bbox_inches="tight",
    )

    # save Dataframe with data
    ClinicalActiveVsNonactiveContacts_filepath = os.path.join(
        results_path, f"ClinicalActiveVsNonactiveContacts_rel_psd_0_to_1_{freqBand}_{singleContacts_or_average}.pickle"
    )
    with open(ClinicalActiveVsNonactiveContacts_filepath, "wb") as file:
        pickle.dump(data_MonoBeta8Ranks, file)

    # active_and_inactive_MonoBeta8Ranks.to_json(os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_relativeToRank1_psd_{freqBand}.json"))

    print(
        "new files: ",
        f"ClinicalActiveVsNonactiveContacts_rel_psd_0_to_1_{freqBand}_{singleContacts_or_average}.pickle",
        f"\nand: ClinicalActiveVsNonactiveContacts_statistics_rel_psd_0_to_1_{freqBand}_{singleContacts_or_average}.pickle",
        "\nwritten in in: ",
        results_path,
        f"\nnew figure: ClinicalActiveVsNonactiveContacts_rel_psd_0_to_1_{freqBand}__{singleContacts_or_average}.png",
        "\nwritten in: ",
        figures_path,
    )

    return {
        "data_monopol_rel_0_to_1": data_monopol_rel_0_to_1,
        "data_MonoBeta8Ranks": data_MonoBeta8Ranks,
        "results_dictionary": results_dictionary,
    }


def active_contacts_per_rank(
    freqBand: str,
):
    """
    Loads the file:  ClinicalActiveVsNonactiveContacts_relativeToRank1_psd_beta_singleContacts.pickle


    """

    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # load the grouped file with monopolar PSD averages, ranks per electrode and active or inactive use per contact
    active_vs_inactive = loadResults.load_ClinicalActiveVsInactive(
        freqBand=freqBand, attribute="relativeToRank1_psd", singleContacts_or_average="singleContacts"
    )

    ranks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sessions = ["fu3m", "fu12m", "fu18m"]

    percentage_active = {}
    sample_size = {}

    for session in sessions:
        # filter dependending on session
        session_df = active_vs_inactive.loc[(active_vs_inactive.session == session)]

        # sample size within each session
        sample_counts = session_df.session.value_counts()
        sample_size[f"{session}"] = [session, sample_counts[0]]

        for rank in ranks:
            # filter all contacts of the same rank from column "Rank8contacts"
            rank_df = session_df.loc[(active_vs_inactive.Rank8contacts == rank)]

            # calculate percentage of active from all 1.0 ranked contacts
            counts = rank_df.clinicalUse.value_counts()  # counts[0] = inactive, counts[1] = active

            inactive_contacts = counts[0]

            # if no active contacts in the dataframe, active = 0
            if "active" not in rank_df.values:
                active_contacts = 0

            elif "active" in rank_df.values:
                active_contacts = counts[1]

            total = inactive_contacts + active_contacts
            percentage_active[f"{session}_{rank}"] = [session, rank, ((active_contacts / total) * 100)]

    # write Dataframe out of the dictionaries with percentages of active contacts per rank
    percentage_active_df = pd.DataFrame(percentage_active)
    percentage_active_df.rename(index={0: "session", 1: "beta_rank", 2: "percentage_active"}, inplace=True)
    percentage_active_df = percentage_active_df.transpose()

    # write Dataframe with information of how many samples are within a session
    sample_size_df = pd.DataFrame(sample_size)
    sample_size_df.rename(index={0: "session", 1: "number_of_contacts"}, inplace=True)
    sample_size_df = sample_size_df.transpose()

    # Divide the Dataframe into 3 groups, one per session, each session will be plotted as different line
    fu3m_df = percentage_active_df.loc[(percentage_active_df.session == "fu3m")]
    fu12m_df = percentage_active_df.loc[(percentage_active_df.session == "fu12m")]
    fu18m_df = percentage_active_df.loc[(percentage_active_df.session == "fu18m")]

    ################### PLOT PERCENTAGE OF ACTIVE CONTACTS PER SESSION ###################
    fig = plt.figure()
    plt.plot(fu3m_df["beta_rank"], fu3m_df["percentage_active"], label="3MFU")
    plt.plot(fu12m_df["beta_rank"], fu12m_df["percentage_active"], label="12MFU")
    plt.plot(fu18m_df["beta_rank"], fu18m_df["percentage_active"], label="18MFU")

    plt.xlabel("'beta-rank' of contacts within an electrode")
    plt.ylabel("active contacts [%]")
    plt.title("How many 'beta-ranked' contacts are clinically used?")
    plt.legend()

    fig.savefig(figures_path + f"\\{freqBand}_ranked_active_contacts.png")

    return {
        "sample_size_df": sample_size_df,
        "fu3m_df": fu3m_df,
        "fu12m_df": fu12m_df,
        "fu18m_df": fu18m_df,
    }


def bestClinicalStimContacts_LevelsComparison():
    """

    Load the Excel file with clinical stimulation parameters from the data folder.
    Plot the difference of levels between several sessions.


    Input:
        -

    """
    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    ##################### LOAD CLINICAL STIMULATION PARAMETERS #####################

    bestClinicalStim_file = loadResults.load_BestClinicalStimulation_excel()
    compareLevels_sheet = bestClinicalStim_file["compareLevels"]

    ##################### COMPARE CONTACT LEVELS OF FU3M AND FU12M #####################

    # select only fu3m rows and fu12m rows and merge
    fu3m_level = compareLevels_sheet[compareLevels_sheet.session == "fu3m"]
    fu12m_level = compareLevels_sheet[compareLevels_sheet.session == "fu12m"]

    # merge fu3m and fu12m
    comparefu3m_fu12m = fu3m_level.merge(fu12m_level, left_on="subject_hemisphere", right_on="subject_hemisphere")

    # calculate the difference between levels, add to a new column
    comparefu3m_fu12m["difference_levels"] = (
        comparefu3m_fu12m["ContactLevel_x"] - comparefu3m_fu12m["ContactLevel_y"]
    ).apply(abs)
    comparefu3m_fu12m.dropna(axis=0)

    ##################### COMPARE CONTACT LEVELS OF FU12M AND LONGTERM #####################

    # select only LONGTERM rows, including fu18m, fu20m, fu22m
    longterm_level = compareLevels_sheet[
        (compareLevels_sheet.session == "fu18m")
        | (compareLevels_sheet.session == "fu20m")
        | (compareLevels_sheet.session == "fu22m")
    ]  # use | instead of or here

    # merge fu12m and longterm
    comparefu12m_longterm = fu12m_level.merge(
        longterm_level, left_on="subject_hemisphere", right_on="subject_hemisphere"
    )

    # calculate the difference between levels, add to a new column
    comparefu12m_longterm["difference_levels"] = (
        comparefu12m_longterm["ContactLevel_x"] - comparefu12m_longterm["ContactLevel_y"]
    ).apply(abs)
    comparefu12m_longterm.dropna(axis=0)

    ##################### PLOT BOTH COMPARISONS: FU3M - FU12M AND FU12M - LONGTERM #####################

    colors0 = sns.color_palette("viridis", n_colors=len(comparefu3m_fu12m.index))
    colors1 = sns.color_palette("viridis", n_colors=len(comparefu12m_longterm.index))

    fontdict = {"size": 25}

    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    ##################### PLOT CONTACT LEVELS OF FU3M AND FU12M #####################
    sns.histplot(
        data=comparefu3m_fu12m,
        x="difference_levels",
        stat="count",
        hue="subject_hemisphere",
        multiple="stack",
        bins=np.arange(-0.25, 4, 0.5),
        palette=colors0,
        ax=axes[0],
    )

    axes[0].set_title("3MFU vs. 12MFU", fontdict=fontdict)

    legend3_12 = axes[0].get_legend()
    handles = legend3_12.legendHandles

    legend3_12_list = list(comparefu3m_fu12m.subject_hemisphere.values)
    axes[0].legend(handles, legend3_12_list, title='subject hemisphere', title_fontsize=15, fontsize=15)

    ##################### PLOT CONTACT LEVELS OF FU12M AND LONTERM #####################
    sns.histplot(
        data=comparefu12m_longterm,
        x="difference_levels",
        stat="count",
        hue="subject_hemisphere",
        multiple="stack",
        bins=np.arange(-0.25, 4, 0.5),
        palette=colors1,
        ax=axes[1],
    )

    axes[1].set_title("12MFU vs. longterm-FU (18, 20, 22MFU)", fontdict=fontdict)

    legend12_longterm = axes[1].get_legend()
    handles = legend12_longterm.legendHandles

    legend12_longterm_list = list(comparefu12m_longterm.subject_hemisphere.values)
    axes[1].legend(handles, legend12_longterm_list, title='subject hemisphere', title_fontsize=15, fontsize=15)

    ##################### ADJUST THE PLOTS #####################

    for ax in axes:
        ax.set_xlabel("difference between contact levels", fontsize=25)
        ax.set_ylabel("Count", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)

    fig.suptitle("Difference of active contact levels", fontsize=30)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()

    fig.savefig(
        figures_path + "\\ActiveClinicalStimContact_Levels_fu3m_fu12m_and_fu12m_longterm.png", bbox_inches="tight"
    )

    ### save the Dataframes with pickle
    comparefu3m_fu12m_filepath = os.path.join(results_path, f"ActiveClinicalStimContact_Levels_fu3m_fu12m.pickle")
    with open(comparefu3m_fu12m_filepath, "wb") as file:
        pickle.dump(comparefu3m_fu12m, file)

    comparefu12m_longterm_filepath = os.path.join(
        results_path, f"ActiveClinicalStimContact_Levels_fu12m_longterm.pickle"
    )
    with open(comparefu12m_longterm_filepath, "wb") as file:
        pickle.dump(comparefu12m_longterm, file)

    print(
        "new file: ",
        "ActiveClinicalStimContact_Levels_fu3m_fu12m.pickle",
        "\nwritten in in: ",
        results_path,
        "\nnew figure: ActiveClinicalStimContact_Levels_fu3m_fu12m_and_fu12m_longterm.png",
        "\nwritten in: ",
        figures_path,
    )


def fooof_mono_beta_and_clinical_activity_write_dataframes(fooof_version: str):
    """
    Combine the dataframes with fooof monopolar beta power estimates and best clinical stimulation parameters

    """

    # Input:
    #     - similarity_calculation: "inverse_distance", "exp_neg_distance"

    # - loaded fooof data with monopolar beta power estimations for all contacts (n=8):
    #     filename: fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}.pickle

    # - load the best clnical stimulation Excel file
    #     BestClinicalStimulation.xlsx
    #     in c:\Users\jebe12\Research\Longterm_beta_project\data

    # 1) add to the monopolar beta power dataframe following columns:
    #     - current_polarity
    #     - clinical_activity
    #     - sessioni_clinical_activity

    # 2) write a second dataframe with averaged beta power and beta ranks per group (electrode, active vs inactive)

    # load the fooof mono beta data
    loaded_fooof_mono_beta = loadResults.load_pickle_group_result(
        filename="fooof_monoRef_all_contacts_weight_beta_psd_by_inverse_distance", fooof_version=fooof_version
    )

    # load Excel file with best clinical stimulation parameters
    best_clinical_stimulation = loadResults.load_BestClinicalStimulation_excel()
    best_clinical_contacts = best_clinical_stimulation["BestContacts_one_longterm"]  # or BestClinicalContacts

    ##################### FILTER THE monopolar beta dataframe: clinically ACTIVE contacts #####################
    active_and_inactive_contacts_data = pd.DataFrame()

    for idx, row in best_clinical_contacts.iterrows():
        sub_hem = best_clinical_contacts.subject_hemisphere.values[idx]
        session = best_clinical_contacts.session.values[idx]

        ############# ACTIVE #############
        active_contacts = str(best_clinical_contacts.CathodalContact.values[idx])  # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        active_contacts_list = active_contacts.split("_")  # e.g. ["2A", "2B", "2C", "3"]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of active_contacts
        sub_hem_ses_active = loaded_fooof_mono_beta.loc[
            (loaded_fooof_mono_beta["subject_hemisphere"] == sub_hem)
            & (loaded_fooof_mono_beta["session"] == session)
            & (loaded_fooof_mono_beta["contact"].isin(active_contacts_list))
        ]
        sub_hem_ses_active_copy = sub_hem_ses_active.copy()

        # add the current polarity and clinical activity
        current_polarity = float(best_clinical_contacts.currentPolarity.values[idx])  # e.g. 0.33
        sub_hem_ses_active_copy["current_polarity"] = current_polarity
        sub_hem_ses_active_copy["clinical_activity"] = "active"

        ############# INACTIVE #############
        inactive_contacts = str(best_clinical_contacts.InactiveContacts.values[idx])  # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        inactive_contacts_list = inactive_contacts.split("_")  # e.g. ["2A", "2B", "2C", "3"]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of active_contacts
        sub_hem_ses_inactive = loaded_fooof_mono_beta.loc[
            (loaded_fooof_mono_beta["subject_hemisphere"] == sub_hem)
            & (loaded_fooof_mono_beta["session"] == session)
            & (loaded_fooof_mono_beta["contact"].isin(inactive_contacts_list))
        ]
        sub_hem_ses_inactive_copy = sub_hem_ses_inactive.copy()

        # add the current polarity and clinical activity
        sub_hem_ses_inactive_copy["current_polarity"] = 0
        sub_hem_ses_inactive_copy["clinical_activity"] = "inactive"

        # concatenate single rows to new Dataframe
        active_and_inactive_contacts_data = pd.concat(
            [active_and_inactive_contacts_data, sub_hem_ses_active_copy, sub_hem_ses_inactive_copy], ignore_index=True
        )

    # clean up the dataframe
    active_and_inactive_contacts_data = active_and_inactive_contacts_data.drop(columns=["index"])

    # add session and clinical activity as combined new column, necessary for grouping later
    active_and_inactive_contacts_data["session_clinical_activity"] = active_and_inactive_contacts_data[
        ["session", "clinical_activity"]
    ].agg("_".join, axis=1)

    ##################### GET AVERAGE OF BETA POWER OR BETA RANKS: CREATE NEW DATAFRAME #####################

    electrode_average_active_vs_inactive = (
        {}
    )  # store averages of ranks and psd values of active vs. inactive contacts per STN

    STN_unique = list(active_and_inactive_contacts_data["subject_hemisphere"].unique())  # list of all existing STNs

    for STN in STN_unique:
        # get dataframe only of one STN
        STN_dataframe = active_and_inactive_contacts_data.loc[
            (active_and_inactive_contacts_data.subject_hemisphere == STN)
        ]

        # get all existing sessions per STN
        sessions_unique = list(STN_dataframe["session"].unique())

        for ses in sessions_unique:
            STN_session_dataframe = STN_dataframe.loc[(STN_dataframe.session == ses)]

            # get average of active contacts
            STN_session_active = STN_session_dataframe.loc[(STN_session_dataframe.clinical_activity == "active")]
            mean_active_ranks = STN_session_active["rank_8"].values.mean()
            mean_active_beta_psd = STN_session_active["estimated_monopolar_beta_psd"].values.mean()
            mean_active_beta_psd_rel_to_rank1 = STN_session_active["beta_psd_rel_to_rank1"].values.mean()
            mean_active_beta_psd_rel_range_0_to_1 = STN_session_active["beta_psd_rel_range_0_to_1"].values.mean()

            # get average of inactive contacts
            STN_session_inactive = STN_session_dataframe.loc[(STN_session_dataframe.clinical_activity == "inactive")]
            mean_inactive_ranks = STN_session_inactive["rank_8"].values.mean()
            mean_inactive_beta_psd = STN_session_inactive["estimated_monopolar_beta_psd"].values.mean()
            mean_inactive_beta_psd_rel_to_rank1 = STN_session_inactive["beta_psd_rel_to_rank1"].values.mean()
            mean_inactive_beta_psd_rel_range_0_to_1 = STN_session_inactive["beta_psd_rel_range_0_to_1"].values.mean()

            # store MEAN values in dictionary
            electrode_average_active_vs_inactive[f"{STN}_{ses}_active"] = [
                STN,
                ses,
                mean_active_ranks,
                mean_active_beta_psd,
                mean_active_beta_psd_rel_to_rank1,
                mean_active_beta_psd_rel_range_0_to_1,
                "active",
            ]
            electrode_average_active_vs_inactive[f"{STN}_{ses}_inactive"] = [
                STN,
                ses,
                mean_inactive_ranks,
                mean_inactive_beta_psd,
                mean_inactive_beta_psd_rel_to_rank1,
                mean_inactive_beta_psd_rel_range_0_to_1,
                "inactive",
            ]

    # transform the dictionary to Dataframe
    electrode_average_active_vs_inactive_df = pd.DataFrame(electrode_average_active_vs_inactive)
    electrode_average_active_vs_inactive_df.rename(
        index={
            0: "subject_hemisphere",
            1: "session",
            2: "electrode_mean_beta_rank",
            3: "electrode_mean_beta_psd",
            4: "electrode_mean_beta_psd_rel_to_rank1",
            5: "electrode_mean_beta_psd_rel_range_0_to_1",
            6: "clinical_activity",
        },
        inplace=True,
    )
    electrode_average_active_vs_inactive_df = electrode_average_active_vs_inactive_df.transpose()

    # important to transform datatype of columns MEAN_beta_rank and MEAN_beta_psd to float (otherwise Error when plotting with seaborn)
    electrode_average_active_vs_inactive_df = electrode_average_active_vs_inactive_df.astype(
        {"electrode_mean_beta_rank": float}
    )
    electrode_average_active_vs_inactive_df = electrode_average_active_vs_inactive_df.astype(
        {"electrode_mean_beta_psd": float}
    )
    electrode_average_active_vs_inactive_df = electrode_average_active_vs_inactive_df.astype(
        {"electrode_mean_beta_psd_rel_to_rank1": float}
    )
    electrode_average_active_vs_inactive_df = electrode_average_active_vs_inactive_df.astype(
        {"electrode_mean_beta_psd_rel_range_0_to_1": float}
    )

    electrode_average_active_vs_inactive_df["session_clinical_activity"] = electrode_average_active_vs_inactive_df[
        ["session", "clinical_activity"]
    ].agg("_".join, axis=1)

    return {
        "single_contacts": active_and_inactive_contacts_data,
        "electrode_average": electrode_average_active_vs_inactive_df,
    }


def fooof_mono_beta_threshold_label(similarity_calculation: str, beta_threshold: float):
    """
    Input:
        - similarity_calculation: "inverse_distance", "exp_neg_distance"
        - beta_threshold: e.g. 0.7

    This function adds a column to the dataframe written with fooof_mono_beta_active_vs_inactive = activeStimContacts.fooof_mono_beta_and_clinical_activity_write_dataframes()

    Depending on the given threshold, each row will be labeled either above_threshold or below_threshold depending on the normalized beta power and the given threshold.


    """

    beta_and_clinical_activity_data = fooof_mono_beta_and_clinical_activity_write_dataframes(
        similarity_calculation=similarity_calculation
    )

    data_to_analyze = beta_and_clinical_activity_data["single_contacts"]

    # add column "beta_relevance" and add labels depending on the beta psd rel to rank 1 being above or below a threshold
    data_to_analyze_copy = data_to_analyze.copy()

    # define a function, that return the label, that will be added into the dataframe depending on the normalized beta value
    def label_beta_power(normalized_beta):
        if normalized_beta > beta_threshold:
            return "above"

        elif normalized_beta < beta_threshold:
            return "below"

    # Apply this function to every row and add the label to a new column
    data_to_analyze_copy[f"beta_threshold_{beta_threshold}"] = data_to_analyze_copy["beta_psd_rel_to_rank1"].apply(
        lambda x: label_beta_power(x)
    )

    return data_to_analyze_copy


def fooof_mono_beta_count_active_and_above_threshold(similarity_calculation: str, beta_threshold: float):
    """
    Input:
        - similarity_calculation: "inverse_distance", "exp_neg_distance"
        - beta_threshold: e.g. 0.7

    Count for each STN and each session, all ratios
        - active/all beta above threshold
        - inactive/all beta above threshold
        - active/all beta below threshold
        - inactive/all beta below threshold


    """

    sessions = ["fu3m", "fu12m", "fu18m"]

    beta_threshold_data = fooof_mono_beta_threshold_label(
        similarity_calculation=similarity_calculation, beta_threshold=beta_threshold
    )

    sub_hem_list = list(beta_threshold_data.subject_hemisphere.unique())

    activity_beta_threshold_dict = {}
    contingency_dict = {}

    for stn in sub_hem_list:
        stn_data = beta_threshold_data.loc[beta_threshold_data.subject_hemisphere == stn]

        for ses in sessions:
            # check if session exists for this stn
            if ses not in stn_data.session.values:
                continue

            else:
                ses_stn_data = stn_data.loc[stn_data.session == ses]

            contingency_table = pd.crosstab(
                ses_stn_data[f"beta_threshold_{beta_threshold}"], ses_stn_data["clinical_activity"]
            )  # table counting all binary value combinations
            # store contingency table in dictionary
            contingency_dict[f"{stn}_{ses}"] = contingency_table

            # get rel values of each combination
            total_above = np.sum(contingency_table.iloc[0])  # first row
            total_below = np.sum(contingency_table.iloc[1])  # second row
            total_active = np.sum(contingency_table["active"])
            total_inactive = np.sum(contingency_table["inactive"])

            active_from_total_above = (
                contingency_table["active"][0]
            ) / total_above  # how many contacts are active from all contacts that are above threshold
            inactive_from_total_above = (contingency_table["inactive"][0]) / total_above

            active_from_total_below = (
                contingency_table["active"][1]
            ) / total_below  # how many contacts are active from all contacts that are below threshold
            inactive_from_total_below = (contingency_table["inactive"][1]) / total_below

            above_from_total_active = (
                contingency_table["active"][0]
            ) / total_active  # how many contacts are above threshold from all active contacts
            below_from_total_active = (contingency_table["active"][1]) / total_active

            above_from_total_inactive = (
                contingency_table["inactive"][0]
            ) / total_inactive  # how many contacts are above threshold from all active contacts
            below_from_total_inactive = (contingency_table["inactive"][1]) / total_inactive

            activity_beta_threshold_dict[f"{stn}_{ses}"] = [
                stn,
                ses,
                total_above,
                total_below,
                total_active,
                total_inactive,
                active_from_total_above,
                inactive_from_total_above,
                active_from_total_below,
                inactive_from_total_below,
                above_from_total_active,
                below_from_total_active,
                above_from_total_inactive,
                below_from_total_inactive,
            ]

    # write dataframe from dictionary
    # TODO: write one columns "count_activity_from_total_above" -> with all values active and inactive and a new column "activity_from_total_above" with binary options active or inactive
    activity_beta_threshold_df = pd.DataFrame(activity_beta_threshold_dict)
    activity_beta_threshold_df.rename(
        index={
            0: "subject_hemisphere",
            1: "session",
            2: f"total_above_{beta_threshold}",
            3: f"total_below_{beta_threshold}",
            4: "total_active",
            5: "total_inactive",
            6: "active_from_total_above",
            7: "inactive_from_total_above",
            8: "active_from_total_below",
            9: "inactive_from_total_below",
            10: "above_from_total_active",
            11: "below_from_total_active",
            12: "above_from_total_inactive",
            13: "below_from_total_inactive",
        },
        inplace=True,
    )
    activity_beta_threshold_df = activity_beta_threshold_df.transpose()

    return {"activity_beta_threshold_df": activity_beta_threshold_df, "contingency_dict": contingency_dict}


def fooof_mono_beta_threshold_plot(similarity_calculation: str, beta_threshold: float, data_to_plot: str):
    """
    Input:
        - similarity_calculation: "inverse_distance", "exp_neg_distance"
        - beta_threshold: e.g. 0.7
        - data_to_plot: e.g. "activity_from_total_above", "threshold_from_total_active", "activity_from_total_above",
        "activity_from_total_below", "threshold_from_total_inactive"

    """

    # load data from fooof_mono_beta_count_active_and_above_threshold()
    data_to_analyze = fooof_mono_beta_count_active_and_above_threshold(
        similarity_calculation=similarity_calculation, beta_threshold=beta_threshold
    )

    data_to_analyze = data_to_analyze["activity_beta_threshold_df"]

    # replace session strings by integers
    data_to_analyze = data_to_analyze.replace(to_replace=["fu3m", "fu12m", "fu18m"], value=[3, 12, 18])

    ##################### PERFORM STATISTICAL TEST  #####################

    # ses_groups= ["fu3m_group1", "fu3m_group2", "fu12m_group1", "fu12m_group2", "fu18m_group1", "fu18m_group2"]
    # ses_clinical_activity_stats_test= [("fu3m_active", "fu3m_inactive"), ("fu12m_active", "fu12m_inactive"), ("fu18m_active", "fu18m_inactive")]
    sessions = [3, 12, 18]

    all_results_statistics = []
    describe_array_group1 = {}
    describe_array_group2 = {}

    for ses in sessions:
        ses_df = data_to_analyze.loc[data_to_analyze.session == ses]

        if (
            data_to_plot == "activity_from_total_above"
        ):  # how many from all contacts above threshold were clinically active?
            first_array = np.array(ses_df.active_from_total_above.values.astype(float))
            group_1 = "active"
            second_array = np.array(ses_df.inactive_from_total_above.values.astype(float))
            group_2 = "inactive"

        elif data_to_plot == "threshold_from_total_active":  # how many from all active contacts were above threshold?
            first_array = np.array(ses_df.above_from_total_active.values.astype(float))
            group_1 = "above"
            second_array = np.array(ses_df.below_from_total_active.values.astype(float))
            group_2 = "below"

        # get sample size of each pair
        sample_size = len(first_array)

        description_group_1 = scipy.stats.describe(first_array)
        description_group_2 = scipy.stats.describe(second_array)

        describe_array_group1[f"{ses}_{data_to_plot}"] = description_group_1
        describe_array_group2[f"{ses}_{data_to_plot}"] = description_group_2

        # Perform Wilcoxon Test, same sample size in both groups are the same
        results_stats = pg.wilcoxon(
            first_array, second_array
        )  # pair is always a tuple, comparing first and second component of this tuple

        results_stats[f'comparison_{data_to_plot}'] = '_'.join(str(ses))  # new column "comparison" with the session
        results_stats["sample_size"] = sample_size

        all_results_statistics.append(results_stats)

    significance_results = pd.concat(all_results_statistics)

    description_data_group1 = pd.DataFrame(describe_array_group1)
    description_data_group1.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_data_group1 = description_data_group1.transpose()
    description_data_group1_copy = description_data_group1.copy()
    description_data_group1_copy["group"] = group_1

    description_data_group2 = pd.DataFrame(describe_array_group2)
    description_data_group2.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_data_group2 = description_data_group2.transpose()
    description_data_group2_copy = description_data_group2.copy()
    description_data_group2_copy["group"] = group_2

    ##################### STORE RESULTS IN DICTIONARY AND SAVE #####################

    results_dictionary = {
        "significance_results": significance_results,
        "description_results_group1": description_data_group1_copy,
        "description_results_group2": description_data_group2_copy,
    }

    # save as pickle
    results_filepath = os.path.join(
        results_path,
        f"fooof_beta_clinical_activity_statistics_beta_{data_to_plot}_{beta_threshold}_{similarity_calculation}.pickle",
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dictionary, file)

    ##################### PLOT VIOLINPLOT OF relative PSD to rank 1 OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    # fig=plt.figure()
    fig, axes = plt.subplots(1, 1, figsize=(15, 12))
    # fontdict = {"size": 25}
    # ax = fig.add_subplot()

    if data_to_plot == "activity_from_total_above":
        y_label = "active_from_total_above"

        sns.violinplot(
            data=data_to_analyze,
            x="session",
            y="active_from_total_above",
            # hue="session_clinical_activity",
            palette="coolwarm",
            inner="box",
            ax=axes,
            scale="count",
            # scale_hue=True,
            dodge=True,
        )  # scale="count" will scales the width of violins depending on their observations

        # statistical test
        # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
        # pairs = list(combinations(ses_clinicalUse, 2))

        # annotator = Annotator(axes, pairs, data=data_to_analyze, x='session_clinical_activity', y=y_values)

        # if single_contacts_or_average == "single_contacts":
        #     annotator.configure(test='Mann-Whitney', text_format='star')

        # if single_contacts_or_average == "electrode_average":
        #     annotator.configure(test='Wilcoxon', text_format='star')

        # annotator.apply_and_annotate()

        # problem: hue=subject_hemisphere plots all single contacts per subject, but doesn´t respect groups of clinically active vs inactive..

        sns.stripplot(
            data=data_to_analyze,
            x="session",
            y="active_from_total_above",
            # hue="session_clinical_activity",
            ax=axes,
            size=7,
            color="grey",  # palette = "tab20c", "mako", "viridis", "cubehelix", "rocket_r", "vlag", "coolwarm"
            alpha=0.5,  # Transparency of dots
            dodge=True,  # datapoints of groups active, inactive are plotted next to each other
        )

    elif data_to_plot == "threshold_from_total_active":
        y_label = "above_from_total_active"

        sns.violinplot(
            data=data_to_analyze,
            x="session",
            y="above_from_total_active",
            # hue="session",
            palette="coolwarm",
            inner="box",
            ax=axes,
            scale="count",
            scale_hue=True,
            dodge=True,
        )  # scale="count" will scales the width of violins depending on their observations

        # statistical test
        # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
        # pairs = list(combinations(ses_clinicalUse, 2))

        # annotator = Annotator(axes, pairs, data=data_to_analyze, x='session_clinical_activity', y=y_values)

        # if single_contacts_or_average == "single_contacts":
        #     annotator.configure(test='Mann-Whitney', text_format='star')

        # if single_contacts_or_average == "electrode_average":
        #     annotator.configure(test='Wilcoxon', text_format='star')

        # annotator.apply_and_annotate()

        # problem: hue=subject_hemisphere plots all single contacts per subject, but doesn´t respect groups of clinically active vs inactive..

        sns.stripplot(
            data=data_to_analyze,
            x="session",
            y="above_from_total_active",
            # hue="session_clinical_activity",
            ax=axes,
            size=5,
            color="grey",  # palette = "tab20c", "mako", "viridis", "cubehelix", "rocket_r", "vlag", "coolwarm"
            alpha=0.5,  # Transparency of dots
            dodge=True,  # datapoints of groups active, inactive are plotted next to each other
        )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    fig.suptitle(data_to_plot, fontsize=25)
    axes.set_ylabel(y_label, fontsize=20)
    axes.set_xlabel("months post-surgery", fontsize=20)
    axes.tick_params(axis="x", labelsize=20)
    axes.tick_params(axis="y", labelsize=20)
    # plt.ylim(y_lim)
    fig.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            figures_path,
            f"fooof_beta_clinical_activity_beta_{data_to_plot}_{beta_threshold}_{similarity_calculation}.png",
        ),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            figures_path,
            f"fooof_beta_clinical_activity_beta_{data_to_plot}_{beta_threshold}_{similarity_calculation}.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    print(
        "new files: ",
        f"fooof_beta_clinical_activity_statistics_beta_{data_to_plot}_{beta_threshold}_{similarity_calculation}.pickle",
        "\nwritten in in: ",
        results_path,
        f"\nnew figures: fooof_beta_clinical_activity_beta_{data_to_plot}_{beta_threshold}_{similarity_calculation}.png",
        f"\nand fooof_beta_clinical_activity_beta_{data_to_plot}_{beta_threshold}_{similarity_calculation}.svg"
        "\nwritten in: ",
        figures_path,
    )

    return {
        "significance_results": significance_results,
        "description_data_group1": description_data_group1_copy,
        "description_data_group2": description_data_group2_copy,
    }


def fooof_mono_beta_and_clinical_activity_statistical_test(
    single_contacts_or_average: str, feature: str, similarity_calculation: str, fooof_version: str
):
    """

    Input:
        - single_contacts_or_average: "single_contacts" or "electrode_average"
        - feature: "rank", "raw_beta_power", "rel_beta_power_to_rank_1", "rel_beta_power_range_0_to_1"
        - similarity_calculation: "inverse_distance", "exp_neg_distance"





    """

    beta_and_clinical_activity_data = fooof_mono_beta_and_clinical_activity_write_dataframes(
        fooof_version=fooof_version
    )

    if single_contacts_or_average == "single_contacts":
        data_to_analyze = beta_and_clinical_activity_data["single_contacts"]

        if feature == "rank":
            y_values = "rank_8"
            y_label = "beta power rank per electrode"
            title = "beta rank of clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

        if feature == "raw_beta_power":
            y_values = "estimated_monopolar_beta_psd"
            y_label = "beta power"
            title = "beta power of clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

        if feature == "rel_beta_power_to_rank_1":
            y_values = "beta_psd_rel_to_rank1"
            y_label = "beta power relative to highest beta per electrode"
            title = "beta power normalized to highest beta \nof clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

        if feature == "rel_beta_power_range_0_to_1":
            y_values = "beta_psd_rel_range_0_to_1"
            y_label = "beta power ranging from 0 to 1 per electrode"
            title = "beta power normalized to lowest and highest beta \nof clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

    elif single_contacts_or_average == "electrode_average":
        data_to_analyze = beta_and_clinical_activity_data["electrode_average"]

        if feature == "rank":
            y_values = "electrode_mean_beta_rank"
            y_label = "beta power rank, averaged per group"
            title = "beta rank of clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

        if feature == "raw_beta_power":
            y_values = "electrode_mean_beta_psd"
            y_label = "beta power, averaged per group"
            title = "beta power of clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

        if feature == "rel_beta_power_to_rank_1":
            y_values = "electrode_mean_beta_psd_rel_to_rank1"
            y_label = "beta power relative to highest beta per electrode \naveraged per group"
            title = "beta power normalized to highest beta \nof clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

        if feature == "rel_beta_power_range_0_to_1":
            y_values = "electrode_mean_beta_psd_rel_range_0_to_1"
            y_label = "beta power ranging from 0 to 1 per electrode \naveraged per group"
            title = "beta power normalized to lowest and highest beta \nof clinically active vs. inactive stimulation contacts"
            # y_lim = 0, 1.25

    # sanity check: let me know where there are NaN values and drop these rows
    # nan_rows =

    ##################### PERFORM STATISTICAL TEST  #####################

    ses_clinical_activity = [
        "fu3m_active",
        "fu3m_inactive",
        "fu12m_active",
        "fu12m_inactive",
        "fu18or24m_active",
        "fu18or24m_inactive",
    ]
    ses_clinical_activity_stats_test = [
        ("fu3m_active", "fu3m_inactive"),
        ("fu12m_active", "fu12m_inactive"),
        ("fu18or24m_active", "fu18or24m_inactive"),
    ]
    pairs = list(combinations(ses_clinical_activity, 2))
    all_results_statistics = []
    describe_arrays = {}
    mean_and_std = {}

    # pair = tuple e.g. fu3m_active, fu3m_inactive
    # for pair in pairs:
    for ses_activity in ses_clinical_activity_stats_test:
        first_in_pair = data_to_analyze.loc[(data_to_analyze.session_clinical_activity == ses_activity[0])]
        second_in_pair = data_to_analyze.loc[(data_to_analyze.session_clinical_activity == ses_activity[1])]

        if feature == "rank":
            if single_contacts_or_average == "single_contacts":
                first_in_pair = np.array(first_in_pair.rank_8.values)
                second_in_pair = np.array(second_in_pair.rank_8.values)

            elif single_contacts_or_average == "electrode_average":
                first_in_pair = np.array(first_in_pair.electrode_mean_beta_rank.values)
                second_in_pair = np.array(second_in_pair.electrode_mean_beta_rank.values)

        elif feature == "raw_beta_power":
            if single_contacts_or_average == "single_contacts":
                first_in_pair = np.array(first_in_pair.estimated_monopolar_beta_psd.values)
                second_in_pair = np.array(second_in_pair.estimated_monopolar_beta_psd.values)

            elif single_contacts_or_average == "electrode_average":
                first_in_pair = np.array(first_in_pair.electrode_mean_beta_psd.values)
                second_in_pair = np.array(second_in_pair.electrode_mean_beta_psd.values)

        elif feature == "rel_beta_power_to_rank_1":
            if single_contacts_or_average == "single_contacts":
                first_in_pair = np.array(first_in_pair.beta_psd_rel_to_rank1.values)
                second_in_pair = np.array(second_in_pair.beta_psd_rel_to_rank1.values)

            elif single_contacts_or_average == "electrode_average":
                first_in_pair = np.array(first_in_pair.electrode_mean_beta_psd_rel_to_rank1.values)
                second_in_pair = np.array(second_in_pair.electrode_mean_beta_psd_rel_to_rank1.values)

        elif feature == "rel_beta_power_range_0_to_1":
            if single_contacts_or_average == "single_contacts":
                first_in_pair = np.array(first_in_pair.beta_psd_rel_range_0_to_1.values)
                second_in_pair = np.array(second_in_pair.beta_psd_rel_range_0_to_1.values)

            elif single_contacts_or_average == "electrode_average":
                first_in_pair = np.array(first_in_pair.electrode_mean_beta_psd_rel_range_0_to_1.values)
                second_in_pair = np.array(second_in_pair.electrode_mean_beta_psd_rel_range_0_to_1.values)

        # get sample size of each pair
        sample_size = len(first_in_pair)

        # Perform Wilcoxon Test, only if same sample size in active and inactive groups -> only when averaged
        if single_contacts_or_average == "electrode_average":
            results_stats = pg.wilcoxon(
                first_in_pair, second_in_pair
            )  # pair is always a tuple, comparing first and second component of this tuple

        elif single_contacts_or_average == "single_contacts":
            results_stats = pg.mwu(
                first_in_pair, second_in_pair
            )  # pair is always a tuple, comparing first and second component of this tuple

        results_stats[f'comparison_{single_contacts_or_average}_{feature}'] = '_'.join(
            ses_activity
        )  # new column "comparison" with the pair being compared e.g. fu3m_active and fu3m_inactive
        results_stats["sample_size_one_session_activity_group"] = sample_size

        all_results_statistics.append(results_stats)

    significance_results = pd.concat(all_results_statistics)

    ##################### GET STATISTICAL IMPORTANT FEATURES #####################
    # describe all 6 groups
    for s_c in ses_clinical_activity:
        # get array of each group
        group = data_to_analyze.loc[(data_to_analyze.session_clinical_activity == s_c)]

        if single_contacts_or_average == "single_contacts":
            if feature == "rank":
                group = np.array(group.rank_8.values)

            elif feature == "raw_beta_power":
                group = np.array(group.estimated_monopolar_beta_psd.values)

            elif feature == "rel_beta_power_to_rank_1":
                group = np.array(group.beta_psd_rel_to_rank1.values)

            elif feature == "rel_beta_power_range_0_to_1":
                group = np.array(group.beta_psd_rel_range_0_to_1.values)

        elif single_contacts_or_average == "electrode_average":
            if feature == "rank":
                group = np.array(group.electrode_mean_beta_rank.values)

            elif feature == "raw_beta_power":
                group = np.array(group.electrode_mean_beta_psd.values)

            elif feature == "rel_beta_power_to_rank_1":
                group = np.array(group.electrode_mean_beta_psd_rel_to_rank1.values)

            elif feature == "rel_beta_power_range_0_to_1":
                group = np.array(group.electrode_mean_beta_psd_rel_range_0_to_1.values)

        standard_deviation = np.std(group)

        mean_and_std[f"{s_c}_{single_contacts_or_average}_{feature}"] = [standard_deviation]

        description = scipy.stats.describe(group)

        describe_arrays[f"{s_c}_{single_contacts_or_average}_{feature}"] = description

    description_results = pd.DataFrame(describe_arrays)
    description_results.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_results = description_results.transpose()

    mean_std = pd.DataFrame(mean_and_std)
    mean_std.rename(index={0: "standard_deviation", 1: "mean"}, inplace=True)
    mean_std = mean_std.transpose()

    description_all = pd.concat([description_results, mean_std])

    ##################### STORE RESULTS IN DICTIONARY AND SAVE #####################

    results_dictionary = {"significance_results": significance_results, "description_results": description_results}

    # save as pickle
    results_filepath = os.path.join(
        results_path,
        f"fooof_beta_clinical_activity_statistics_{feature}_{single_contacts_or_average}_{similarity_calculation}_{fooof_version}.pickle",
    )
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dictionary, file)

    ##################### PLOT VIOLINPLOT OF relative PSD to rank 1 OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    # set seaborn style:
    sns.set_style("whitegrid", {"axes.grid": True, "axes.grid.axis": "y"})

    # fig=plt.figure()
    fig, axes = plt.subplots(1, 1, figsize=(15, 12))
    # fontdict = {"size": 25}
    # ax = fig.add_subplot()

    # sns.violinplot(data=data_MonoBeta8Ranks, x="session_clinicalUse", y=y_values, hue="clinicalUse", palette="Set2", inner="box", ax=ax)
    sns.violinplot(
        data=data_to_analyze,
        x="session",
        y=y_values,
        hue="session_clinical_activity",
        color="white",  # palette="coolwarm"
        # split=True, # delete
        inner="box",  # alternative: quart
        ax=axes,
        scale="count",
        scale_hue=True,
        dodge=True,
    )  # scale="count" will scales the width of violins depending on their observations

    # statistical test
    # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
    # pairs = list(combinations(ses_clinicalUse, 2))

    # annotator = Annotator(axes, pairs, data=data_to_analyze, x='session_clinical_activity', y=y_values)

    # if single_contacts_or_average == "single_contacts":
    #     annotator.configure(test='Mann-Whitney', text_format='star')

    # if single_contacts_or_average == "electrode_average":
    #     annotator.configure(test='Wilcoxon', text_format='star')

    # annotator.apply_and_annotate()

    # problem: hue=subject_hemisphere plots all single contacts per subject, but doesn´t respect groups of clinically active vs inactive..
    sns.stripplot(
        data=data_to_analyze,
        x="session",
        y=y_values,
        hue="session_clinical_activity",
        ax=axes,
        jitter=True,  # delete
        size=5,
        color="grey",  # palette = "tab20c", "mako", "viridis", "cubehelix", "rocket_r", "vlag", "coolwarm"
        alpha=0.5,  # Transparency of dots
        dodge=True,  # datapoints of groups active, inactive are plotted next to each other
    )

    sns.despine(left=True, bottom=True)  # get rid of figure frame

    fig.suptitle(title, fontsize=15)
    axes.set_ylabel(y_label, fontsize=10)
    axes.set_xlabel("months post-surgery", fontsize=10)
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    # plt.ylim(y_lim)
    # axes.grid(axis="y")
    fig.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            figures_path,
            f"fooof_beta_clinical_activity_{feature}_{single_contacts_or_average}_{similarity_calculation}_{fooof_version}.png",
        ),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            figures_path,
            f"fooof_beta_clinical_activity_{feature}_{single_contacts_or_average}_{similarity_calculation}_{fooof_version}.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    print(
        "new files: ",
        f"fooof_beta_clinical_activity_statistics_{feature}_{single_contacts_or_average}_{similarity_calculation}_{fooof_version}.pickle",
        "\nwritten in in: ",
        results_path,
        f"\nnew figures: fooof_beta_clinical_activity_{feature}_{single_contacts_or_average}_{similarity_calculation}_{fooof_version}.png",
        f"\nand fooof_beta_clinical_activity_{feature}_{single_contacts_or_average}_{similarity_calculation}_{fooof_version}.svg"
        "\nwritten in: ",
        figures_path,
    )

    return {
        "data_to_analyze": data_to_analyze,
        "results_dictionary": results_dictionary,
        "description_all": description_all,
    }
