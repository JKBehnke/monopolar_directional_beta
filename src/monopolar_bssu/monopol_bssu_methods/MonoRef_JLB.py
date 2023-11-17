""" Monopolar referencing: Johannes Busch method """

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
from ..utils import io_percept as io_percept
from ..utils import percept_lfp_preprocessing as percept_lfp_preprocessing
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..utils import io_externalized as io_externalized


# import analysis.loadResults as loadcsv


def MonoRef_JLB(incl_sub: str, hemisphere: str, normalization: str, incl_session: list):
    """
    Calculate the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C)

    Input:
        - incl_sub: str, e.g. "024"
        - hemisphere: str, e.g. "Right"
        - normalization: str, e.g. "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - incl_session: list, ["postop", "fu3m", "fu12m", "fu18m"]


    Four different versions of PSD:
        - raw PSD
        - relative PSD to total sum
        - relative PSD to sum 1-100 Hz
        - relative PSD to sum 40-80 Hz

    1) Calculate the percentage of each direction A, B and C:
        - proxy of direction:
            A = 1A2A
            B = 1B2B
            C = 1C2C

        - Percentage of direction = Mean beta power of one direction divided by total mean beta power of all directions

    2) Weight each segmented level 1 and 2 with percentage of direction:
        - proxy of hight:
            1 = 02
            2 = 13

        - Percentage of direction multiplied with mean beta power of each level
        - e.g. 1A = Percentage of direction(A) * mean beta power (02)



     Selecting the monopolar referenced contact with #1 Rank (postop #1 or fu3m #1)

        1) restructure the monopolar Rank and References Dataframes
            - select the monopolar contact with Rank #1 in every session_freqBand column of the monoRankDF
            - store every Rank #1 monopolar contact in a dictionary monopolarFirstRank, transform to DF
            - for every #1 ranked channel in every session and for every frequency band, add all monopolarly referenced PSD values
            - FirstRankChannel_PSD_DF with columns: "session_frequencyBand", "numberOneRank_monopolarChannel", "monoRef_postop_lowBeta", "monoRef_postop_highBeta" etc.

        2) For postop and fu3m baselineRank Channel choose highest ranked channel of each frequency band with corresponding averaged psd values of the same frequency band
            - BetaPsdAverage
            - LowBetaPsdAverage
            - HighBetaPsdAverage

            (e.g. #1 channel postop in lowBeta band = 1A, get all monopolarly averaged PSD values from channel 1A in lowBeta band at all timepoints)


    Return:
        {
        "psdAverageDataframe":psdAverageDF,
        "psdPercentagePerDirection":psdPercentageDF,
        "monopolarReference":monopolRefDF,
        "monopolarRanks":monopolRankDF,
        "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
        "PostopBaseline_beta": postopBaseline_beta,
        "PostopBaseline_lowBeta": postopBaseline_lowBeta,
        "PostopBaseline_highBeta": postopBaseline_highBeta,
        "Fu3mBaseline_beta": fu3mBaseline_beta,
        "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
        "Fu3mBaseline_highBeta": fu3mBaseline_highBeta
        }


    Pitfalls:
        - Error if a patient only had 3 timepoints, especially when postop or fu3m recording is missing
        - so check beforehand how many timepoints the patient went through and adjust the code manually


    """

    time_points = incl_session
    frequency_range = ["lowBeta", "highBeta", "beta"]
    direction_proxy = ["1A2A", "1B2B", "1C2C"]
    level_proxy = ["13", "02"]
    averagedPSD_dict = {}
    percentagePSD_dict = {}
    monopolar_references = {}
    ############# READ CSV FILE WITH AVERAGED PSD as Dataframe #############
    # get path to .csv file
    results_path = find_folders.get_local_path(folder="results", sub=incl_sub)

    # read .csv file as Dataframe
    # psdAverageDataframe = pd.read_csv(os.path.join(results_path, f"SPECTROGRAMpsdAverageFrequencyBands_{normalization}_{hemisphere}"))
    psdAverageDataframe = loadResults.load_PSDjson(
        sub=incl_sub,
        result="PSDaverageFrequencyBands",  # self.result has to be a list, because the loading function is
        hemisphere=hemisphere,
        filter="band-pass",
    )

    # transform dictionary to Dataframe
    psdAverageDataframe = pd.DataFrame(psdAverageDataframe)

    # select the correct normalization
    psdAverageDataframe = psdAverageDataframe[
        psdAverageDataframe.absoluteOrRelativePSD == normalization
    ]

    for tp in time_points:
        # check if timepoint exists

        for f, fq in enumerate(frequency_range):
            # filter the Dataframe to only get rows within different frequency bands of each session
            session_frequency_Dataframe = psdAverageDataframe[
                (psdAverageDataframe["frequencyBand"] == fq)
                & (psdAverageDataframe["session"] == tp)
            ]

            ################### WEIGHT DIRECTION ###################
            # get all relevant averaged psd values for each direction
            for dir in direction_proxy:
                # get the row that contains the the bipolar channel of interest "1A2A", "1B2B", "1C2C"
                directionDF = session_frequency_Dataframe[
                    session_frequency_Dataframe["bipolarChannel"].str.contains(dir)
                ]

                # store these 3 averaged psd values (in fifth column pf the initial DF) in the dictionary
                # averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{dir}"] = [tp, fq, dir, directionDF.iloc[:,4].item()]
                averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{dir}"] = [
                    tp,
                    fq,
                    dir,
                    directionDF.iloc[:, 4].item(),
                ]

            # calculate total mean beta power of all directions: sum of averaged PSD 1A2A + 1B2B + 1C2C
            averagedPsd_A = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_1A2A"][3]
            averagedPsd_B = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_1B2B"][3]
            averagedPsd_C = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_1C2C"][3]

            sumABC = averagedPsd_A + averagedPsd_B + averagedPsd_C

            # calculate the percentage of each direction of the total mean beta power of all directions
            for dir in direction_proxy:
                percentagePSD = (
                    averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{dir}"][3] / sumABC
                )

                percentagePSD_dict[f"percentagePSD_{tp}_{fq}_{dir}"] = [
                    tp,
                    fq,
                    dir,
                    percentagePSD,
                ]

            percentagePSD_A = percentagePSD_dict[f"percentagePSD_{tp}_{fq}_1A2A"][3]
            percentagePSD_B = percentagePSD_dict[f"percentagePSD_{tp}_{fq}_1B2B"][3]
            percentagePSD_C = percentagePSD_dict[f"percentagePSD_{tp}_{fq}_1C2C"][3]

            ################### WEIGHT LEVEL ###################

            # get both relevant averaged PSD values for the levels 1 and 2
            for lev in level_proxy:
                # get the row that contains the the bipolar channels of interest "02", "13"
                levelDF = session_frequency_Dataframe[
                    session_frequency_Dataframe["bipolarChannel"].str.contains(lev)
                ]

                # store these 2 averaged psd values (in fifth column) in the dictionary
                averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{lev}"] = [
                    tp,
                    fq,
                    lev,
                    levelDF.iloc[:, 4].item(),
                ]

            # get averaged PSD values for both levels
            averagedPsd_level1 = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_02"][3]
            averagedPsd_level2 = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_13"][3]

            # calculate the monopolar reference for each segmented contact
            monopol_1A = percentagePSD_A * averagedPsd_level1
            monopol_1B = percentagePSD_B * averagedPsd_level1
            monopol_1C = percentagePSD_C * averagedPsd_level1
            monopol_2A = percentagePSD_A * averagedPsd_level2
            monopol_2B = percentagePSD_B * averagedPsd_level2
            monopol_2C = percentagePSD_C * averagedPsd_level2

            # store monopolar references in a dictionary
            monopolar_references[f"monoRef_{tp}_{fq}"] = [
                monopol_1A,
                monopol_1B,
                monopol_1C,
                monopol_2A,
                monopol_2B,
                monopol_2C,
            ]

    #################### WRITE DATAFRAMES TO STORE VALUES ####################

    # write DataFrame of averaged psd values in each frequency band depending on the chosen normalization
    psdAverageDF = pd.DataFrame(averagedPSD_dict)
    psdAverageDF.rename(
        index={0: "session", 1: "frequency_band", 2: "channel", 3: "averagedPSD"},
        inplace=True,
    )  # rename the rows
    psdAverageDF = (
        psdAverageDF.transpose()
    )  # Dataframe with 1 columns and rows for each single power spectrum

    # write DataFrame of percentage psd values in each frequency band depending on the chosen normalization
    psdPercentageDF = pd.DataFrame(percentagePSD_dict)
    psdPercentageDF.rename(
        index={
            0: "session",
            1: "frequency_band",
            2: "direction",
            3: "percentagePSD_perDirection",
        },
        inplace=True,
    )  # rename the rows
    psdPercentageDF = (
        psdPercentageDF.transpose()
    )  # Dataframe with 1 columns and rows for each single power spectrum

    # write DataFrame of monopolar reference values in each frequency band and session timepoint
    monopolRefDF = pd.DataFrame(monopolar_references)
    monopolRefDF.rename(
        index={
            0: "monopolarRef_1A",
            1: "monopolarRef_1B",
            2: "monopolarRef_1C",
            3: "monopolarRef_2A",
            4: "monopolarRef_2B",
            5: "monopolarRef_2C",
        },
        inplace=True,
    )  # rename the rows

    # write DataFrame of ranks of monopolar references in each frequency band and session timepoint
    monopolRankDF = monopolRefDF.rank(
        ascending=False
    )  # new Dataframe ranking monopolar values from monopolRefDF from high to low

    MonoRef_JLB_result = {
        "BIP_psdAverage": psdAverageDF,
        "BIP_directionalPercentage": psdPercentageDF,
        "monopolar_psdAverage": monopolRefDF,
        "monopolar_psdRank": monopolRankDF,
    }
    # # save Dataframes as csv in the results folder
    # psdAverageDF.to_csv(os.path.join(results_path,f"averagedPSD_{normalization}_{hemisphere}"), sep=",")
    # psdPercentageDF.to_csv(os.path.join(results_path,f"percentagePsdDirection_{normalization}_{hemisphere}"), sep=",")
    # monopolRefDF.to_csv(os.path.join(results_path,f"monopolarReference_{normalization}_{hemisphere}"), sep=",")
    # monopolRankDF.to_csv(os.path.join(results_path,f"monopolarRanks_{normalization}_{hemisphere}"), sep=",")

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    MonoRef_JLB_result_filepath = os.path.join(
        results_path,
        f"sub{incl_sub}_{hemisphere}_MonoRef_JLB_result_{normalization}_band-pass.pickle",
    )
    with open(MonoRef_JLB_result_filepath, "wb") as file:
        pickle.dump(MonoRef_JLB_result, file)

    print(
        f"New file: sub{incl_sub}_{hemisphere}_MonoRef_JLB_result_{normalization}_band-pass.pickle",
        f"\nwritten in: {results_path}",
    )

    return MonoRef_JLB_result

    # ################ Restructure the Dataframes ################
    # # Replace every rank #1 in monoRankDF with the monopolar channel (in index)
    # monoRank_replace1DF = monopolRankDF.apply(lambda x: x.where(x != 1.0, monopolRankDF.index), axis=0)

    # # drop first column "Unnamed: 0" with all monopolar Ref channel names
    # monoRank_replace1DF.drop(columns=monoRank_replace1DF.columns[0], axis=1,  inplace=True)

    # # only select the strings values with the monopolar channel #1 rank for each column (session_frequencyBand)
    # monopolarFirstRank = {}

    # # loop over each column and only keep the value with the monopolar channel of the 1st ranked PSD
    # for col in monoRank_replace1DF.columns:
    #     # extract the column as a series
    #     column = monoRank_replace1DF[col]

    #     # exclude float values and replace floats by nan
    #     # lambda function returns the value if it is a string, otherwise it returns np.nan
    #     column = column.apply(lambda x: x if isinstance(x, str) else np.nan)

    #     # drop all NaN values
    #     column.dropna(how='all', inplace=True)

    #     # find the first value that is a string (e.g. "monopolarRef_1A")
    #     #The next function returns the first value of each column that is a string. If the sequence is empty, the default value None is returned.
    #     value = next((value for value in column.values if isinstance(value, str)), None)

    #     # add the result to the dictionary
    #     monopolarFirstRank[col] = value

    # # convert the dictionary to a dataframe
    # monopolarFirstRankDF = pd.DataFrame(list(monopolarFirstRank.items()), columns=['session_frequencyBand', 'numberOneRank_monopolarChannel'])

    # # from monoRefDF extract only the row equal to the value of the column 'numberOneRank_monopolarChannel'

    # # loop through each #1 rank value and store the matching dataframe row from monoRefDF in a dictionary
    # FirstRankRef_dict = {}

    # for index, value in monopolarFirstRankDF['numberOneRank_monopolarChannel'].iteritems():
    #     FirstRankRef_dict[f"{index}_{value}"] = monopolRefDF[monopolRefDF.index.str.contains(value)]

    # # first make a new Dataframe of selected monoRef rows, by concatenating all values of FirstRankRef_dict
    # FirstRankRef_DF = pd.concat(FirstRankRef_dict.values(), keys=FirstRankRef_dict.keys(), ignore_index=True) # keys need to be specified

    # # drop the first column with monopolar channel names, because this column already exists in the monopolarFirstRank DF that will be concatenated
    # FirstRankRef_DF.drop(columns=FirstRankRef_DF.columns[0], axis=1, inplace=True) # inplace=True will modify the original DF and will not create a new DF

    # # now concatenate the FirstRankDF (with #1 ranked monopolar contacts) with the FirstRankRefDF (with all referenced psd values of this #1 ranked contact)
    # FirstRankChannel_PSD_DF = pd.concat([monopolarFirstRankDF, FirstRankRef_DF], axis=1)

    # ################ SELECT HIGHEST RANK FOR POSTOP AND FU3M BASELINE ################

    # # Dataframes for postop Baseline first ranks in each frequency band
    # postop_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("postop")
    # fu3m_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("fu3m")
    # beta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_beta")
    # lowBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_lowBeta")
    # highBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_highBeta")

    # #######    POSTOP BASELINE  #######
    # # first select the row from the highest ranked channel in each frequency band
    # postopBaseline_beta_channelRow = FirstRankChannel_PSD_DF[postop_mask & beta_mask]
    # postopBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[postop_mask & lowBeta_mask]
    # postopBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[postop_mask & highBeta_mask]

    # # # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    # postopBaseline_beta = postopBaseline_beta_channelRow.loc[:,postopBaseline_beta_channelRow.columns.str.contains("_beta")]
    # postopBaseline_lowBeta = postopBaseline_lowBeta_channelRow.loc[:,postopBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")]
    # postopBaseline_highBeta = postopBaseline_highBeta_channelRow.loc[:,postopBaseline_highBeta_channelRow.columns.str.contains("_highBeta")]

    # #######    FU3M BASELINE  #######
    #  # first select the row from the highest ranked channel in each frequency band
    # fu3mBaseline_beta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & beta_mask]
    # fu3mBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & lowBeta_mask]
    # fu3mBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & highBeta_mask]

    # # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    # fu3mBaseline_beta = fu3mBaseline_beta_channelRow.loc[:,fu3mBaseline_beta_channelRow.columns.str.contains("_beta")]
    # fu3mBaseline_lowBeta = fu3mBaseline_lowBeta_channelRow.loc[:,fu3mBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")]
    # fu3mBaseline_highBeta = fu3mBaseline_highBeta_channelRow.loc[:,fu3mBaseline_highBeta_channelRow.columns.str.contains("_highBeta")]

    # # save Dataframes as csv in the results folder
    # FirstRankChannel_PSD_DF.to_csv(os.path.join(results_path,f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"), sep=",")
    # postopBaseline_beta.to_csv(os.path.join(results_path,f"postopBaseline_beta{normalization}_{hemisphere}"), sep=",")
    # postopBaseline_lowBeta.to_csv(os.path.join(results_path,f"postopBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
    # postopBaseline_highBeta.to_csv(os.path.join(results_path,f"postopBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
    # fu3mBaseline_beta.to_csv(os.path.join(results_path,f"fu3mBaseline_beta{normalization}_{hemisphere}"), sep=",")
    # fu3mBaseline_lowBeta.to_csv(os.path.join(results_path,f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
    # fu3mBaseline_highBeta.to_csv(os.path.join(results_path,f"fu3mBaseline_highBeta{normalization}_{hemisphere}"), sep=",")

    # return {
    #     "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
    #     "PostopBaseline_beta": postopBaseline_beta,
    #     "PostopBaseline_lowBeta": postopBaseline_lowBeta,
    #     "PostopBaseline_highBeta": postopBaseline_highBeta,
    #     "Fu3mBaseline_beta": fu3mBaseline_beta,
    #     "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
    #     "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,
    # }


def MonoRefPsd_highestRank(sub: str, normalization: str, hemisphere: str):
    """
    Selecting the monopolar referenced contact with #1 Rank (postop #1 or fu3m #1)

        - sub: str,
        - normalization: str,
        - hemisphere: str,

        1) Load and restructure the monopolar Rank and References Dataframes
            - select the monopolar contact with Rank #1 in every session_freqBand column of the monoRankDF
            - store every Rank #1 monopolar contact in a dictionary monopolarFirstRank, transform to DF
            - for every #1 ranked channel in every session and for every frequency band, add all monopolarly referenced PSD values
            - FirstRankChannel_PSD_DF with columns: "session_frequencyBand", "numberOneRank_monopolarChannel", "monoRef_postop_lowBeta", "monoRef_postop_highBeta" etc.

        2) For postop and fu3m baselineRank Channel choose highest ranked channel of each frequency band with corresponding averaged psd values of the same frequency band
            - BetaPsdAverage
            - LowBetaPsdAverage
            - HighBetaPsdAverage

            (e.g. #1 channel postop in lowBeta band = 1A, get all monopolarly averaged PSD values from channel 1A in lowBeta band at all timepoints)


    """

    ############# READ CSV FILE of monopolar referenced AVERAGED PSD as Dataframe #############
    # get path to results folder of subject
    results_path = findfolders.get_local_path(folder="results", sub=sub)

    # read .csv file as Dataframe
    monoRef_result = loadcsv.load_MonoRef_JLBresultCSV(
        sub=sub, normalization=normalization, hemisphere=hemisphere
    )

    # get Dataframe of monopolar references/ monopolarly referenced PSD values
    monoRefDF = monoRef_result["monopolRefDF"]

    # get Dataframe of monopolar ranks (column "Unnamed: 0" with monopolar contact names n=6)
    monoRankDF = monoRef_result["monopolRankDF"]

    ################ Restructure the Dataframes ################
    # Replace every rank #1 in monoRankDF with the monopolar channel (in first column: 'Unnamed: 0')
    monoRank_replace1DF = monoRankDF.apply(
        lambda x: x.where(x != 1.0, monoRankDF["Unnamed: 0"]), axis=0
    )

    # drop first column "Unnamed: 0" with all monopolar Ref channel names
    monoRank_replace1DF.drop(
        columns=monoRank_replace1DF.columns[0], axis=1, inplace=True
    )

    # only select the strings values with the monopolar channel #1 rank for each column (session_frequencyBand)
    monopolarFirstRank = {}

    # loop over each column
    for col in monoRank_replace1DF.columns:
        # extract the column as a series
        column = monoRank_replace1DF[col]

        # exclude float values and replace floats by nan
        # lambda function returns the value if it is a string, otherwise it returns np.nan
        column = column.apply(lambda x: x if isinstance(x, str) else np.nan)

        # drop all NaN values
        column.dropna(how="all", inplace=True)

        # find the first value that is a string (e.g. "monopolarRef_1A")
        # The next function returns the first value of each column that is a string. If the sequence is empty, the default value None is returned.
        value = next((value for value in column.values if isinstance(value, str)), None)

        # add the result to the dictionary
        monopolarFirstRank[col] = value

    # convert the dictionary to a dataframe
    monopolarFirstRankDF = pd.DataFrame(
        list(monopolarFirstRank.items()),
        columns=["session_frequencyBand", "numberOneRank_monopolarChannel"],
    )

    # from monoRefDF extract only the row equal to the value of the column 'numberOneRank_monopolarChannel'

    # loop through each #1 rank value and store the matching dataframe row from monoRefDF in a dictionary
    FirstRankRef_dict = {}

    for index, value in monopolarFirstRankDF[
        "numberOneRank_monopolarChannel"
    ].iteritems():
        FirstRankRef_dict[f"{index}_{value}"] = monoRefDF[
            monoRefDF["Unnamed: 0"].str.contains(value)
        ]

    # first make a new Dataframe of selected monoRef rows, by concatenating all values of FirstRankRef_dict
    FirstRankRef_DF = pd.concat(
        FirstRankRef_dict.values(), keys=FirstRankRef_dict.keys(), ignore_index=True
    )  # keys need to be specified

    # drop the first column with monopolar channel names, because this column already exists in the monopolarFirstRank DF that will be concatenated
    FirstRankRef_DF.drop(
        columns=FirstRankRef_DF.columns[0], axis=1, inplace=True
    )  # inplace=True will modify the original DF and will not create a new DF

    # now concatenate the FirstRankDF (with #1 ranked monopolar contacts) with the FirstRankRefDF (with all referenced psd values of this #1 ranked contact)
    FirstRankChannel_PSD_DF = pd.concat([monopolarFirstRankDF, FirstRankRef_DF], axis=1)

    ################ SELECT HIGHEST RANK FOR POSTOP AND FU3M BASELINE ################

    # Dataframes for postop Baseline first ranks in each frequency band
    postop_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains(
        "postop"
    )
    fu3m_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("fu3m")
    beta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_beta")
    lowBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains(
        "_lowBeta"
    )
    highBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains(
        "_highBeta"
    )

    #######    POSTOP BASELINE  #######
    # first select the row from the highest ranked channel in each frequency band
    postopBaseline_beta_channelRow = FirstRankChannel_PSD_DF[postop_mask & beta_mask]
    postopBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[
        postop_mask & lowBeta_mask
    ]
    postopBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[
        postop_mask & highBeta_mask
    ]

    # # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    postopBaseline_beta = postopBaseline_beta_channelRow.loc[
        :, postopBaseline_beta_channelRow.columns.str.contains("_beta")
    ].reset_index(drop=True)
    postopBaseline_lowBeta = postopBaseline_lowBeta_channelRow.loc[
        :, postopBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")
    ].reset_index(drop=True)
    postopBaseline_highBeta = postopBaseline_highBeta_channelRow.loc[
        :, postopBaseline_highBeta_channelRow.columns.str.contains("_highBeta")
    ].reset_index(drop=True)

    #######    FU3M BASELINE  #######
    # first select the row from the highest ranked channel in each frequency band
    fu3mBaseline_beta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & beta_mask]
    fu3mBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & lowBeta_mask]
    fu3mBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[
        fu3m_mask & highBeta_mask
    ]

    # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    fu3mBaseline_beta = fu3mBaseline_beta_channelRow.loc[
        :, fu3mBaseline_beta_channelRow.columns.str.contains("_beta")
    ].reset_index(drop=True)
    fu3mBaseline_lowBeta = fu3mBaseline_lowBeta_channelRow.loc[
        :, fu3mBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")
    ].reset_index(drop=True)
    fu3mBaseline_highBeta = fu3mBaseline_highBeta_channelRow.loc[
        :, fu3mBaseline_highBeta_channelRow.columns.str.contains("_highBeta")
    ].reset_index(drop=True)

    # save Dataframes as csv in the results folder
    FirstRankChannel_PSD_DF.to_csv(
        os.path.join(
            results_path, f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"
        ),
        sep=",",
    )
    postopBaseline_beta.to_csv(
        os.path.join(results_path, f"postopBaseline_beta{normalization}_{hemisphere}"),
        sep=",",
    )
    postopBaseline_lowBeta.to_csv(
        os.path.join(
            results_path, f"postopBaseline_lowBeta{normalization}_{hemisphere}"
        ),
        sep=",",
    )
    postopBaseline_highBeta.to_csv(
        os.path.join(
            results_path, f"postopBaseline_highBeta{normalization}_{hemisphere}"
        ),
        sep=",",
    )
    fu3mBaseline_beta.to_csv(
        os.path.join(results_path, f"fu3mBaseline_beta{normalization}_{hemisphere}"),
        sep=",",
    )
    fu3mBaseline_lowBeta.to_csv(
        os.path.join(results_path, f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"),
        sep=",",
    )
    fu3mBaseline_highBeta.to_csv(
        os.path.join(
            results_path, f"fu3mBaseline_highBeta{normalization}_{hemisphere}"
        ),
        sep=",",
    )

    return {
        "FirstRankChannel_PSD_DF": FirstRankChannel_PSD_DF,
        "PostopBaseline_beta": postopBaseline_beta,
        "PostopBaseline_lowBeta": postopBaseline_lowBeta,
        "PostopBaseline_highBeta": postopBaseline_highBeta,
        "Fu3mBaseline_beta": fu3mBaseline_beta,
        "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
        "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,
    }


def fooof_monoRef_JLB(fooof_version: str):
    """
    FIRST WEIGHT POWER SPECTRA, THEN AVERAGE BETA AFTERWARDS!
    Calculate the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C)

    Input:
        - power_range: "beta", "low_beta", "high_beta" (TODO: for this I have to rewrite the original FOOOF JSON files!)

    Load the fooof Dataframe and edit it:




    1) Calculate the percentage of each direction A, B and C:
        - proxy of direction:
            A = 1A2A
            B = 1B2B
            C = 1C2C

        - Percentage of direction = Mean beta power of one direction divided by total mean beta power of all directions

    2) Weight each segmented level 1 and 2 with percentage of direction:
        - proxy of hight:
            1 = 02
            2 = 13

        - Percentage of direction multiplied with mean beta power of each level
        - e.g. 1A = Percentage of direction(A) * mean beta power (02)


    """

    results_path = find_folders.get_local_path(folder="GroupResults")

    incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]
    segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]

    monopolar_results_single = {}
    monopolar_results_all = pd.DataFrame()

    ############# Load the FOOOF dataframe #############
    beta_average_DF = loadResults.load_fooof_beta_ranks(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        all_or_one_chan="beta_all",
        all_or_one_longterm_ses="one_longterm_session",
    )

    # loop over sessions
    for ses in incl_sessions:
        # check if session exists
        if ses not in beta_average_DF.session.values:
            continue

        session_Dataframe = beta_average_DF[beta_average_DF.session == ses]
        # copying session_Dataframe to add new columns
        session_Dataframe_copy = session_Dataframe.copy()
        session_Dataframe_copy = session_Dataframe_copy.reset_index()
        session_Dataframe_copy = session_Dataframe_copy.drop(columns=["index"])

        stn_unique = list(session_Dataframe_copy.subject_hemisphere.unique())

        ##################### for every STN: get directional percentage of beta power and weight every monopolar contact #####################
        for stn in stn_unique:
            stn_data = session_Dataframe_copy.loc[
                session_Dataframe_copy.subject_hemisphere == stn
            ]

            # get FOOOF power from the relevant directional channels and level channels
            ### direction ###
            beta_1A2A = stn_data.loc[stn_data.bipolar_channel == "1A2A"]
            beta_1A2A = beta_1A2A["fooof_power_spectrum"].values[0]

            beta_1B2B = stn_data.loc[stn_data.bipolar_channel == "1B2B"]
            beta_1B2B = beta_1B2B["fooof_power_spectrum"].values[0]

            beta_1C2C = stn_data.loc[stn_data.bipolar_channel == "1C2C"]
            beta_1C2C = beta_1C2C["fooof_power_spectrum"].values[0]

            # percentage of each direction
            sum_directions = np.sum([beta_1A2A, beta_1B2B, beta_1C2C], axis=0)
            sum_directions[
                sum_directions == 0
            ] = np.nan  # replace 0 by NaN, so division by zero won't happen

            direction_A = beta_1A2A / (sum_directions / 3)
            direction_B = beta_1B2B / (sum_directions / 3)
            direction_C = beta_1C2C / (sum_directions / 3)

            ### level ###
            level_1 = stn_data.loc[stn_data.bipolar_channel == "02"]
            level_1 = level_1["fooof_power_spectrum"].values[0]

            level_2 = stn_data.loc[stn_data.bipolar_channel == "13"]
            level_2 = level_2["fooof_power_spectrum"].values[0]

            ### calculate the monopolar estimate of spectral FOOOF power for all segmental contacts ###
            for s, segment in enumerate(segmental_contacts):
                # get level
                if "1" in segment:
                    level = level_1

                elif "2" in segment:
                    level = level_2

                # get direction
                if "A" in segment:
                    direction = direction_A

                elif "B" in segment:
                    direction = direction_B

                elif "C" in segment:
                    direction = direction_C

                weighted_power = direction * level

                # store monopolar references in a dictionary
                monopolar_results_single[f"{ses}_{stn}_{segment}"] = [
                    ses,
                    stn,
                    segment,
                    weighted_power,
                ]

    #################### WRITE DATAFRAMES seperately for each STN to also rank within an STN and session ####################

    monopolar_dataframe = pd.DataFrame(monopolar_results_single)
    monopolar_dataframe.rename(
        index={
            0: "session",
            1: "subject_hemisphere",
            2: "contact",
            3: "weighted_fooof_power_spectrum",
        },
        inplace=True,
    )  # rename the rows
    monopolar_dataframe = monopolar_dataframe.transpose()

    # average of beta band from weighted power spectrum
    monopolar_dataframe_copy = monopolar_dataframe.copy()
    monopolar_dataframe_copy["estimated_monopolar_beta_psd"] = monopolar_dataframe_copy[
        "weighted_fooof_power_spectrum"
    ]
    monopolar_dataframe_copy["estimated_monopolar_beta_psd"] = monopolar_dataframe_copy[
        "estimated_monopolar_beta_psd"
    ].apply(lambda row: np.mean(row[13:36]))

    # rank monopolar estimates for every session and stn
    for ses in incl_sessions:
        # check if session exists
        if ses not in monopolar_dataframe_copy.session.values:
            continue

        session_Dataframe_2 = monopolar_dataframe_copy[
            monopolar_dataframe_copy.session == ses
        ]
        # copying session_Dataframe to add new columns
        stn_unique_2 = list(session_Dataframe_2.subject_hemisphere.unique())

        for stn in stn_unique_2:
            stn_data_2 = session_Dataframe_2.loc[
                session_Dataframe_2.subject_hemisphere == stn
            ]
            stn_data_2_copy = stn_data_2.copy()
            stn_data_2_copy["rank"] = stn_data_2_copy[
                "estimated_monopolar_beta_psd"
            ].rank(
                ascending=False
            )  # rank monopolar estimates per stn and session

            # normalize to maximal beta
            max_value_dir = stn_data_2_copy["estimated_monopolar_beta_psd"].max()
            stn_data_2_copy["beta_relative_to_max"] = (
                stn_data_2_copy["estimated_monopolar_beta_psd"] / max_value_dir
            )

            # cluster values into 3 categories: <40%, 40-70% and >70%
            stn_data_2_copy["beta_cluster"] = stn_data_2_copy[
                "beta_relative_to_max"
            ].apply(helpers.assign_cluster)

            # merge all dataframes (per session per STN)
            monopolar_results_all = pd.concat([monopolar_results_all, stn_data_2_copy])

    # save monopolar psd estimate Dataframes as pickle files
    MonoRef_JLB_result_filepath = os.path.join(
        results_path, f"MonoRef_JLB_fooof_beta_{fooof_version}.pickle"
    )
    with open(MonoRef_JLB_result_filepath, "wb") as file:
        pickle.dump(monopolar_results_all, file)

    print(
        f"MonoRef_JLB_fooof_beta_{fooof_version}.pickle",
        f"\nwritten in: {results_path}",
    )

    return monopolar_results_all


############## externalized BSSU ##############


# def fooof_externalized_bssu_monoRef_JLB(fooof_version: str):
#     """
#     FIRST WEIGHT POWER SPECTRA, THEN AVERAGE BETA AFTERWARDS!
#     Calculate the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C)

#     Input:
#         - power_range: "beta", "low_beta", "high_beta" (TODO: for this I have to rewrite the original FOOOF JSON files!)
#         - data_type: "fooof", "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"


#     Load the fooof Dataframe and edit it:


#     1) Calculate the percentage of each direction A, B and C:
#         - proxy of direction:
#             A = 1A2A
#             B = 1B2B
#             C = 1C2C

#         - Percentage of direction = Mean beta power of one direction divided by total mean beta power of all directions

#     2) Weight each segmented level 1 and 2 with percentage of direction:
#         - proxy of hight:
#             1 = 02
#             2 = 13

#         - Percentage of direction multiplied with mean beta power of each level
#         - e.g. 1A = Percentage of direction(A) * mean beta power (02)


#     """

#     results_path = find_folders.get_local_path(folder="GroupResults")

#     segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]

#     monopolar_results_single = {}
#     monopolar_results_all = pd.DataFrame()

#     ############# Load the FOOOF dataframe #############

#     externalized_bssu_fooof = load_data.load_externalized_pickle(
#         filename="fooof_externalized_group_BSSU_only_high_pass_filtered",
#         fooof_version=fooof_version,
#         reference="bipolar_to_lowermost",
#     )

#     # rename column "contact" to "bipolar_channel"
#     externalized_bssu_fooof.rename(columns={"contact": "bipolar_channel"}, inplace=True)

#     stn_unique = list(externalized_bssu_fooof.subject_hemisphere.unique())

#     ##################### for every STN: get directional percentage of beta power and weight every monopolar contact #####################
#     for stn in stn_unique:
#         stn_data = externalized_bssu_fooof.loc[externalized_bssu_fooof.subject_hemisphere == stn]

#         # get FOOOF power from the relevant directional channels and level channels
#         ### direction ###
#         beta_1A2A = stn_data.loc[stn_data.bipolar_channel == "1A2A"]
#         beta_1A2A = beta_1A2A["fooof_power_spectrum"].values[0]

#         beta_1B2B = stn_data.loc[stn_data.bipolar_channel == "1B2B"]
#         beta_1B2B = beta_1B2B["fooof_power_spectrum"].values[0]

#         beta_1C2C = stn_data.loc[stn_data.bipolar_channel == "1C2C"]
#         beta_1C2C = beta_1C2C["fooof_power_spectrum"].values[0]

#         # percentage of each direction
#         sum_directions = np.sum([beta_1A2A, beta_1B2B, beta_1C2C], axis=0)
#         sum_directions[sum_directions == 0] = np.nan  # replace 0 by NaN, so division by zero won't happen

#         direction_A = beta_1A2A / (sum_directions / 3)
#         direction_B = beta_1B2B / (sum_directions / 3)
#         direction_C = beta_1C2C / (sum_directions / 3)

#         ### level ###
#         level_1 = stn_data.loc[stn_data.bipolar_channel == "02"]
#         level_1 = level_1["fooof_power_spectrum"].values[0]

#         level_2 = stn_data.loc[stn_data.bipolar_channel == "13"]
#         level_2 = level_2["fooof_power_spectrum"].values[0]

#         ### calculate the monopolar estimate of spectral FOOOF power for all segmental contacts ###
#         for s, segment in enumerate(segmental_contacts):
#             # get level
#             if "1" in segment:
#                 level = level_1

#             elif "2" in segment:
#                 level = level_2

#             # get direction
#             if "A" in segment:
#                 direction = direction_A

#             elif "B" in segment:
#                 direction = direction_B

#             elif "C" in segment:
#                 direction = direction_C

#             weighted_power = direction * level

#             # store monopolar references in a dictionary
#             monopolar_results_single[f"{stn}_{segment}"] = ["postop", stn, segment, weighted_power]

#     #################### WRITE DATAFRAMES seperately for each STN to also rank within an STN and session ####################

#     monopolar_dataframe = pd.DataFrame(monopolar_results_single)
#     monopolar_dataframe.rename(
#         index={0: "session", 1: "subject_hemisphere", 2: "contact", 3: "weighted_fooof_power_spectrum"}, inplace=True
#     )  # rename the rows
#     monopolar_dataframe = monopolar_dataframe.transpose()

#     # average of beta band from weighted power spectrum
#     monopolar_dataframe_copy = monopolar_dataframe.copy()
#     monopolar_dataframe_copy["estimated_monopolar_beta_psd"] = monopolar_dataframe_copy["weighted_fooof_power_spectrum"]
#     monopolar_dataframe_copy["estimated_monopolar_beta_psd"] = monopolar_dataframe_copy[
#         "estimated_monopolar_beta_psd"
#     ].apply(lambda row: np.mean(row[13:36]))

#     # rank monopolar estimates for every session and stn

#     # copying session_Dataframe to add new columns
#     stn_unique_2 = list(monopolar_dataframe_copy.subject_hemisphere.unique())

#     for stn in stn_unique_2:
#         stn_data_2 = monopolar_dataframe_copy.loc[monopolar_dataframe_copy.subject_hemisphere == stn]
#         stn_data_2_copy = stn_data_2.copy()
#         stn_data_2_copy["rank"] = stn_data_2_copy["estimated_monopolar_beta_psd"].rank(
#             ascending=False
#         )  # rank monopolar estimates per stn and session

#         # normalize to maximal beta
#         max_value_dir = stn_data_2_copy["estimated_monopolar_beta_psd"].max()
#         stn_data_2_copy["beta_relative_to_max"] = stn_data_2_copy["estimated_monopolar_beta_psd"] / max_value_dir

#         # cluster values into 3 categories: <40%, 40-70% and >70%
#         stn_data_2_copy["beta_cluster"] = stn_data_2_copy["beta_relative_to_max"].apply(helpers.assign_cluster)

#         # merge all dataframes (per session per STN)
#         monopolar_results_all = pd.concat([monopolar_results_all, stn_data_2_copy])

#     # save monopolar psd estimate Dataframes as pickle files
#     MonoRef_JLB_result_filepath = os.path.join(
#         results_path, f"MonoRef_JLB_fooof_externalized_BSSU_beta_{fooof_version}.pickle"
#     )
#     with open(MonoRef_JLB_result_filepath, "wb") as file:
#         pickle.dump(monopolar_results_all, file)

#     print(f"MonoRef_JLB_fooof_externalized_BSSU_beta_{fooof_version}.pickle", f"\nwritten in: {results_path}")

#     return monopolar_results_all


def externalized_bssu_monoRef_JLB(fooof_version: str, data_type: str):
    """
    FIRST WEIGHT POWER SPECTRA, THEN AVERAGE BETA AFTERWARDS!
    Calculate the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C)

    Input:
        - power_range: "beta", "low_beta", "high_beta" (TODO: for this I have to rewrite the original FOOOF JSON files!)
        - data_type: "fooof", "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"


    Load the fooof Dataframe and edit it:




    1) Calculate the percentage of each direction A, B and C:
        - proxy of direction:
            A = 1A2A
            B = 1B2B
            C = 1C2C

        - Percentage of direction = Mean beta power of one direction divided by total mean beta power of all directions

    2) Weight each segmented level 1 and 2 with percentage of direction:
        - proxy of hight:
            1 = 02
            2 = 13

        - Percentage of direction multiplied with mean beta power of each level
        - e.g. 1A = Percentage of direction(A) * mean beta power (02)


    """

    results_path = find_folders.get_local_path(folder="GroupResults")

    segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]

    monopolar_results_single = {}
    monopolar_results_all = pd.DataFrame()
    weighted_power_spectra = {}

    ############# Load the FOOOF dataframe #############

    loaded_data = io_externalized.load_data_to_weight(data_type=data_type)
    externalized_data = loaded_data["loaded_data"]

    # rename column contact to bipolar_channel
    # externalized_data.rename(
    #     columns={loaded_data["contact_channel"]: "bipolar_channel"}, inplace=True
    # )
    spectra_column = loaded_data["spectra"]

    # get frequencies for power plots
    if data_type != "fooof":
        frequencies = externalized_data["frequencies"].values[0]

    elif data_type == "fooof":
        frequencies = np.arange(2, 46)  # fooof model v2: 2-45 Hz

    stn_unique = list(externalized_data.subject_hemisphere.unique())

    ##################### for every STN: get directional percentage of beta power and weight every monopolar contact #####################
    for stn in stn_unique:
        stn_data = externalized_data.loc[externalized_data.subject_hemisphere == stn]

        # get FOOOF power from the relevant directional channels and level channels
        ### direction ###
        beta_1A2A = stn_data.loc[stn_data.bipolar_channel == "1A2A"]
        beta_1A2A = beta_1A2A[f"{spectra_column}"].values[0]

        beta_1B2B = stn_data.loc[stn_data.bipolar_channel == "1B2B"]
        beta_1B2B = beta_1B2B[f"{spectra_column}"].values[0]

        beta_1C2C = stn_data.loc[stn_data.bipolar_channel == "1C2C"]
        beta_1C2C = beta_1C2C[f"{spectra_column}"].values[0]

        # percentage of each direction
        sum_directions = np.sum([beta_1A2A, beta_1B2B, beta_1C2C], axis=0)
        sum_directions[
            sum_directions == 0
        ] = np.nan  # replace 0 by NaN, so division by zero won't happen

        direction_A = beta_1A2A / (sum_directions / 3)
        direction_B = beta_1B2B / (sum_directions / 3)
        direction_C = beta_1C2C / (sum_directions / 3)

        ### level ###
        level_1 = stn_data.loc[stn_data.bipolar_channel == "02"]
        level_1 = level_1[f"{spectra_column}"].values[0]

        level_2 = stn_data.loc[stn_data.bipolar_channel == "13"]
        level_2 = level_2[f"{spectra_column}"].values[0]

        ### calculate the monopolar estimate of spectral FOOOF power for all segmental contacts ###
        weighted_power_spectra_single_stn = {}
        for s, segment in enumerate(segmental_contacts):
            # get level
            if "1" in segment:
                level = level_1

            elif "2" in segment:
                level = level_2

            # get direction
            if "A" in segment:
                direction = direction_A

            elif "B" in segment:
                direction = direction_B

            elif "C" in segment:
                direction = direction_C

            weighted_power = direction * level

            # store monopolar references in a dictionary
            monopolar_results_single[f"{stn}_{segment}"] = [
                "postop",
                stn,
                segment,
                weighted_power,
            ]
            weighted_power_spectra_single_stn[segment] = weighted_power

        weighted_power_spectra[stn] = {
            "weighted_power": weighted_power_spectra_single_stn,
            "frequencies": frequencies,
        }

    #################### WRITE DATAFRAMES seperately for each STN to also rank within an STN and session ####################

    monopolar_dataframe = pd.DataFrame(monopolar_results_single)
    monopolar_dataframe.rename(
        index={
            0: "session",
            1: "subject_hemisphere",
            2: "contact",
            3: "weighted_fooof_power_spectrum",
        },
        inplace=True,
    )  # rename the rows
    monopolar_dataframe = monopolar_dataframe.transpose()

    # average of beta band from weighted power spectrum
    monopolar_dataframe_copy = monopolar_dataframe.copy()
    monopolar_dataframe_copy["estimated_monopolar_beta_psd"] = monopolar_dataframe_copy[
        "weighted_fooof_power_spectrum"
    ]
    monopolar_dataframe_copy["estimated_monopolar_beta_psd"] = monopolar_dataframe_copy[
        "estimated_monopolar_beta_psd"
    ].apply(lambda row: np.mean(row[13:36]))

    # rank monopolar estimates for every session and stn

    # copying session_Dataframe to add new columns
    stn_unique_2 = list(monopolar_dataframe_copy.subject_hemisphere.unique())

    for stn in stn_unique_2:
        stn_data_2 = monopolar_dataframe_copy.loc[
            monopolar_dataframe_copy.subject_hemisphere == stn
        ]
        stn_data_2_copy = stn_data_2.copy()
        stn_data_2_copy["rank"] = stn_data_2_copy["estimated_monopolar_beta_psd"].rank(
            ascending=False
        )  # rank monopolar estimates per stn and session

        # normalize to maximal beta
        max_value_dir = stn_data_2_copy["estimated_monopolar_beta_psd"].max()
        stn_data_2_copy["beta_relative_to_max"] = (
            stn_data_2_copy["estimated_monopolar_beta_psd"] / max_value_dir
        )

        # cluster values into 3 categories: <40%, 40-70% and >70%
        stn_data_2_copy["beta_cluster"] = stn_data_2_copy["beta_relative_to_max"].apply(
            externalized_lfp_preprocessing.assign_cluster
        )

        # merge all dataframes (per session per STN)
        monopolar_results_all = pd.concat([monopolar_results_all, stn_data_2_copy])

    # save monopolar psd estimate Dataframes as pickle files
    io_externalized.save_result_dataframe_as_pickle(
        data=monopolar_results_all,
        filename=f"MonoRef_JLB_{data_type}_externalized_BSSU_beta_{fooof_version}",
    )

    io_externalized.save_result_dataframe_as_pickle(
        data=weighted_power_spectra,
        filename=f"MonoRef_JLB_{data_type}_externalized_BSSU_weighted_power_spectra_{fooof_version}",
    )

    return monopolar_results_all, weighted_power_spectra
