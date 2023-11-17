""" monopolar Referencing: weighting power by Euclidean coordinates and distance to the contact of interest 
(Robert approach) """


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import plotly
import plotly.graph_objs as go

import pickle

# utility functions
from ..utils import loadResults as loadResults
from ..utils import find_folders as find_folders
from ..utils import io_percept as io_percept
from ..utils import percept_lfp_preprocessing as percept_lfp_preprocessing
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..utils import io_externalized as io_externalized


#################### new version with SOURCERY help ####################


# helper functions
def calculate_coordinates(only_segmental: str):
    #####################  defining the coordinates of monopolar contacts #####################
    # rcosθ+(rsinθ)i
    # z coordinates of the vertical axis
    # xy coordinates of the polar plane around the percept device

    d = 2  # SenSight B33005: 0.5mm spacing between electrodes, 1.5mm electrode length
    # so 2mm from center of one contact to center of next contact

    r = 0.65  # change this radius as you wish - needs to be optimised

    if only_segmental == "yes":
        contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
        channels = [
            "1A1B",
            "1A1C",
            "1B1C",
            "2A2B",
            "2A2C",
            "2B2C",
            "1A2A",
            "1B2B",
            "1C2C",
        ]
        contact_coordinates = {
            "1A": [d * 1.0, r * np.cos(0) + r * 1j * np.sin(0)],
            "1B": [d * 1.0, r * np.cos(2 * np.pi / 3) + r * 1j * np.sin(2 * np.pi / 3)],
            "1C": [d * 1.0, r * np.cos(4 * np.pi / 3) + r * 1j * np.sin(4 * np.pi / 3)],
            "2A": [d * 2.0, r * np.cos(0) + r * 1j * np.sin(0)],
            "2B": [d * 2.0, r * np.cos(2 * np.pi / 3) + r * 1j * np.sin(2 * np.pi / 3)],
            "2C": [d * 2.0, r * np.cos(4 * np.pi / 3) + r * 1j * np.sin(4 * np.pi / 3)],
        }

        # contact_coordinates = tuple z-coord + xy-coord

    elif only_segmental == "no":
        contacts = ["0", "3"]
        channels = [
            "01",
            "12",
            "23",
            "1A1B",
            "1A1C",
            "1B1C",
            "2A2B",
            "2A2C",
            "2B2C",
            "1A2A",
            "1B2B",
            "1C2C",
        ]
        contact_coordinates = {
            "0": [d * 0.0, 0 + 0 * 1j],
            "1": [d * 1.0, 0 + 0 * 1j],
            "2": [d * 2.0, 0 + 0 * 1j],
            "3": [d * 3.0, 0 + 0 * 1j],
            "1A": [d * 1.0, r * np.cos(0) + r * 1j * np.sin(0)],
            "1B": [d * 1.0, r * np.cos(2 * np.pi / 3) + r * 1j * np.sin(2 * np.pi / 3)],
            "1C": [d * 1.0, r * np.cos(4 * np.pi / 3) + r * 1j * np.sin(4 * np.pi / 3)],
            "2A": [d * 2.0, r * np.cos(0) + r * 1j * np.sin(0)],
            "2B": [d * 2.0, r * np.cos(2 * np.pi / 3) + r * 1j * np.sin(2 * np.pi / 3)],
            "2C": [d * 2.0, r * np.cos(4 * np.pi / 3) + r * 1j * np.sin(4 * np.pi / 3)],
        }

    # contact_coordinates = tuple z-coord + xy-coord

    return contacts, channels, contact_coordinates


def plot_coordinates(contact_coordinates):
    ##################### lets plot the monopolar contact coordinates! #####################
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    plot_data = []

    for contact in contact_coordinates.keys():
        zs = contact_coordinates[contact][0]

        y = contact_coordinates[contact][1]
        xs = np.real(y)
        ys = np.imag(y)

        trace = go.Scatter3d(
            x=np.array([xs]),
            y=np.array([ys]),
            z=np.array([zs]),
            mode="markers",
            marker={
                "size": 10,
                "opacity": 0.8,
            },
        )
        plot_data.append(trace)

    # Configure the layout.
    layout = go.Layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    # data = [trace]

    plot_figure = go.Figure(data=plot_data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)


def calculate_average_coordinates_between_contacts(
    session_Dataframe_coord, only_segmental, contact_coordinates
):
    ##################### looping over bipolar channels, get average coordinates between 2 contacts #####################
    for idx in session_Dataframe_coord.index:
        # extracting contact names
        bipolar_channel = session_Dataframe_coord.loc[
            idx, "bipolar_channel"
        ]  # e.g. 1A2A

        # extracting individual monopolar contact names from bipolar channels
        if len(bipolar_channel) == 4:  # if len ==4 e.g. 1A2A
            contact_1 = bipolar_channel[:2]  # 1A
            contact_2 = bipolar_channel[2:]  # 2A
            # channel_group = "segments"

        elif len(bipolar_channel) == 2:
            if only_segmental == "yes":
                continue
            contact_1 = bipolar_channel[0]  # 0
            contact_2 = bipolar_channel[1]  # 1
        # storing monopolar contact names for bipolar contacts
        # e.g. channel 1A2A: contact1 = 1A, contact2 = 2A
        session_Dataframe_coord.loc[idx, "contact1"] = contact_1
        session_Dataframe_coord.loc[idx, "contact2"] = contact_2

        # extracting coordinates of each monopolar contact from dictionary contact_coordinates
        coords1 = contact_coordinates[contact_1]
        coords2 = contact_coordinates[contact_2]

        # computing mean distance between monopolar contacts to get the bipolar average coordinate
        # coords1, e.g. contact 1A -> tuple of (z-coordinates, xy-coordinates)
        z_av = np.mean(
            [coords1[0], coords2[0]]
        )  # average of z-coord from contact1 and contact2
        xy_av = np.mean(
            [coords1[1], coords2[1]]
        )  # average of xy-coord from contact1 and contact2

        # storing new coordinates of bipolar contacts
        session_Dataframe_coord.loc[
            idx, "coord_z"
        ] = z_av  # mean of z-coordinates from contact 1 and 2
        session_Dataframe_coord.loc[
            idx, "coord_xy"
        ] = xy_av  # mean of xy-coordinates from contact 1 and 2

    return session_Dataframe_coord


def weight_power_of_single_contacts(
    mono_data,
    mono_data_copy,
    stn_ses_bipolar,
    contacts,
    similarity_calculation,
    spectra_column,
):
    """ """
    # for stn in stn_list:
    #     # only select bipolar channels of this stn and session
    #     stn_ses_bipolar = ses_dataframe.loc[ses_dataframe.subject_hemisphere == stn]
    #     stn_ses_bipolar = stn_ses_bipolar.reset_index()

    #     # Create Dataframe with the coordinates of 6 contact coordinates: 1A, 1B, 1C, 2A, 2B, 2C
    #     mono_data = pd.DataFrame(contact_coordinates).T
    #     mono_data.columns = ['coord_z', 'coord_xy']  # columns with z- and xy-coordinates of each contact

    #     # copy mono_data dataframe to add new columns
    #     mono_data_copy = mono_data.copy()

    #     mono_data_copy["session"] = f"{ses}"
    #     mono_data_copy["subject_hemisphere"] = f"{stn}"
    weighted_power_spectra = {}

    for contact in contacts:
        # extracting coordinates for mono polar contacts
        coord_z = mono_data.loc[contact, "coord_z"]
        coord_xy = mono_data.loc[contact, "coord_xy"]

        # loop over all bipolar channels and compute distance to monopolar contact
        all_dists = (
            []
        )  # list of all distances from all averaged bipolar coordinates to one monopolar coordinate

        for bipolar_channel in stn_ses_bipolar.index:
            # finding difference from the monopolar contact to each bipolar mean coordinates
            diff_z = abs(coord_z - stn_ses_bipolar.loc[bipolar_channel, "coord_z"])
            diff_xy = abs(coord_xy - stn_ses_bipolar.loc[bipolar_channel, "coord_xy"])

            # compute euclidean distance based on both directions
            # Pythagoras: a^2 + b^2 = c^2
            # a=difference of z-coord
            # b=difference of xy-coord
            # c=distance of interest
            dist = np.sqrt(diff_z**2 + diff_xy**2)

            # append the distance
            all_dists.append(dist)

        # collect all distances in numpy array
        all_dists = np.array(all_dists)

        # compute similarity from distances
        if similarity_calculation == "inverse_distance":
            similarity = 1 / all_dists

        elif similarity_calculation == "exp_neg_distance":
            similarity = np.exp(
                -all_dists
            )  # alternative to 1/x, but exp^-x doesn´t reach 0

        # ########### weight beta directly ###########
        # # weighting the beta of bipolar contacts by their similarity to the monopolar contact
        # weighted_beta = stn_ses_bipolar['beta_average'].values * similarity  # two arrays with same length = 9 bip_chans

        # # storing the weighted beta for the mono polar contact
        # mono_data_copy.loc[contact, 'estimated_monopolar_beta_psd'] = np.sum(
        #     weighted_beta
        # )  # sum of all 9 weighted psdAverages = one monopolar contact psdAverage

        ########### weight fooof_power_spectrum ###########
        # weighting the power of bipolar contacts by their similarity to the monopolar contact
        weighted_power = (
            stn_ses_bipolar[spectra_column].values * similarity
        )  # 9 arrays for only segmental

        # sum of all 9 arrays
        sum_weighted_power = np.sum(
            weighted_power, axis=0
        )  # sum of all 9 weighted power spectra = one monopolar contact power spectrum, one array

        # store weighted power spectrum into dict
        weighted_power_spectra[contact] = sum_weighted_power

        beta_from_weighted_power = np.mean(
            sum_weighted_power[11:34]
        )  # 2-95 Hz = 94 values
        # index 0 = 2 Hz, index 11 = 13 Hz, index 33 = 35 Hz

        # storing the weighted power for the mono polar contact
        mono_data_copy.loc[
            contact, "estimated_monopolar_beta_psd"
        ] = beta_from_weighted_power

        # add column with contact
        mono_data_copy.loc[contact, "contact"] = contact

    # ranking the weighted monopolar psd
    mono_data_copy["rank"] = mono_data_copy["estimated_monopolar_beta_psd"].rank(
        ascending=False
    )  # rank highest psdAverage as 1.0

    # normalize to maximal beta
    max_value = mono_data_copy["estimated_monopolar_beta_psd"].max()
    mono_data_copy["beta_relative_to_max"] = (
        mono_data_copy["estimated_monopolar_beta_psd"] / max_value
    )

    # cluster values into 3 categories: <40%, 40-70% and >70%
    mono_data_copy["beta_cluster"] = mono_data_copy["beta_relative_to_max"].apply(
        percept_lfp_preprocessing.assign_cluster
    )

    return mono_data_copy, weighted_power_spectra


def fooof_weight_psd_by_euclidean_distance(
    fooof_spectrum: str,
    fooof_version: str,
    only_segmental: str,
    similarity_calculation: str,
):
    """

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - only_segmental: str e.g. "yes" or "no"

        - similarity_calculation: "inverse_distance", "exp_neg_distance"


    1) define imaginary coordinates for segmental contacts only or for all contacts
        - plot the imaginary contact coordinates using plotly



    2) Load the fooof dataframe and edit dataframe

        - check which sessions exist for this patient
        - only for segmental bipolar channels: add columns
            subject_hemisphere
            contact1
            contact2
            bip_chan
            coord_z = mean coordinate between contact 1 and 2
            coord_xy = mean coordinate between contact 1 and 2
            channel_group

        - delete all rows with Ring bipolar channels using drop NaN

        save Dataframe for each session: session_data[f"{ses}_bipolar_Dataframe"]


    3) Calculate for each segmental contact the estimated PSD

        - new dataframe per session with segmental contacts as index

        - calculate an array with all euclidean distances based on both directions in z- and xy-axis

            diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
            diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])

            dist = np.sqrt(diff_z**2 + diff_xy**2)

        - compute similarity from distances

            similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x does't reach 0

        - normalization of similarity is not necessary when only using segmental bipolar recordings
        -> each contact should have the same similarities

        - weight the recorded psd in a frequency band:

            for each bipolar segmental channel:
            weighted_beta = averaged PSD * similarity
            weighted_power = fooof_power_spectrum * similarity

        -> monopolar estimated psd of one segmental contact = np.sum(weighted_beta)


    4) save the dictionary sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

        - in results_path of the subject
        - keys of dictionary:

            f"{ses}_bipolar_Dataframe" with bipolar content: contact1 and contact2 with coordinates and psd average of bipolar channels

            f"{ses}_monopolar_Dataframe" with monopolar content: contact, estimated_monopol_psd_{freqBand}, rank



    TODO: ask Rob again, is normalization of similarity in this case with only segmental contacts not necessary?

    """
    spectra_column = "fooof_power_spectrum"
    # segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    # segmental_channels = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
    incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]

    # calculate and plot coordinates:
    calculated_coordinates = calculate_coordinates(only_segmental)
    contacts = calculated_coordinates[0]
    channels = calculated_coordinates[1]
    contact_coordinates = calculated_coordinates[2]

    # plot coordinates
    plot_coordinates(contact_coordinates)

    #####################  Loading the Data #####################
    beta_average_DF = loadResults.load_fooof_beta_ranks(
        fooof_spectrum=fooof_spectrum,
        fooof_version=fooof_version,
        all_or_one_chan="beta_ranks_all",
        all_or_one_longterm_ses="one_longterm_session",
    )

    # only take rows of channels of interest
    beta_average_DF = beta_average_DF.loc[
        beta_average_DF.bipolar_channel.isin(channels)
    ]

    session_data = {}
    weighted_power_spectra = {}
    # loop over sessions
    for ses in incl_sessions:
        # check if session exists
        if ses not in beta_average_DF.session.values:
            continue

        session_Dataframe = beta_average_DF[beta_average_DF.session == ses]
        # copying session_Dataframe to add new columns
        session_Dataframe_coord = session_Dataframe.copy()
        session_Dataframe_coord = session_Dataframe_coord.reset_index()
        session_Dataframe_coord = session_Dataframe_coord.drop(
            columns=["index", "level_0"]
        )

        data_with_calculated_coordinates = (
            calculate_average_coordinates_between_contacts(
                session_Dataframe_coord, only_segmental, contact_coordinates
            )
        )

        # store copied and modified session_Dataframe into session dictionary
        session_data[f"{ses}_bipolar_Dataframe"] = data_with_calculated_coordinates

    ##################### Calculate beta psd average of each monopolar contact from all averaged coordinates #####################

    for ses in incl_sessions:
        session_data[f"{ses}_monopolar_Dataframe"] = pd.DataFrame()

        ses_dataframe = session_data[f"{ses}_bipolar_Dataframe"]

        stn_list = list(ses_dataframe.subject_hemisphere.unique())

        for stn in stn_list:
            # only select bipolar channels of this stn and session
            stn_ses_bipolar = ses_dataframe.loc[ses_dataframe.subject_hemisphere == stn]
            stn_ses_bipolar = stn_ses_bipolar.reset_index()

            # Create Dataframe with the coordinates of 6 contact coordinates: 1A, 1B, 1C, 2A, 2B, 2C
            mono_data = pd.DataFrame(contact_coordinates).T
            mono_data.columns = [
                "coord_z",
                "coord_xy",
            ]  # columns with z- and xy-coordinates of each contact

            # copy mono_data dataframe to add new columns
            mono_data_copy = mono_data.copy()

            mono_data_copy["session"] = f"{ses}"
            mono_data_copy["subject_hemisphere"] = f"{stn}"

            weighted_power_dataframe = weight_power_of_single_contacts(
                mono_data,
                mono_data_copy,
                stn_ses_bipolar,
                contacts,
                similarity_calculation,
                spectra_column,
            )

            mono_data_copy = weighted_power_dataframe[0]
            weighted_power = weighted_power_dataframe[1]

            session_data[f"{ses}_monopolar_Dataframe"] = pd.concat(
                [session_data[f"{ses}_monopolar_Dataframe"], mono_data_copy]
            )

            weighted_power_spectra[f"{ses}_{stn}"] = weighted_power

    if only_segmental == "yes":
        filename = "only_segmental_"

    else:
        filename = "segments_and_rings_"

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    io_percept.save_result_dataframe_as_pickle(
        data=session_data,
        filename=f"fooof_monoRef_{filename}weight_beta_psd_by_{similarity_calculation}_{fooof_spectrum}_{fooof_version}",
    )

    io_percept.save_result_dataframe_as_pickle(
        data=weighted_power_spectra,
        filename=f"fooof_euclidean_weighted_power_spectra_{filename}{similarity_calculation}_{fooof_spectrum}_{fooof_version}",
    )

    return session_data


def fooof_monoRef_weight_psd_by_distance_all_contacts(
    similarity_calculation: str, fooof_version: str
):
    """

    Input:

        - similarity_calculation: "inverse_distance", "exp_neg_distance"
        - fooof_version: "v1" or "v2"

    merging the monopolar estimated beta power from segmented contacts only from segmental channels
    and the ring contacts (0 and 3) from all 13 bipolar channels

    """

    sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]

    # load the dataframes from segmented and ring contacts
    segmented_data = fooof_weight_psd_by_euclidean_distance(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        only_segmental="yes",
        similarity_calculation=similarity_calculation,
    )

    ring_data = fooof_weight_psd_by_euclidean_distance(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        only_segmental="no",
        similarity_calculation=similarity_calculation,
    )

    # clean up the dataframes

    merged_data = pd.DataFrame()

    for ses in sessions:
        segmented_clean_data = segmented_data[
            f"{ses}_monopolar_Dataframe"
        ]  # DF of only one session
        segmented_clean_data = segmented_clean_data.dropna()

        ring_clean_data = ring_data[
            f"{ses}_monopolar_Dataframe"
        ]  # DF of only one session
        ring_clean_data = ring_clean_data.dropna()

        # merge into complete dataframe
        merged_data = pd.concat(
            [merged_data, segmented_clean_data, ring_clean_data], ignore_index=True
        )

    all_ranked_data = pd.DataFrame()
    electrodes_list = sorted(merged_data.subject_hemisphere.unique())

    for electrode in electrodes_list:
        electrode_data = merged_data.loc[merged_data.subject_hemisphere == electrode]

        for ses in sessions:
            if ses not in electrode_data.session.values:
                continue

            electrode_session_data = electrode_data.loc[electrode_data.session == ses]

            # rank estimated monopolar beta of all 8 contacts
            electrode_session_copy = electrode_session_data.copy()
            electrode_session_copy["rank_8"] = electrode_session_copy[
                "estimated_monopolar_beta_psd"
            ].rank(ascending=False)
            electrode_session_copy = electrode_session_copy.drop(columns=["rank"])
            electrode_session_copy = electrode_session_copy.reset_index()

            # add column with relative beta power to beta rank 1 power
            beta_rank_1 = electrode_session_copy[
                electrode_session_copy["rank_8"] == 1.0
            ]
            beta_rank_1 = beta_rank_1["estimated_monopolar_beta_psd"].values[
                0
            ]  # just taking psdAverage of rank 1.0

            electrode_session_copy[
                "beta_psd_rel_to_rank1"
            ] = electrode_session_copy.apply(
                lambda row: row["estimated_monopolar_beta_psd"] / beta_rank_1, axis=1
            )  # in each row add to new value psd/beta_rank1
            electrode_session_copy[
                "beta_relative_to_max"
            ] = electrode_session_copy.apply(
                lambda row: row["estimated_monopolar_beta_psd"] / beta_rank_1, axis=1
            )  # in each row add to new value psd/beta_rank1

            # add column with relative beta power to beta rank1 and rank8, so values ranging from 0 to 1
            # value of rank 8
            beta_rank_8 = electrode_session_copy[
                electrode_session_copy["rank_8"] == 8.0
            ]
            beta_rank_8 = beta_rank_8["estimated_monopolar_beta_psd"].values[
                0
            ]  # just taking psdAverage of rank 8.0

            beta_rank_1 = (
                beta_rank_1 - beta_rank_8
            )  # this is necessary to get value 1.0 after dividng the subtracted PSD value of rank 1 by itself

            # in each row add in new column: (psd-beta_rank_8)/beta_rank1
            electrode_session_copy[
                "beta_psd_rel_range_0_to_1"
            ] = electrode_session_copy.apply(
                lambda row: (row["estimated_monopolar_beta_psd"] - beta_rank_8)
                / beta_rank_1,
                axis=1,
            )

            # cluster values into 3 categories: <40%, 40-70% and >70%
            electrode_session_copy["beta_cluster"] = electrode_session_copy[
                "beta_relative_to_max"
            ].apply(externalized_lfp_preprocessing.assign_cluster)

            # save
            all_ranked_data = pd.concat(
                [all_ranked_data, electrode_session_copy], ignore_index=True
            )

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    io_percept.save_result_dataframe_as_pickle(
        data=all_ranked_data,
        filename=f"fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}_{fooof_version}",
    )

    return all_ranked_data


########################## externalized re-referenced BSSU ##########################


def externalized_bssu_weight_psd_by_euclidean_distance(
    fooof_version: str,
    data_type: str,
    only_segmental: str,
    similarity_calculation: str,
):
    """

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - only_segmental: str e.g. "yes" or "no"

        - similarity_calculation: "inverse_distance", "exp_neg_distance"
        - data_type: "fooof", "notch_and_band_pass_filtered", "unfiltered", "only_high_pass_filtered"




    1) define imaginary coordinates for segmental contacts only or for all contacts
        - plot the imaginary contact coordinates using plotly



    2) Load the fooof dataframe and edit dataframe

        - check which sessions exist for this patient
        - only for segmental bipolar channels: add columns
            subject_hemisphere
            contact1
            contact2
            bip_chan
            coord_z = mean coordinate between contact 1 and 2
            coord_xy = mean coordinate between contact 1 and 2
            channel_group

        - delete all rows with Ring bipolar channels using drop NaN

        save Dataframe for each session: session_data[f"{ses}_bipolar_Dataframe"]


    3) Calculate for each segmental contact the estimated PSD

        - new dataframe per session with segmental contacts as index

        - calculate an array with all euclidean distances based on both directions in z- and xy-axis

            diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
            diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])

            dist = np.sqrt(diff_z**2 + diff_xy**2)

        - compute similarity from distances

            similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x does't reach 0

        - normalization of similarity is not necessary when only using segmental bipolar recordings
        -> each contact should have the same similarities

        - weight the recorded psd in a frequency band:

            for each bipolar segmental channel:
            weighted_beta = averaged PSD * similarity
            weighted_power = fooof_power_spectrum * similarity

        -> monopolar estimated psd of one segmental contact = np.sum(weighted_beta)


    4) save the dictionary sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

        - in results_path of the subject
        - keys of dictionary:

            f"{ses}_bipolar_Dataframe" with bipolar content: contact1 and contact2 with coordinates and psd average of bipolar channels

            f"{ses}_monopolar_Dataframe" with monopolar content: contact, estimated_monopol_psd_{freqBand}, rank



    TODO: ask Rob again, is normalization of similarity in this case with only segmental contacts not necessary?

    """
    # segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    # segmental_channels = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]

    # calculate and plot coordinates:
    calculated_coordinates = calculate_coordinates(only_segmental)
    contacts = calculated_coordinates[0]
    channels = calculated_coordinates[1]
    contact_coordinates = calculated_coordinates[2]

    # plot coordinates
    plot_coordinates(contact_coordinates)

    #####################  Loading the Data #####################
    loaded_data = io_externalized.load_data_to_weight(data_type=data_type)
    externalized_data = loaded_data["loaded_data"]
    # fooof_externalized_dataframe = load_data.load_externalized_pickle(
    #     filename="fooof_externalized_group_BSSU_only_high_pass_filtered",
    #     fooof_version=fooof_version,
    #     reference="bipolar_to_lowermost",
    # )

    # rename column contact to bipolar_channel
    # externalized_data = externalized_data.rename(columns={loaded_data["contact_channel"]: "bipolar_channel"})
    spectra_column = loaded_data["spectra"]

    # only take rows of channels of interest
    externalized_data = externalized_data.loc[
        externalized_data.bipolar_channel.isin(channels)
    ]

    # get frequencies for power plots
    if data_type != "fooof":
        frequencies = externalized_data["frequencies"].values[0]

    elif data_type == "fooof":
        frequencies = np.arange(2, 46)  # fooof model v2: 2-45 Hz

    session_data = {}
    weighted_power_spectra = {}

    data_with_calculated_coordinates = calculate_average_coordinates_between_contacts(
        externalized_data, only_segmental, contact_coordinates
    )

    # store copied and modified session_Dataframe into session dictionary
    session_data["bipolar_Dataframe"] = data_with_calculated_coordinates

    ##################### Calculate beta psd average of each monopolar contact from all averaged coordinates #####################

    session_data["monopolar_Dataframe"] = pd.DataFrame()

    ses_dataframe = session_data["bipolar_Dataframe"]

    stn_list = list(ses_dataframe.subject_hemisphere.unique())

    for stn in stn_list:
        # only select bipolar channels of this stn and session
        stn_ses_bipolar = ses_dataframe.loc[ses_dataframe.subject_hemisphere == stn]
        stn_ses_bipolar = stn_ses_bipolar.reset_index()

        # Create Dataframe with the coordinates of 6 contact coordinates: 1A, 1B, 1C, 2A, 2B, 2C
        mono_data = pd.DataFrame(contact_coordinates).T
        mono_data.columns = [
            "coord_z",
            "coord_xy",
        ]  # columns with z- and xy-coordinates of each contact

        # copy mono_data dataframe to add new columns
        mono_data_copy = mono_data.copy()

        mono_data_copy["session"] = "postop"
        mono_data_copy["subject_hemisphere"] = f"{stn}"

        weighted_power_dataframe = weight_power_of_single_contacts(
            mono_data,
            mono_data_copy,
            stn_ses_bipolar,
            contacts,
            similarity_calculation,
            spectra_column=spectra_column,
        )

        mono_data_copy = weighted_power_dataframe[0]
        weighted_power = weighted_power_dataframe[1]

        session_data["monopolar_Dataframe"] = pd.concat(
            [session_data["monopolar_Dataframe"], mono_data_copy]
        )

        # weighted_power_spectra[stn] = weighted_power
        weighted_power_spectra[stn] = {
            "weighted_power": weighted_power,
            "frequencies": frequencies,
        }

    if only_segmental == "yes":
        filename = "only_segmental_"

    else:
        filename = "segments_and_rings_"

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    io_externalized.save_result_dataframe_as_pickle(
        data=session_data,
        filename=f"{data_type}_externalized_BSSU_monoRef_{filename}weight_beta_psd_by_{similarity_calculation}_{fooof_version}",
    )

    io_externalized.save_result_dataframe_as_pickle(
        data=weighted_power_spectra,
        filename=f"{data_type}_externalized_BSSU_euclidean_weighted_power_spectra_{filename}{similarity_calculation}_{fooof_version}",
    )

    return session_data


def fooof_externalized_bssu_monoRef_weight_psd_by_distance_all_contacts(
    similarity_calculation: str, fooof_version: str
):
    """

    Input:

        - similarity_calculation: "inverse_distance", "exp_neg_distance"
        - fooof_version: "v1" or "v2"

    merging the monopolar estimated beta power from segmented contacts only from segmental channels
    and the ring contacts (0 and 3) from all 13 bipolar channels

    """

    sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]

    # load the dataframes from segmented and ring contacts
    segmented_data = fooof_weight_psd_by_euclidean_distance(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        only_segmental="yes",
        similarity_calculation=similarity_calculation,
    )

    ring_data = fooof_weight_psd_by_euclidean_distance(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        only_segmental="no",
        similarity_calculation=similarity_calculation,
    )

    # clean up the dataframes

    merged_data = pd.DataFrame()

    for ses in sessions:
        segmented_clean_data = segmented_data[
            f"{ses}_monopolar_Dataframe"
        ]  # DF of only one session
        segmented_clean_data = segmented_clean_data.dropna()

        ring_clean_data = ring_data[
            f"{ses}_monopolar_Dataframe"
        ]  # DF of only one session
        ring_clean_data = ring_clean_data.dropna()

        # merge into complete dataframe
        merged_data = pd.concat(
            [merged_data, segmented_clean_data, ring_clean_data], ignore_index=True
        )

    all_ranked_data = pd.DataFrame()
    electrodes_list = sorted(merged_data.subject_hemisphere.unique())

    for electrode in electrodes_list:
        electrode_data = merged_data.loc[merged_data.subject_hemisphere == electrode]

        for ses in sessions:
            if ses not in electrode_data.session.values:
                continue

            electrode_session_data = electrode_data.loc[electrode_data.session == ses]

            # rank estimated monopolar beta of all 8 contacts
            electrode_session_copy = electrode_session_data.copy()
            electrode_session_copy["rank_8"] = electrode_session_copy[
                "estimated_monopolar_beta_psd"
            ].rank(ascending=False)
            electrode_session_copy = electrode_session_copy.drop(columns=["rank"])
            electrode_session_copy = electrode_session_copy.reset_index()

            # add column with relative beta power to beta rank 1 power
            beta_rank_1 = electrode_session_copy[
                electrode_session_copy["rank_8"] == 1.0
            ]
            beta_rank_1 = beta_rank_1["estimated_monopolar_beta_psd"].values[
                0
            ]  # just taking psdAverage of rank 1.0

            electrode_session_copy[
                "beta_psd_rel_to_rank1"
            ] = electrode_session_copy.apply(
                lambda row: row["estimated_monopolar_beta_psd"] / beta_rank_1, axis=1
            )  # in each row add to new value psd/beta_rank1
            electrode_session_copy[
                "beta_relative_to_max"
            ] = electrode_session_copy.apply(
                lambda row: row["estimated_monopolar_beta_psd"] / beta_rank_1, axis=1
            )  # in each row add to new value psd/beta_rank1

            # add column with relative beta power to beta rank1 and rank8, so values ranging from 0 to 1
            # value of rank 8
            beta_rank_8 = electrode_session_copy[
                electrode_session_copy["rank_8"] == 8.0
            ]
            beta_rank_8 = beta_rank_8["estimated_monopolar_beta_psd"].values[
                0
            ]  # just taking psdAverage of rank 8.0

            beta_rank_1 = (
                beta_rank_1 - beta_rank_8
            )  # this is necessary to get value 1.0 after dividng the subtracted PSD value of rank 1 by itself

            # in each row add in new column: (psd-beta_rank_8)/beta_rank1
            electrode_session_copy[
                "beta_psd_rel_range_0_to_1"
            ] = electrode_session_copy.apply(
                lambda row: (row["estimated_monopolar_beta_psd"] - beta_rank_8)
                / beta_rank_1,
                axis=1,
            )

            # cluster values into 3 categories: <40%, 40-70% and >70%
            electrode_session_copy["beta_cluster"] = electrode_session_copy[
                "beta_relative_to_max"
            ].apply(externalized_lfp_preprocessing.assign_cluster)

            # save
            all_ranked_data = pd.concat(
                [all_ranked_data, electrode_session_copy], ignore_index=True
            )

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    io_externalized.save_result_dataframe_as_pickle(
        data=all_ranked_data,
        filename=f"fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}_{fooof_version}",
    )

    return all_ranked_data
