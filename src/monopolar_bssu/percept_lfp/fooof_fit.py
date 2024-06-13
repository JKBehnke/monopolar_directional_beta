""" FOOOF Model """


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
import scipy
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import fooof
from fooof.plts.spectra import plot_spectrum

# Local Imports
from ..classes import mainAnalysis_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults
from ..tfr import bssu_from_source_JSON as bssu_json
from ..utils import percept_helpers as helpers
from .. utils import sub_session_dict as sub_session_dict


HEMISPHERES = ["Right", "Left"]
CHANNEL_GROUPS = ["RingL", "SegmIntraL", "SegmInterL", "RingR", "SegmIntraR", "SegmInterR"]
RIGHT_CHANNEL_GROUPS = ["RingR", "SegmIntraR", "SegmInterR"]
LEFT_CHANNEL_GROUPS = ["RingL", "SegmIntraL", "SegmInterL"]
ALL_CHANNELS = ["03", "13", "02", "12", "01", "23", "1A1B", "1B1C", "1A1C", "2A2B", "2B2C", "2A2C", "1A2A", "1B2B", "1C2C"]
SFREQ = 250 # sampling frequency of the data

def get_input_y_n(message: str) -> str:
    """Get `y` or `n` user input."""
    while True:
        user_input = input(f"{message} (y/n)? ")
        if user_input.lower() in ["y", "n"]:
            break
        print(f"Input must be `y` or `n`. Got: {user_input}." " Please provide a valid input.")
    return user_input


def get_input_w_wo_knee(message: str) -> str:
    """Get `w` or `wo` user input."""
    while True:
        user_input = input(f"{message} (w/wo)? ")
        if user_input.lower() in ["w", "wo"]:
            break
        print(f"Input must be `w` or `wo`. Got: {user_input}." " Please provide a valid input.")
    return user_input


def fooof_model_predefined(fooof_version: str):
    """
    Choose between version 1 and 2: "v1" or "v2"

    FOOOF settings v1:
        freq_range = [1, 95]  # frequency range to fit FOOOF model

        model = fooof.FOOOF(
            peak_width_limits=[2, 15.0],
            max_n_peaks=6,
            min_peak_height=0.2,
            peak_threshold=2.0,
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    FOOOF settings v2:
        freq_range = [2, 95]

        model = fooof.FOOOF(
            peak_width_limits=[3, 20.0],
            max_n_peaks=4,
            min_peak_height=0.1
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    """

    allowed = ["v1", "v2"]

    if fooof_version not in allowed:
        print(f"fooof_version input must be in {allowed}.")

    if fooof_version == "v1":
        freq_range = [1, 95]  # frequency range to fit FOOOF model

        model = fooof.FOOOF(
            peak_width_limits=[2, 15.0],  # TODO: 3, 15
            max_n_peaks=6,  # TODO 3 oder 4
            min_peak_height=0.2,
            peak_threshold=2.0,
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    elif fooof_version == "v2":
        freq_range = [2, 95]  # frequency range to fit FOOOF model

        model = fooof.FOOOF(
            peak_width_limits=[3, 20.0],
            max_n_peaks=4,
            min_peak_height=0.1,
            aperiodic_mode="fixed",  # fitting without knee component
            verbose=True,
        )

    return {"freq_range": freq_range, "model": model}


def fooof_fit_single_cleaned(subject: str, fooof_version: str):
    """
    Runs FOOOF for a single subject, all existing sessions.
    Reads the clean Power Spectra pickle files: SPECTROGRAMPSD_clean.pickle in each subject folder

    """

    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    fooof_results = {}

    # get path to results folder of each subject
    local_figures_path = findfolders.get_local_path(folder="figures", sub=subject)
    local_results_path = findfolders.get_local_path(folder="results", sub=subject)

    # get sessions of a subject
    sessions = sub_session_dict.get_sessions(sub=subject)

    for hemisphere in HEMISPHERES:
        # get power spectrum and frequencies from each STN

        clean_power_spectra_DF = loadResults.load_sub_pickle_file(sub=subject, filename="SPECTROGRAMPSD_clean")

        hem_DF = clean_power_spectra_DF.loc[clean_power_spectra_DF["hemisphere"] == hemisphere]

        # only take normalization rawPSD, filter unfiltered
        unfiltered_hem_DF = hem_DF.loc[hem_DF["filter"] == "unfiltered"]
        unfiltered_raw_hem_DF = unfiltered_hem_DF.loc[unfiltered_hem_DF["normalization"] == "rawPsd"]


        for ses in sessions:

            ses_DF = unfiltered_raw_hem_DF.loc[unfiltered_raw_hem_DF["session"] == ses]

            for chan in ALL_CHANNELS:
                # get the power spectra and frequencies from each channel
                chan_DF = ses_DF.loc[ses_DF["bipolar_channel"] == chan]

                power_spectrum = np.array(chan_DF.psd.values[0])
                freqs = np.array(chan_DF.frequencies.values[0])

                ############ SET PLOT LAYOUT ############
                fig, ax = plt.subplots(4, 1, figsize=(7, 20))

                # Plot the unfiltered Power spectrum in first ax
                plot_spectrum(freqs, power_spectrum, log_freqs=False, log_powers=False, ax=ax[0])
                ax[0].grid(False)

                ############ SET FOOOF MODEL ############

                model_version = fooof_model_predefined(fooof_version=fooof_version)
                freq_range = model_version["freq_range"]
                model = model_version["model"]

                # always fit a large Frequency band, later you can select Peaks within specific freq bands
                model.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

                # Plot an example power spectrum, with a model fit in second ax
                # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
                model.plot(ax=ax[1], plt_log=True)  # to evaluate the aperiodic component
                model.plot(
                    ax=ax[2], plt_log=False
                )  # To see the periodic component better without log in frequency axis
                ax[1].grid(False)
                ax[2].grid(False)

                # check if fooof attributes are None:
                if model._peak_fit is None:
                    print(f"subject {subject}, session {ses}, {chan}: model peak fit is None.")
                    continue

                if model._ap_fit is None:
                    print(f"subject {subject}, session {ses}, {chan}: model aperiodic fit is None.")
                    continue

                # plot only the fooof spectrum of the periodic component
                fooof_power_spectrum = 10 ** (model._peak_fit + model._ap_fit) - (10**model._ap_fit)
                plot_spectrum(
                    np.arange(1, (len(fooof_power_spectrum) + 1)),
                    fooof_power_spectrum,
                    log_freqs=False,
                    log_powers=False,
                    ax=ax[3],
                )
                # frequencies: 1-95 Hz with 1 Hz resolution

                # titles
                fig.suptitle(f"sub {subject}, {hemisphere} hemisphere, {ses}, bipolar channel: {chan}", fontsize=25)

                ax[0].set_title("unfiltered, raw power spectrum", fontsize=20, y=0.97, pad=-20)
                ax[3].set_title("power spectrum of periodic component", fontsize=20)

                # mark beta band
                x1 = 13
                x2 = 35
                ax[3].axvspan(x1, x2, color="whitesmoke")
                ax[3].grid(False)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        local_figures_path,
                        f"fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{fooof_version}.svg",
                    ),
                    bbox_inches="tight",
                    format="svg",
                )
                fig.savefig(
                    os.path.join(
                        local_figures_path,
                        f"fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{fooof_version}.png",
                    ),
                    bbox_inches="tight",
                )

                # extract parameters from the chosen model
                # model.print_results()

                ############ SAVE APERIODIC PARAMETERS ############
                # goodness of fit
                err = model.get_params('error')
                r_sq = model.r_squared_

                # aperiodic components
                exp = model.get_params('aperiodic_params', 'exponent')
                offset = model.get_params('aperiodic_params', 'offset')

                # periodic component
                log_power_fooof_periodic_plus_aperiodic = (
                    model._peak_fit + model._ap_fit
                )  # periodic+aperiodic component in log Power axis
                fooof_periodic_component = model._peak_fit  # just periodic component, flattened spectrum

                ############ SAVE ALL PEAKS IN ALPHA; HIGH AND LOW BETA ############

                number_peaks = model.n_peaks_

                # get the highest Peak of each frequency band as an array: CF center frequency, Power, BandWidth
                alpha_peak = fooof.analysis.get_band_peak_fm(
                    model, band=(8.0, 12.0), select_highest=True, attribute="peak_params"
                )

                low_beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 20.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                high_beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(21.0, 35.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 35.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                gamma_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(60.0, 90.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                # save all results in dictionary
                STN = "_".join([subject, hemisphere])

                fooof_results[f"{subject}_{hemisphere}_{ses}_{chan}"] = [
                    STN,
                    ses,
                    chan,
                    err,
                    r_sq,
                    exp,
                    offset,
                    fooof_power_spectrum,
                    log_power_fooof_periodic_plus_aperiodic,
                    fooof_periodic_component,
                    number_peaks,
                    alpha_peak,
                    low_beta_peak,
                    high_beta_peak,
                    beta_peak,
                    gamma_peak,
                ]
    # store results in a DataFrame
    fooof_results_df = pd.DataFrame(fooof_results)
    fooof_results_df.rename(
        index={
            0: "subject_hemisphere",
            1: "session",
            2: "bipolar_channel",
            3: "fooof_error",
            4: "fooof_r_sq",
            5: "fooof_exponent",
            6: "fooof_offset",
            7: "fooof_power_spectrum",
            8: "periodic_plus_aperiodic_power_log",
            9: "fooof_periodic_flat",
            10: "fooof_number_peaks",
            11: "alpha_peak_CF_power_bandWidth",
            12: "low_beta_peak_CF_power_bandWidth",
            13: "high_beta_peak_CF_power_bandWidth",
            14: "beta_peak_CF_power_bandWidth",
            15: "gamma_peak_CF_power_bandWidth",
        },
        inplace=True,
    )

    fooof_results_df = fooof_results_df.transpose()  # just one subject

    return fooof_results_df

def fooof_group_percept_clean(incl_sub: list, fooof_version: str):
    """
    FOOOF Percept, all included

    Input:
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "036",
            "038", "040", "041", "045", "047", "048", "049", "050", "052", "055", "059", "060", "061", "062", "063", "065", "066"]

        - fooof_version: str "v1", "v2"

    1) Load the Power Spectrum from main Class:
        - unfiltered
        - rawPSD (not normalized)
        - all channels: Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - condition: only ran m0s0 so far, if you also want to run m1s0, make sure you analyze data seperately!
        - save the features Power Spectrum and Frequencies as variable for each subject, hemisphere, session, channel combination


    2) First set and fit a FOOOF model without a knee -> within a frequency range from 1-95 Hz (broad frequency range for fitting the aperiodic component)
        - peak_width_limits=[2, 15.0],  # must be a list, low limit should be more than twice as frequency resolution, usually not more than 15Hz bw
        - max_n_peaks=6,                # 4, 5 sometimes misses important peaks, 6 better even though there might be more false positives in high frequencies
        - min_peak_height=0.2,          # 0.2 detects false positives in gamma but better than 0.35 missing relevant peaks in low frequencies
        - peak_threshold=2.0,           # default 2.0, lower if necessary to detect peaks more sensitively
        - aperiodic_mode="fixed",       # fitting without knee component, because there are no knees found so far in the STN
        - verbose=True,

        frequency range for parameterization: 1-95 Hz

        plot a figure with the raw Power spectrum and the fitted model

    3) save figure into figure folder of each subject:
        figure filename: fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_wo_knee.png



    4) Extract following parameters and save as columns into DF
        - 0: "subject_hemisphere",
        - 1: "session",
        - 2: "bipolar_channel",
        - 3: "fooof_error",
        - 4: "fooof_r_sq",
        - 5: "fooof_exponent",
        - 6: "fooof_offset",
        - 7: "fooof_power_spectrum", # with 95 values, 1 Hz frequency resolution, so 1-95 Hz
        - 8: "periodic_plus_aperiodic_power_log",
        - 9: "fooof_periodic_flat",
        - 10: "fooof_number_peaks",
        - 11: "alpha_peak_CF_power_bandWidth",
        - 12: "low_beta_peak_CF_power_bandWidth",
        - 13: "high_beta_peak_CF_power_bandWidth",
        - 14: "beta_peak_CF_power_bandWidth",
        - 15: "gamma_peak_CF_power_bandWidth",


    5) save Dataframe into results folder of each subject
        - filename: "fooof_model_sub{subject}.json"

    """

    fooof_group_results_df = pd.DataFrame()

    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    for subject in incl_sub:
        fooof_single_subject_result = fooof_fit_single_cleaned(subject=subject, fooof_version=fooof_version)

        fooof_group_results_df = pd.concat([fooof_group_results_df, fooof_single_subject_result], ignore_index=True)

    helpers.save_result_dataframe_as_pickle(
        data=fooof_group_results_df, filename=f"fooof_group_data_percept_{fooof_version}"
    )

    return fooof_group_results_df

def exclude_unclean_data_fooof_group():
    """ """
    loaded_fooof_result = loadResults.load_pickle_group_result(filename="fooof_group_data_percept", fooof_version="v2")

    # exclude uncleaned channels
    # rename fu18m and fu24m to fu18o24m


def fooof_fit_single_subject(subject: str, fooof_version: str):
    """

    OLD --> this function takes the uncleaned power spectra, no ecg artifacts removed!
    Input: 

        - subject: "024"
        - fooof_version: "v1" or "v2"

    """

    # define variables
    hemispheres = ["Right", "Left"]
    sessions = ['postop', 'fu3m', 'fu12m', 'fu18m', 'fu24m']
    channels = [
        '03',
        '13',
        '02',
        '12',
        '01',
        '23',
        '1A1B',
        '1B1C',
        '1A1C',
        '2A2B',
        '2B2C',
        '2A2C',
        '1A2A',
        '1B2B',
        '1C2C',
    ]

    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    fooof_results = {}

    # get path to results folder of each subject
    local_figures_path = findfolders.get_local_path(folder="figures", sub=subject)
    local_results_path = findfolders.get_local_path(folder="results", sub=subject)

    for hemisphere in hemispheres:
        # get power spectrum and frequencies from each STN
        data_power_spectrum = mainAnalysis_class.MainClass(
            sub=subject,
            hemisphere=hemisphere,
            filter="unfiltered",
            result="PowerSpectrum",
            incl_session=sessions,
            pickChannels=channels,
            normalization=["rawPsd"],
            feature=["frequency", "time_sectors", "rawPsd", "SEM_rawPsd"],
        )

        for ses in sessions:
            try:
                getattr(data_power_spectrum, ses)

            except AttributeError:
                continue

            for chan in channels:
                # get the power spectra and frequencies from each channel
                chan_data = getattr(data_power_spectrum, ses)
                chan_data = getattr(chan_data, f"BIP_{chan}")

                power_spectrum = np.array(chan_data.rawPsd.data)
                freqs = np.array(chan_data.frequency.data)

                ############ SET PLOT LAYOUT ############
                fig, ax = plt.subplots(4, 1, figsize=(7, 20))

                # Plot the unfiltered Power spectrum in first ax
                plot_spectrum(freqs, power_spectrum, log_freqs=False, log_powers=False, ax=ax[0])
                ax[0].grid(False)

                ############ SET FOOOF MODEL ############

                model_version = fooof_model_predefined(fooof_version=fooof_version)
                freq_range = model_version["freq_range"]
                model = model_version["model"]

                # always fit a large Frequency band, later you can select Peaks within specific freq bands
                model.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

                # Plot an example power spectrum, with a model fit in second ax
                # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
                model.plot(ax=ax[1], plt_log=True)  # to evaluate the aperiodic component
                model.plot(
                    ax=ax[2], plt_log=False
                )  # To see the periodic component better without log in frequency axis
                ax[1].grid(False)
                ax[2].grid(False)

                # check if fooof attributes are None:
                if model._peak_fit is None:
                    print(f"subject {subject}, session {ses}, {chan}: model peak fit is None.")
                    continue

                if model._ap_fit is None:
                    print(f"subject {subject}, session {ses}, {chan}: model aperiodic fit is None.")
                    continue

                # plot only the fooof spectrum of the periodic component
                fooof_power_spectrum = 10 ** (model._peak_fit + model._ap_fit) - (10**model._ap_fit)
                plot_spectrum(
                    np.arange(1, (len(fooof_power_spectrum) + 1)),
                    fooof_power_spectrum,
                    log_freqs=False,
                    log_powers=False,
                    ax=ax[3],
                )
                # frequencies: 1-95 Hz with 1 Hz resolution

                # titles
                fig.suptitle(f"sub {subject}, {hemisphere} hemisphere, {ses}, bipolar channel: {chan}", fontsize=25)

                ax[0].set_title("unfiltered, raw power spectrum", fontsize=20, y=0.97, pad=-20)
                ax[3].set_title("power spectrum of periodic component", fontsize=20)

                # mark beta band
                x1 = 13
                x2 = 35
                ax[3].axvspan(x1, x2, color="whitesmoke")
                ax[3].grid(False)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        local_figures_path,
                        f"fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{fooof_version}.svg",
                    ),
                    bbox_inches="tight",
                    format="svg",
                )
                fig.savefig(
                    os.path.join(
                        local_figures_path,
                        f"fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{fooof_version}.png",
                    ),
                    bbox_inches="tight",
                )

                # extract parameters from the chosen model
                # model.print_results()

                ############ SAVE APERIODIC PARAMETERS ############
                # goodness of fit
                err = model.get_params('error')
                r_sq = model.r_squared_

                # aperiodic components
                exp = model.get_params('aperiodic_params', 'exponent')
                offset = model.get_params('aperiodic_params', 'offset')

                # periodic component
                log_power_fooof_periodic_plus_aperiodic = (
                    model._peak_fit + model._ap_fit
                )  # periodic+aperiodic component in log Power axis
                fooof_periodic_component = model._peak_fit  # just periodic component, flattened spectrum

                ############ SAVE ALL PEAKS IN ALPHA; HIGH AND LOW BETA ############

                number_peaks = model.n_peaks_

                # get the highest Peak of each frequency band as an array: CF center frequency, Power, BandWidth
                alpha_peak = fooof.analysis.get_band_peak_fm(
                    model, band=(8.0, 12.0), select_highest=True, attribute="peak_params"
                )

                low_beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 20.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                high_beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(21.0, 35.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 35.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                gamma_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(60.0, 90.0),
                    select_highest=True,
                    attribute="peak_params",
                )

                # save all results in dictionary
                STN = "_".join([subject, hemisphere])

                fooof_results[f"{subject}_{hemisphere}_{ses}_{chan}"] = [
                    STN,
                    ses,
                    chan,
                    err,
                    r_sq,
                    exp,
                    offset,
                    fooof_power_spectrum,
                    log_power_fooof_periodic_plus_aperiodic,
                    fooof_periodic_component,
                    number_peaks,
                    alpha_peak,
                    low_beta_peak,
                    high_beta_peak,
                    beta_peak,
                    gamma_peak,
                ]
    # store results in a DataFrame
    fooof_results_df = pd.DataFrame(fooof_results)
    fooof_results_df.rename(
        index={
            0: "subject_hemisphere",
            1: "session",
            2: "bipolar_channel",
            3: "fooof_error",
            4: "fooof_r_sq",
            5: "fooof_exponent",
            6: "fooof_offset",
            7: "fooof_power_spectrum",
            8: "periodic_plus_aperiodic_power_log",
            9: "fooof_periodic_flat",
            10: "fooof_number_peaks",
            11: "alpha_peak_CF_power_bandWidth",
            12: "low_beta_peak_CF_power_bandWidth",
            13: "high_beta_peak_CF_power_bandWidth",
            14: "beta_peak_CF_power_bandWidth",
            15: "gamma_peak_CF_power_bandWidth",
        },
        inplace=True,
    )

    fooof_results_df = fooof_results_df.transpose()  # just one subject

    return fooof_results_df


def fooof_fit_percept(incl_sub: list, fooof_version: str):
    """
    FOOOF Percept, old version, this function takes the uncleaned power spectra, no ecg artifacts removed!

    Input:
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029",
        "030", "031", "032", "033", "038", "041",
        "045", "047", "050", "059", "060", "061", "062", "063", "065"]
        - fooof_version: str "v1", "v2"

    1) Load the Power Spectrum from main Class:
        - unfiltered
        - rawPSD (not normalized)
        - all channels: Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - condition: only ran m0s0 so far, if you also want to run m1s0, make sure you analyze data seperately!
        - save the features Power Spectrum and Frequencies as variable for each subject, hemisphere, session, channel combination


    2) First set and fit a FOOOF model without a knee -> within a frequency range from 1-95 Hz (broad frequency range for fitting the aperiodic component)
        - peak_width_limits=[2, 15.0],  # must be a list, low limit should be more than twice as frequency resolution, usually not more than 15Hz bw
        - max_n_peaks=6,                # 4, 5 sometimes misses important peaks, 6 better even though there might be more false positives in high frequencies
        - min_peak_height=0.2,          # 0.2 detects false positives in gamma but better than 0.35 missing relevant peaks in low frequencies
        - peak_threshold=2.0,           # default 2.0, lower if necessary to detect peaks more sensitively
        - aperiodic_mode="fixed",       # fitting without knee component, because there are no knees found so far in the STN
        - verbose=True,

        frequency range for parameterization: 1-95 Hz

        plot a figure with the raw Power spectrum and the fitted model

    3) save figure into figure folder of each subject:
        figure filename: fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_wo_knee.png



    4) Extract following parameters and save as columns into DF
        - 0: "subject_hemisphere",
        - 1: "session",
        - 2: "bipolar_channel",
        - 3: "fooof_error",
        - 4: "fooof_r_sq",
        - 5: "fooof_exponent",
        - 6: "fooof_offset",
        - 7: "fooof_power_spectrum", # with 95 values, 1 Hz frequency resolution, so 1-95 Hz
        - 8: "periodic_plus_aperiodic_power_log",
        - 9: "fooof_periodic_flat",
        - 10: "fooof_number_peaks",
        - 11: "alpha_peak_CF_power_bandWidth",
        - 12: "low_beta_peak_CF_power_bandWidth",
        - 13: "high_beta_peak_CF_power_bandWidth",
        - 14: "beta_peak_CF_power_bandWidth",
        - 15: "gamma_peak_CF_power_bandWidth",


    5) save Dataframe into results folder of each subject
        - filename: "fooof_model_sub{subject}.json"

    """

    fooof_group_results_df = pd.DataFrame()

    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    for subject in incl_sub:
        fooof_single_subject_result = fooof_fit_single_subject(subject=subject, fooof_version=fooof_version)

        fooof_group_results_df = pd.concat([fooof_group_results_df, fooof_single_subject_result], ignore_index=True)

    # one exception: sub-030 fu24m, no Perceive, so run extra FOOOF from JSON
    sub_030_fooof_from_json = bssu_json.write_missing_FOOOF_data_add_to_old_FOOOF(
        sub="030",
        session="fu24m",
        condition="m0s0",
        json_filename="Report_Json_Session_Report_20230830T100130.json",
        fooof_version="v2",
    )

    # add sub_030 fu24m to group dataframe
    fooof_group_results_df = pd.concat([fooof_group_results_df, sub_030_fooof_from_json], ignore_index=True)

    helpers.save_result_dataframe_as_pickle(
        data=fooof_group_results_df, filename=f"fooof_group_data_percept_{fooof_version}"
    )

    return fooof_group_results_df


def highest_beta_channels_fooof_percept(fooof_spectrum: str, fooof_version: str):
    """
    Load the file "fooof_group_data_percept_{fooof_version}.pickle"
    from the group result folder

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - fooof_version:
            "v1" or "v2"

    1) calculate beta average for each channel and rank within 1 stn, 1 session and 1 channel group

    2) rank beta averages and only select the channels with rank 1.0

    Output highest_beta_df
        - containing all stns, all sessions, all channels with rank 1.0 within their channel group

    """

    # load the group dataframe
    fooof_group_result = loadResults.load_pickle_group_result(
        filename="fooof_group_data_percept", fooof_version=fooof_version
    )

    # fooof_group_result = loadResults.load_group_fooof_result(fooof_version=fooof_version)

    # frequency_range = ["beta", "low_beta", "high_beta"]

    # create new column: first duplicate column fooof power spectrum, then apply calculation to each row -> average of indices [13:36] so averaging the beta range
    fooof_group_result_copy = fooof_group_result.copy()

    if fooof_spectrum == "periodic_spectrum":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_power_spectrum"]

    elif fooof_spectrum == "periodic_plus_aperiodic":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["periodic_plus_aperiodic_power_log"]

    elif fooof_spectrum == "periodic_flat":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_periodic_flat"]

    fooof_group_result_copy["beta_average"] = fooof_group_result_copy["beta_average"].apply(
        lambda row: np.mean(row[13:36])
    )

    ################################ WRITE DATAFRAME ONLY WITH HIGHEST BETA CHANNELS PER STN | SESSION | CHANNEL_GROUP ################################
    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]

    stn_unique = fooof_group_result_copy.subject_hemisphere.unique().tolist()

    highest_beta_df = pd.DataFrame()
    beta_ranks_all_channels = pd.DataFrame()
    beta_ranks_all_and_all_channels = pd.DataFrame()

    for stn in stn_unique:
        stn_df = fooof_group_result_copy.loc[fooof_group_result_copy.subject_hemisphere == stn]

        for ses in sessions:
            # check if session exists
            if ses not in stn_df.session.values:
                continue

            else:
                stn_ses_df = stn_df.loc[stn_df.session == ses]  # df of only 1 stn and 1 session

            # save data for all channels, no ranks
            beta_ranks_all_and_all_channels = pd.concat([beta_ranks_all_and_all_channels, stn_ses_df])

            for group in channel_group:
                if group == "ring":
                    channels = ['01', '12', '23']

                elif group == "segm_inter":
                    channels = ["1A2A", "1B2B", "1C2C"]

                elif group == "segm_intra":
                    channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                group_comp_df = stn_ses_df.loc[
                    stn_ses_df["bipolar_channel"].isin(channels)
                ].reset_index()  # df of only 1 stn, 1 session and 1 channel group

                # rank beta average of channels within one channel group
                group_comp_df_copy = group_comp_df.copy()
                group_comp_df_copy["beta_rank"] = group_comp_df_copy["beta_average"].rank(ascending=False)

                beta_ranks = group_comp_df_copy.copy()

                # only keep the row with beta rank 1.0
                group_comp_df_copy = group_comp_df_copy.loc[group_comp_df_copy.beta_rank == 1.0]

                # save to ranked_beta_df
                beta_ranks_all_channels = pd.concat([beta_ranks_all_channels, beta_ranks])
                highest_beta_df = pd.concat([highest_beta_df, group_comp_df_copy])

    # this dataframe contains only the highest beta channel per session and lfp group
    helpers.save_result_dataframe_as_pickle(
        data=highest_beta_df, filename=f"highest_beta_channels_fooof_{fooof_spectrum}_{fooof_version}"
    )

    # this dataframe contains 12 lfp channels and beta ranks per lfp group
    helpers.save_result_dataframe_as_pickle(
        data=beta_ranks_all_channels, filename=f"beta_ranks_all_channels_fooof_{fooof_spectrum}_{fooof_version}"
    )

    # this dataframe contains all 15 LFP channels and no ranks
    helpers.save_result_dataframe_as_pickle(
        data=beta_ranks_all_and_all_channels, filename=f"beta_all_channels_fooof_{fooof_spectrum}_{fooof_version}"
    )

    return {
        "highest_beta_df": highest_beta_df,
        "beta_ranks_all_channels": beta_ranks_all_channels,
        "beta_ranks_all_and_all_channels": beta_ranks_all_and_all_channels,
    }


def fooof_fit_power_spectra(incl_sub: list, fooof_version: str):
    """
    OLD VERSION ONLY MODELING WITHOUT KNEE BECAUSE NOT NEEDED IN THE STN

    Input:
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029",
        "030", "031", "032", "033", "038", "041",
        "045", "047", "050", "059", "060", "061", "062", "063", "065"]
        - fooof_version: str "v1", "v2"

    1) Load the Power Spectrum from main Class:
        - unfiltered
        - rawPSD (not normalized)
        - all channels: Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - condition: only ran m0s0 so far, if you also want to run m1s0, make sure you analyze data seperately!
        - save the features Power Spectrum and Frequencies as variable for each subject, hemisphere, session, channel combination


    2) First set and fit a FOOOF model without a knee -> within a frequency range from 1-95 Hz (broad frequency range for fitting the aperiodic component)
        - peak_width_limits=[2, 15.0],  # must be a list, low limit should be more than twice as frequency resolution, usually not more than 15Hz bw
        - max_n_peaks=6,                # 4, 5 sometimes misses important peaks, 6 better even though there might be more false positives in high frequencies
        - min_peak_height=0.2,          # 0.2 detects false positives in gamma but better than 0.35 missing relevant peaks in low frequencies
        - peak_threshold=2.0,           # default 2.0, lower if necessary to detect peaks more sensitively
        - aperiodic_mode="fixed",       # fitting without knee component, because there are no knees found so far in the STN
        - verbose=True,

        frequency range for parameterization: 1-95 Hz

        plot a figure with the raw Power spectrum and the fitted model

    3) save figure into figure folder of each subject:
        figure filename: fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_wo_knee.png



    4) Extract following parameters and save as columns into DF
        - 0: "subject_hemisphere",
        - 1: "session",
        - 2: "bipolar_channel",
        - 3: "fooof_error",
        - 4: "fooof_r_sq",
        - 5: "fooof_exponent",
        - 6: "fooof_offset",
        - 7: "fooof_power_spectrum", # with 95 values, 1 Hz frequency resolution, so 1-95 Hz
        - 8: "periodic_plus_aperiodic_power_log",
        - 9: "fooof_periodic_flat",
        - 10: "fooof_number_peaks",
        - 11: "alpha_peak_CF_power_bandWidth",
        - 12: "low_beta_peak_CF_power_bandWidth",
        - 13: "high_beta_peak_CF_power_bandWidth",
        - 14: "beta_peak_CF_power_bandWidth",
        - 15: "gamma_peak_CF_power_bandWidth",


    5) save Dataframe into results folder of each subject
        - filename: "fooof_model_sub{subject}.json"

    """

    # define variables
    hemispheres = ["Right", "Left"]
    sessions = ['postop', 'fu3m', 'fu12m', 'fu18m', 'fu24m']
    channels = [
        '03',
        '13',
        '02',
        '12',
        '01',
        '23',
        '1A1B',
        '1B1C',
        '1A1C',
        '2A2B',
        '2B2C',
        '2A2C',
        '1A2A',
        '1B2B',
        '1C2C',
    ]

    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    for subject in incl_sub:
        fooof_results = {}

        # get path to results folder of each subject
        local_figures_path = findfolders.get_local_path(folder="figures", sub=subject)
        local_results_path = findfolders.get_local_path(folder="results", sub=subject)

        for hemisphere in hemispheres:
            # get power spectrum and frequencies from each STN
            data_power_spectrum = mainAnalysis_class.MainClass(
                sub=subject,
                hemisphere=hemisphere,
                filter="unfiltered",
                result="PowerSpectrum",
                incl_session=sessions,
                pickChannels=channels,
                normalization=["rawPsd"],
                feature=["frequency", "time_sectors", "rawPsd", "SEM_rawPsd"],
            )

            for ses in sessions:
                try:
                    getattr(data_power_spectrum, ses)

                except AttributeError:
                    continue

                for chan in channels:
                    # get the power spectra and frequencies from each channel
                    chan_data = getattr(data_power_spectrum, ses)
                    chan_data = getattr(chan_data, f"BIP_{chan}")

                    power_spectrum = np.array(chan_data.rawPsd.data)
                    freqs = np.array(chan_data.frequency.data)

                    ############ SET PLOT LAYOUT ############
                    fig, ax = plt.subplots(4, 1, figsize=(7, 20))

                    # Plot the unfiltered Power spectrum in first ax
                    plot_spectrum(freqs, power_spectrum, log_freqs=False, log_powers=False, ax=ax[0])
                    ax[0].grid(False)

                    ############ SET FOOOF MODEL ############

                    model_version = fooof_model_predefined(fooof_version=fooof_version)
                    freq_range = model_version["freq_range"]
                    model = model_version["model"]

                    # always fit a large Frequency band, later you can select Peaks within specific freq bands
                    model.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

                    # Plot an example power spectrum, with a model fit in second ax
                    # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
                    model.plot(ax=ax[1], plt_log=True)  # to evaluate the aperiodic component
                    model.plot(
                        ax=ax[2], plt_log=False
                    )  # To see the periodic component better without log in frequency axis
                    ax[1].grid(False)
                    ax[2].grid(False)

                    # check if fooof attributes are None:
                    if model._peak_fit is None:
                        print(f"subject {subject}, session {ses}, {chan}: model peak fit is None.")
                        continue

                    if model._ap_fit is None:
                        print(f"subject {subject}, session {ses}, {chan}: model aperiodic fit is None.")
                        continue

                    # plot only the fooof spectrum of the periodic component
                    fooof_power_spectrum = 10 ** (model._peak_fit + model._ap_fit) - (10**model._ap_fit)
                    plot_spectrum(
                        np.arange(1, (len(fooof_power_spectrum) + 1)),
                        fooof_power_spectrum,
                        log_freqs=False,
                        log_powers=False,
                        ax=ax[3],
                    )
                    # frequencies: 1-95 Hz with 1 Hz resolution

                    # titles
                    fig.suptitle(f"sub {subject}, {hemisphere} hemisphere, {ses}, bipolar channel: {chan}", fontsize=25)

                    ax[0].set_title("unfiltered, raw power spectrum", fontsize=20, y=0.97, pad=-20)
                    ax[3].set_title("power spectrum of periodic component", fontsize=20)

                    # mark beta band
                    x1 = 13
                    x2 = 35
                    ax[3].axvspan(x1, x2, color="whitesmoke")
                    ax[3].grid(False)

                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(
                            local_figures_path,
                            f"fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{fooof_version}.svg",
                        ),
                        bbox_inches="tight",
                        format="svg",
                    )
                    fig.savefig(
                        os.path.join(
                            local_figures_path,
                            f"fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{fooof_version}.png",
                        ),
                        bbox_inches="tight",
                    )

                    # extract parameters from the chosen model
                    # model.print_results()

                    ############ SAVE APERIODIC PARAMETERS ############
                    # goodness of fit
                    err = model.get_params('error')
                    r_sq = model.r_squared_

                    # aperiodic components
                    exp = model.get_params('aperiodic_params', 'exponent')
                    offset = model.get_params('aperiodic_params', 'offset')

                    # periodic component
                    log_power_fooof_periodic_plus_aperiodic = (
                        model._peak_fit + model._ap_fit
                    )  # periodic+aperiodic component in log Power axis
                    fooof_periodic_component = model._peak_fit  # just periodic component, flattened spectrum

                    ############ SAVE ALL PEAKS IN ALPHA; HIGH AND LOW BETA ############

                    number_peaks = model.n_peaks_

                    # get the highest Peak of each frequency band as an array: CF center frequency, Power, BandWidth
                    alpha_peak = fooof.analysis.get_band_peak_fm(
                        model, band=(8.0, 12.0), select_highest=True, attribute="peak_params"
                    )

                    low_beta_peak = fooof.analysis.get_band_peak_fm(
                        model,
                        band=(13.0, 20.0),
                        select_highest=True,
                        attribute="peak_params",
                    )

                    high_beta_peak = fooof.analysis.get_band_peak_fm(
                        model,
                        band=(21.0, 35.0),
                        select_highest=True,
                        attribute="peak_params",
                    )

                    beta_peak = fooof.analysis.get_band_peak_fm(
                        model,
                        band=(13.0, 35.0),
                        select_highest=True,
                        attribute="peak_params",
                    )

                    gamma_peak = fooof.analysis.get_band_peak_fm(
                        model,
                        band=(60.0, 90.0),
                        select_highest=True,
                        attribute="peak_params",
                    )

                    # save all results in dictionary
                    STN = "_".join([subject, hemisphere])

                    fooof_results[f"{subject}_{hemisphere}_{ses}_{chan}"] = [
                        STN,
                        ses,
                        chan,
                        err,
                        r_sq,
                        exp,
                        offset,
                        fooof_power_spectrum,
                        log_power_fooof_periodic_plus_aperiodic,
                        fooof_periodic_component,
                        number_peaks,
                        alpha_peak,
                        low_beta_peak,
                        high_beta_peak,
                        beta_peak,
                        gamma_peak,
                    ]
        # store results in a DataFrame
        fooof_results_df = pd.DataFrame(fooof_results)
        fooof_results_df.rename(
            index={
                0: "subject_hemisphere",
                1: "session",
                2: "bipolar_channel",
                3: "fooof_error",
                4: "fooof_r_sq",
                5: "fooof_exponent",
                6: "fooof_offset",
                7: "fooof_power_spectrum",
                8: "periodic_plus_aperiodic_power_log",
                9: "fooof_periodic_flat",
                10: "fooof_number_peaks",
                11: "alpha_peak_CF_power_bandWidth",
                12: "low_beta_peak_CF_power_bandWidth",
                13: "high_beta_peak_CF_power_bandWidth",
                14: "beta_peak_CF_power_bandWidth",
                15: "gamma_peak_CF_power_bandWidth",
            },
            inplace=True,
        )

        fooof_results_df = fooof_results_df.transpose()

        # save DF in subject results folder
        fooof_results_df.to_json(os.path.join(local_results_path, f"fooof_model_sub{subject}_{fooof_version}.json"))

    return {
        "fooof_results_df": fooof_results_df,
    }


def fooof_peaks_per_session():
    """
    Load the group FOOOF json file as Dataframe:
    "fooof_model_group_data.json" from the group result folder

    For all electrodes at all sessions seperately:
    Get the number of Peaks and the % of channels with Peaks within one electrode at one session

    """

    results_path = findfolders.get_local_path(folder="GroupResults")

    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    session_peak_dict = {}

    # load the json file as Dataframe
    fooof_group_result = loadResults.load_group_fooof_result()

    # list of unique STNs
    stn_unique = list(fooof_group_result.subject_hemisphere.unique())

    for stn in stn_unique:
        # get dataframe of one stn
        fooof_stn = fooof_group_result.loc[fooof_group_result.subject_hemisphere == stn]

        for ses in sessions:
            # check if session exists for current STN
            if ses not in fooof_stn.session.values:
                continue

            # get the dataframes for each session seperately
            fooof_session = fooof_stn.loc[(fooof_stn["session"] == ses)]

            # get total number of recordings (per STN all 15 recordings included) per session
            total_number_all_channels_session = len(fooof_session)

            for freq in freq_bands:
                freq_list = []

                for item in fooof_session[f"{freq}_peak_CF_power_bandWidth"].values:
                    # in the column "{freq}_peak_CF_power_bandWidth" each cell contains a list
                    # only take rows with a list, if None is not in the list (so only take rows, if there was a Peak)
                    if None not in item:
                        freq_list.append(item)

                freq_session_df = fooof_session.loc[fooof_session[f"{freq}_peak_CF_power_bandWidth"].isin(freq_list)]

                # count how many freq Peaks exist
                number_freq_peaks_session = len(freq_session_df)

                # calculate % of channels with freq Peaks in this session
                percentage_freq_peaks_session = number_freq_peaks_session / total_number_all_channels_session

                session_peak_dict[f"{stn}_{ses}_{freq}"] = [
                    stn,
                    ses,
                    freq,
                    total_number_all_channels_session,
                    number_freq_peaks_session,
                    percentage_freq_peaks_session,
                ]

    # save the results in a dataframe
    session_peak_df = pd.DataFrame(session_peak_dict)
    session_peak_df.rename(
        index={
            0: "subject_hemisphere",
            1: "session",
            2: "frequency_band",
            3: "total_chans_number",
            4: "number_chans_with_peaks",
            5: "percentage_chans_with_peaks",
        },
        inplace=True,
    )
    session_peak_df = session_peak_df.transpose()

    # save Dataframe with data
    session_peak_filepath = os.path.join(results_path, f"fooof_peaks_per_session.pickle")
    with open(session_peak_filepath, "wb") as file:
        pickle.dump(session_peak_df, file)

    print("file: ", "fooof_peaks_per_session.pickle", "\nwritten in: ", results_path)

    return session_peak_df


def fooof_plot_peaks_per_session():
    """
    Load the file "fooof_peaks_per_session.pickle" from the group results folder

    Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band
        - x = session
        - y = percentage_chans_with_peaks
        - label = freq_band


    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")
    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    # load the pickle file with the numbers and percentages of channels with peaks in all frequency bands
    peaks_per_session = loadResults.load_fooof_peaks_per_session()

    perc_chans_with_peaks_data = {}

    # calculate the Mean and standard deviation across STNs from each frequency band at every session
    for ses in sessions:
        # get the dataframes for each session seperately
        fooof_session = peaks_per_session.loc[(peaks_per_session["session"] == ses)]

        for freq in freq_bands:
            # get dataframes for each frequency
            freq_session_df = fooof_session.loc[fooof_session.frequency_band == freq]

            mean_percentage_chans_with_peaks = np.mean(freq_session_df.percentage_chans_with_peaks.values)
            std_percentage_chans_with_peaks = np.std(freq_session_df.percentage_chans_with_peaks.values)
            sample_size = len(freq_session_df.percentage_chans_with_peaks.values)

            perc_chans_with_peaks_data[f"{ses}_{freq}"] = [
                ses,
                freq,
                mean_percentage_chans_with_peaks,
                std_percentage_chans_with_peaks,
                sample_size,
            ]

            # df_copy = freq_df.copy()
            # new column with mean-std and mean+std
            # df_copy["mean-std"] = df_copy.mean_percentage_chans_with_peaks.values - df_copy.std_percentage_chans_with_peaks.values
            # df_copy["mean+std"] = df_copy.mean_percentage_chans_with_peaks.values + df_copy.std_percentage_chans_with_peaks.values

    # save the mean and std values in a dataframe
    perc_chans_with_peaks_df = pd.DataFrame(perc_chans_with_peaks_data)
    perc_chans_with_peaks_df.rename(
        index={
            0: "session",
            1: "frequency_band",
            2: "mean_percentage_chans_with_peaks",
            3: "std_percentage_chans_with_peaks",
            4: "sample_size",
        },
        inplace=True,
    )
    perc_chans_with_peaks_df = perc_chans_with_peaks_df.transpose()

    ###################### Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band ######################
    fig = plt.figure()
    font = {"size": 14}

    for freq in freq_bands:
        freq_df = perc_chans_with_peaks_df.loc[perc_chans_with_peaks_df.frequency_band == freq]

        if freq == "alpha":
            color = "sandybrown"

        elif freq == "beta":
            color = "darkcyan"

        elif freq == "low_beta":
            color = "turquoise"

        elif freq == "high_beta":
            color = "cornflowerblue"

        elif freq == "gamma":
            color = "plum"

        plt.plot(freq_df.session, freq_df.mean_percentage_chans_with_peaks, label=freq, color=color)
        # plt.fill_between(df_copy.session,
        #                  df_copy["mean-std"],
        #                  df_copy["mean+std"],
        #                  color="gainsboro", alpha=0.5)

        plt.scatter(freq_df.session, freq_df.mean_percentage_chans_with_peaks, color=color)
        # plt.errorbar(freq_df.session, freq_df.mean_percentage_chans_with_peaks, yerr=freq_df.std_percentage_chans_with_peaks, fmt="o", color=color)

    plt.title("BSSU channels with Peaks", fontdict={"size": 19})

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.xlabel("session", fontdict=font)
    plt.ylabel("amount of channels with Peaks \nrelative to all channels per electrode", fontdict=font)
    fig.tight_layout()

    # save figure in group Figures folder
    fig.savefig(os.path.join(figures_path, "fooof_peaks_per_session.png"), bbox_inches="tight")

    print("figure: ", f"fooof_peaks_per_session.png", "\nwritten in: ", figures_path)

    return perc_chans_with_peaks_df


def fooof_peaks_in_freq_band_stats():
    """ """

    figures_path = findfolders.get_local_path(folder="GroupFigures")
    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]

    # load the pickle file with the numbers and percentages of channels with peaks in all frequency bands
    peaks_per_session = loadResults.load_fooof_peaks_per_session()

    for f, freq in enumerate(freq_bands):
        freq_df = peaks_per_session.loc[peaks_per_session.frequency_band == freq]

        # replace session names by integers because of seaborn plot
        freq_df = freq_df.replace(to_replace="postop", value=0)
        freq_df = freq_df.replace(to_replace="fu3m", value=3)
        freq_df = freq_df.replace(to_replace="fu12m", value=12)
        freq_df = freq_df.replace(to_replace="fu18m", value=18)

        fig = plt.figure()
        ax = fig.add_subplot()
        font = {"size": 14}

        sns.violinplot(data=freq_df, x="session", y="percentage_chans_with_peaks", palette="pastel", inner="box", ax=ax)

        sns.stripplot(
            data=freq_df,
            x="session",
            y="percentage_chans_with_peaks",
            ax=ax,
            size=6,
            color="black",
            alpha=0.2,  # Transparency of dots
        )

        sns.despine(left=True, bottom=True)  # get rid of figure frame

        # statistical test: doesn't work if groups have different sample size
        num_sessions = [0, 3, 12, 18]
        pairs = list(combinations(num_sessions, 2))

        annotator = Annotator(ax, pairs, data=freq_df, x='session', y='percentage_chans_with_peaks')
        annotator.configure(
            test='Mann-Whitney', text_format='star'
        )  # or ANOVA first to check if there is any difference between all groups
        annotator.apply_and_annotate()

        plt.title(f"BSSU channels with Peaks in {freq}", fontdict={"size": 19})
        plt.ylabel(f"amount of channels with Peaks \n rel. to all channels per electrode", fontdict=font)
        plt.ylim(-0.25, 2.5)
        plt.xlabel("session", fontdict=font)

        fig.tight_layout()

        # save figure in group Figures folder
        fig.savefig(os.path.join(figures_path, f"fooof_{freq}_peaks_per_session_violinplot.png"), bbox_inches="tight")

    return annotator


def fooof_low_vs_high_beta_ratio():
    """
    Load the file "fooof_peaks_per_session.pickle" from the group results folder

    Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band
        - x = session
        - y = Peaks relative to all Peaks in beta band"
        - label = freq_band
    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")

    peaks_per_session = loadResults.load_fooof_peaks_per_session()

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    rel_low_vs_high_beta = {}

    for ses in sessions:
        session_df = peaks_per_session.loc[peaks_per_session.session == ses]

        low_beta_peaks = session_df.loc[session_df.frequency_band == "low_beta"]
        low_beta_peaks = np.sum(low_beta_peaks.number_chans_with_peaks.values)
        # low_beta_peaks = low_beta_peaks[0]

        high_beta_peaks = session_df.loc[session_df.frequency_band == "high_beta"]
        high_beta_peaks = np.sum(high_beta_peaks.number_chans_with_peaks.values)
        # high_beta_peaks = high_beta_peaks[0]

        # number of low + high beta peaks
        beta_peaks = low_beta_peaks + high_beta_peaks

        # calculate the relative amount of Peaks in low beta and high beta from all Peaks in beta band
        relative_low_beta = low_beta_peaks / beta_peaks
        relative_high_beta = high_beta_peaks / beta_peaks

        rel_low_vs_high_beta[f"{ses}"] = [
            ses,
            beta_peaks,
            low_beta_peaks,
            high_beta_peaks,
            relative_low_beta,
            relative_high_beta,
        ]

    # results transformed to a dataframe
    session_low_vs_high_peak_df = pd.DataFrame(rel_low_vs_high_beta)
    session_low_vs_high_peak_df.rename(
        index={
            0: "session",
            1: "beta_peaks",
            2: "low_beta_peaks",
            3: "high_beta_peaks",
            4: "relative_low_beta",
            5: "relative_high_beta",
        },
        inplace=True,
    )
    session_low_vs_high_peak_df = session_low_vs_high_peak_df.transpose()

    # Plot as lineplot
    fig = plt.figure()

    font = {"size": 14}

    plt.plot(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_low_beta, label="low beta")
    plt.plot(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_high_beta, label="high beta")

    plt.scatter(
        session_low_vs_high_peak_df.session,
        session_low_vs_high_peak_df.relative_low_beta,
    )
    plt.scatter(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_high_beta)

    # alternative: stacked Barplot
    # plt.bar(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_low_beta, label="low beta")
    # plt.bar(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_high_beta, label="high beta", bottom=session_low_vs_high_peak_df.relative_low_beta)

    plt.title("Relative amount of low beta vs high beta peaks", fontdict={"size": 19})

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.xlabel("session", fontdict=font)
    plt.ylabel("Peaks relative to all Peaks in beta band", fontdict=font)
    fig.tight_layout()

    # save figure in group Figures folder
    fig.savefig(os.path.join(figures_path, "fooof_low_vs_high_beta_peaks_per_session.png"), bbox_inches="tight")

    print("figure: ", f"fooof_low_vs_high_beta_peaks_per_session.png", "\nwritten in: ", figures_path)

    return session_low_vs_high_peak_df


def fooof_highest_beta_peak_analysis(highest_beta_session: str, cf_or_power: str):
    """

    Input:
        - highest_beta_session: str "highest_postop" or "all_channels" or "highest_each_session" or "highest_fu3m"
        - cf_or_power: str "power" or "center_frequency"


    Load the group FOOOF json file as Dataframe, depending on input
        - "all_channels" -> "fooof_model_group_data.json" from the group result folder

    Plot a violinplot of the center frequencies of the highest Peaks within the beta band (13-35 Hz) at different sessions
        - x = session
        - y = "Peak center frequency \nin beta band (13-35 Hz)"

    Statistical Test between session groups
        - Mann-Whitney

    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")
    results_path = findfolders.get_local_path(folder="GroupResults")

    # load the json file as df
    if highest_beta_session == "all_channels":
        fooof_result = loadResults.load_group_fooof_result()
        title_name = "Highest Beta Peak Center Frequency (all channels)"

    elif highest_beta_session == "highest_postop":
        fooof_result = highest_beta_channels_fooof(
            fooof_spectrum="periodic_spectrum", highest_beta_session=highest_beta_session
        )
        title_name = "Highest Beta Peak Center Frequency (highest beta channel, baseline postop)"

    elif highest_beta_session == "highest_fu3m":
        fooof_result = highest_beta_channels_fooof(
            fooof_spectrum="periodic_spectrum", highest_beta_session=highest_beta_session
        )
        title_name = "Highest Beta Peak Center Frequency (highest beta channel, baseline 3MFU)"

    elif highest_beta_session == "highest_each_session":
        fooof_result = highest_beta_channels_fooof(
            fooof_spectrum="periodic_spectrum", highest_beta_session=highest_beta_session
        )
        title_name = "Highest Beta Peak Center Frequency (only highest beta channels)"

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    channel_group = ["ring", "segm_inter", "segm_intra"]
    beta_peak_parameters = {}
    group_description = {}

    for group in channel_group:
        if group == "ring":
            channels = ['01', '12', '23']

        elif group == "segm_inter":
            channels = ["1A2A", "1B2B", "1C2C"]

        elif group == "segm_intra":
            channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

        channel_group_df = fooof_result.loc[fooof_result.bipolar_channel.isin(channels)]

        for ses in sessions:
            if ses == "postop":
                numeric_session = 0  # violinplot only allowed integers, not strings for x axis

            elif ses == "fu3m":
                numeric_session = 3

            elif ses == "fu12m":
                numeric_session = 12

            elif ses == "fu18m":
                numeric_session = 18

            session_df = channel_group_df.loc[channel_group_df.session == ses]
            beta_peaks_wo_None = []

            # only get the rows with Peaks (drop all rows with None)
            for item in session_df.beta_peak_CF_power_bandWidth.values:
                if None not in item:
                    beta_peaks_wo_None.append(item)

            beta_peak_ses_df = session_df.loc[session_df["beta_peak_CF_power_bandWidth"].isin(beta_peaks_wo_None)]

            # get only the center frequency from the column beta_peak_CF_power_bandWidth
            for i, item in enumerate(beta_peak_ses_df.beta_peak_CF_power_bandWidth.values):
                # item is a list of center frequency, power, band width of highest Peak in beta band

                beta_cf = item[0]
                beta_power = item[1]
                beta_band_width = item[2]

                beta_peak_parameters[f"{group}_{ses}_{i}"] = [
                    group,
                    numeric_session,
                    beta_cf,
                    beta_power,
                    beta_band_width,
                ]

    # save the results in a dataframe
    beta_peak_parameters_df = pd.DataFrame(beta_peak_parameters)
    beta_peak_parameters_df.rename(
        index={
            0: "channel_group",
            1: "session",
            2: "beta_cf",
            3: "beta_power",
            4: "beta_band_width",
        },
        inplace=True,
    )
    beta_peak_parameters_df = beta_peak_parameters_df.transpose()

    ##################### PLOT VIOLINPLOT OF CENTER FREQUENCIES OF HIGHEST BETA PEAKS #####################
    if cf_or_power == "center_frequency":
        parameter = "beta_cf"

    elif cf_or_power == "power":
        parameter = "beta_power"

    for c_group in channel_group:
        # only get channel group data
        parameters_to_plot = beta_peak_parameters_df.loc[beta_peak_parameters_df.channel_group == c_group]
        # make sure columns are numeric
        parameters_to_plot["session"] = parameters_to_plot["session"].astype(int)
        parameters_to_plot[parameter] = parameters_to_plot[parameter].astype(float)

        fig = plt.figure()
        ax = fig.add_subplot()

        sns.violinplot(data=parameters_to_plot, x="session", y=parameter, palette="pastel", inner="box", ax=ax)

        # statistical test: doesn't work if groups have different sample size
        num_sessions = [0.0, 3.0, 12.0, 18.0]
        pairs = list(combinations(num_sessions, 2))

        annotator = Annotator(ax, pairs, data=parameters_to_plot, x='session', y=parameter)
        annotator.configure(test='Mann-Whitney', text_format='star')  # or t-test_ind ??
        annotator.apply_and_annotate()

        sns.stripplot(
            data=parameters_to_plot,
            x="session",
            y=parameter,
            ax=ax,
            size=6,
            color="black",
            alpha=0.2,  # Transparency of dots
        )

        sns.despine(left=True, bottom=True)  # get rid of figure frame

        if highest_beta_session == "all_channels":
            title_name = f"{c_group} channels: Highest beta peak {cf_or_power} \n(of all channels)"

        elif highest_beta_session == "highest_postop":
            title_name = (
                f"{c_group} channels: Highest beta peak {cf_or_power} \n(of highest beta channel, baseline postop)"
            )

        elif highest_beta_session == "highest_fu3m":
            title_name = (
                f"{c_group} channels: Highest beta peak {cf_or_power} \n(of highest beta channel, baseline 3MFU)"
            )

        elif highest_beta_session == "highest_each_session":
            title_name = f"{c_group} channels: Highest beta peak {cf_or_power} \n(only highest beta channels)"

        plt.title(title_name)
        plt.ylabel(f"Peak {cf_or_power} \nin beta band (13-35 Hz)")
        plt.xlabel("session")

        fig.tight_layout()
        fig.savefig(
            os.path.join(figures_path, f"fooof_highest_beta_peak_{cf_or_power}_{highest_beta_session}_{c_group}.png"),
            bbox_inches="tight",
        )

        print(
            "figure: ",
            f"fooof_highest_beta_peak_{cf_or_power}_{highest_beta_session}_{c_group}.png",
            "\nwritten in: ",
            figures_path,
        )

        ##################### DESCRIPTION OF EACH SESSION GROUP #####################
        # describe each group
        num_sessions = [0.0, 3.0, 12.0, 18.0]

        for ses in num_sessions:
            session_group = parameters_to_plot.loc[parameters_to_plot.session == ses]
            session_group = np.array(session_group.beta_cf.values).astype(float)

            description = scipy.stats.describe(session_group)

            group_description[f"{c_group}_{ses}_months_postop"] = description

    description_results = pd.DataFrame(group_description)
    description_results.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_results = description_results.transpose()

    # save Dataframe with data
    description_results_filepath = os.path.join(
        results_path, f"fooof_{cf_or_power}_session_description_highest_beta_peak_{highest_beta_session}.pickle"
    )
    with open(description_results_filepath, "wb") as file:
        pickle.dump(description_results, file)

    print(
        "file: ",
        f"fooof_{cf_or_power}_session_description_highest_beta_peak_{highest_beta_session}.pickle",
        "\nwritten in: ",
        results_path,
    )

    return {"beta_peak_parameters_df": beta_peak_parameters_df, "description_results": description_results}


def fooof_rank_beta_peak_power():
    """
    load the results file: "fooof_model_group_data.json" with the function load_group_fooof_result()

    1) split the beta peak array into three seperate columns:
        - beta_center_frequency
        - beta_peak_power
        - beta_band_width

    2) for each stn - session - channel group combination:
        - add a column with ranks of beta peak power

    """

    ################# VARIABLES #################
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    channel_groups = ["ring", "segm_inter", "segm_intra"]

    results_path = findfolders.get_local_path(folder="GroupResults")

    # load the fooof result
    fooof_group_result = loadResults.load_group_fooof_result()

    ################# only take the beta peaks and split the array into seperate columns #################
    # define split array function
    split_array = lambda x: pd.Series(x)

    # new columns with single parameters per cell
    fooof_group_result_copy = fooof_group_result.copy()
    fooof_group_result_copy[["beta_center_frequency", "beta_peak_power", "beta_band_width"]] = fooof_group_result_copy[
        "beta_peak_CF_power_bandWidth"
    ].apply(split_array)
    fooof_group_result_copy = fooof_group_result_copy.drop(
        columns=[
            "alpha_peak_CF_power_bandWidth",
            "low_beta_peak_CF_power_bandWidth",
            "high_beta_peak_CF_power_bandWidth",
            "gamma_peak_CF_power_bandWidth",
        ]
    )

    #################  RANK CHANNELS BY THEIR BETA PEAK POWER #################
    stn_unique = list(fooof_group_result_copy.subject_hemisphere.unique())

    rank_beta_power_dataframe = pd.DataFrame()

    for stn in stn_unique:
        stn_dataframe = fooof_group_result_copy.loc[fooof_group_result_copy.subject_hemisphere == stn]

        for group in channel_groups:
            if group == "ring":
                channel_list = ["01", "12", "23"]

            elif group == "segm_inter":
                channel_list = ["1A2A", "1B2B", "1C2C"]

            elif group == "segm_intra":
                channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

            # get only the channels within a channel group
            stn_group_dataframe = stn_dataframe.loc[stn_dataframe["bipolar_channel"].isin(channel_list)]

            # rank channels by their beta peak power for each session
            for ses in sessions:
                # check if session exists for this stn
                if ses not in stn_group_dataframe.session.values:
                    continue

                # select only session rows
                stn_group_ses_df = stn_group_dataframe.loc[stn_group_dataframe.session == ses]

                # rank the beta peak power values for this session
                rank_by_power = stn_group_ses_df.beta_peak_power.rank(ascending=False)  # highest rank = 1.0
                stn_group_ses_df_copy = stn_group_ses_df.copy()
                stn_group_ses_df_copy["rank_beta_power"] = rank_by_power

                # add to big dataframe
                rank_beta_power_dataframe = pd.concat([rank_beta_power_dataframe, stn_group_ses_df_copy])

    # save dataframe as pickle
    rank_beta_power_dataframe_filepath = os.path.join(results_path, "fooof_rank_beta_power_dataframe.pickle")
    with open(rank_beta_power_dataframe_filepath, "wb") as file:
        pickle.dump(rank_beta_power_dataframe, file)

    print("file: ", "fooof_rank_beta_power_dataframe.pickle", "\nwritten in: ", results_path)

    return rank_beta_power_dataframe


def fooof_rank1_baseline_beta_peak(
    session_baseline: str,
):
    """

    Input:
        - session_baseline = str, e.g. "postop", "fu3m"

    Load the file "fooof_rank_beta_power_dataframe.pickle"
    written by fooof_rank_beta_peak_power()

    1) for each stn-session-channel_group:
        - select the channel with highest beta peak in {session_baseline}
        - extract the channel name, peak power and center frequency

    2) for every following fu session:
        - select only the channel from {session_baseline} with the highest peak
        - normalize the beta peak power and peak center frequency accordingly:
            peak_power_fu / peak_power_session_baseline
            peak_cf_fu / peak_cf_session_baseline

    3) add normalized parameters to a new column and concatenate all DF rows together to one

    """

    # load the dataframe with ranks of beta peak power
    rank_beta_power_dataframe = loadResults.load_fooof_rank_beta_peak_power()

    stn_unique = list(rank_beta_power_dataframe.subject_hemisphere.unique())
    sessions_after_postop = ["fu3m", "fu12m", "fu18m"]
    sessions_after_fu3m = ["fu12m", "fu18m"]
    channel_groups = ["ring", "segm_inter", "segm_intra"]

    normalized_peak_to_baseline_session = pd.DataFrame()

    for stn in stn_unique:
        stn_dataframe = rank_beta_power_dataframe.loc[rank_beta_power_dataframe.subject_hemisphere == stn]

        for group in channel_groups:
            if group == "ring":
                channel_list = ["01", "12", "23"]

            elif group == "segm_inter":
                channel_list = ["1A2A", "1B2B", "1C2C"]

            elif group == "segm_intra":
                channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

            # get only the channels within a channel group
            stn_group_dataframe = stn_dataframe.loc[stn_dataframe["bipolar_channel"].isin(channel_list)]

            ################### POSTOP ###################
            if session_baseline not in stn_group_dataframe.session.values:
                continue

            baseline_dataframe = stn_group_dataframe.loc[stn_group_dataframe.session == session_baseline]

            # check if there was a peak
            if 1.0 not in baseline_dataframe.rank_beta_power.values:
                print(f"no peak in: {stn}, {group}")
                continue

            # select the channel with rank == 1.0
            baseline_rank1 = baseline_dataframe.loc[baseline_dataframe.rank_beta_power == 1.0]
            baseline_rank1_channel = baseline_rank1.bipolar_channel.values[0]  # channel name rank 1
            baseline_rank1_peak_power = baseline_rank1.beta_peak_power.values[0]  # beta peak power rank 1
            normalized_baseline_peak_power = baseline_rank1_peak_power - baseline_rank1_peak_power  # always 0

            baseline_rank1_peak_cf = baseline_rank1.beta_center_frequency.values[0]  # beta peak cf rank 1
            normalized_baseline_peak_cf = baseline_rank1_peak_cf - baseline_rank1_peak_cf  # always 0

            # new column: normalized peak power
            baseline_rank1_copy = baseline_rank1.copy()
            baseline_rank1_copy[f"peak_power_rel_to_{session_baseline}"] = normalized_baseline_peak_power

            # new column: normalized peak cf
            baseline_rank1_copy[f"peak_cf_rel_to_{session_baseline}"] = normalized_baseline_peak_cf

            # save to collected DF
            normalized_peak_to_baseline_session = pd.concat([normalized_peak_to_baseline_session, baseline_rank1_copy])

            ################### FOLLOW UP SESSIONS ###################
            # check for which sessions apart from postop exist and get the rows for the same channel at different sessions
            if session_baseline == "postop":
                sessions_after_baseline = sessions_after_postop

            elif session_baseline == "fu3m":
                sessions_after_baseline = sessions_after_fu3m

            for fu_ses in sessions_after_baseline:
                # check if ses exists
                if fu_ses not in stn_group_dataframe.session.values:
                    continue

                fu_dataframe = stn_group_dataframe.loc[stn_group_dataframe.session == fu_ses]

                # select the rank 1 channel from baseline
                channel_selection = fu_dataframe.loc[fu_dataframe.bipolar_channel == baseline_rank1_channel]
                fu_peak_power = channel_selection.beta_peak_power.values[0]
                fu_peak_cf = channel_selection.beta_center_frequency.values[0]

                # normalize by peak power from baseline
                normalized_peak_power = fu_peak_power - baseline_rank1_peak_power

                # normalize by peak cf from baseline
                normalized_peak_cf = fu_peak_cf - baseline_rank1_peak_cf

                # new column: normalized peak power -> NaN value, if no peak
                channel_selection_copy = channel_selection.copy()
                channel_selection_copy[f"peak_power_rel_to_{session_baseline}"] = normalized_peak_power
                channel_selection_copy[f"peak_cf_rel_to_{session_baseline}"] = normalized_peak_cf

                # save to collected DF
                normalized_peak_to_baseline_session = pd.concat(
                    [normalized_peak_to_baseline_session, channel_selection_copy]
                )

    # replace session names by integers
    normalized_peak_to_baseline_session["session"] = normalized_peak_to_baseline_session["session"].replace(
        {"postop": 0, "fu3m": 3, "fu12m": 12, "fu18m": 18}
    )

    return normalized_peak_to_baseline_session


def fooof_plot_highest_beta_peak_normalized_to_baseline(
    session_baseline: str, peak_parameter: str, normalized_to_session_baseline: str
):
    """
    Input:
        - session_baseline = str, "postop" or "fu3m"
        - peak_parameter = str, "power" or "center_frequency"
        - normalized_to_session_baseline = str "yes" or "no"

    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")

    # load the dataframe with normalized peak values
    normalized_peak_to_baseline_session = fooof_rank1_baseline_beta_peak(session_baseline=session_baseline)

    channel_groups = ["ring", "segm_inter", "segm_intra"]

    results_df = pd.DataFrame()

    # plot for each channel group seperately
    for group in channel_groups:
        if group == "ring":
            channel_list = ["01", "12", "23"]

        elif group == "segm_inter":
            channel_list = ["1A2A", "1B2B", "1C2C"]

        elif group == "segm_intra":
            channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

        # get only the channels within a channel group
        group_dataframe = normalized_peak_to_baseline_session.loc[
            normalized_peak_to_baseline_session["bipolar_channel"].isin(channel_list)
        ]

        ###################### choose the right column to plot  ######################
        if peak_parameter == "power":
            if normalized_to_session_baseline == "yes":
                y_parameter = f"peak_power_rel_to_{session_baseline}"
                y_label = f"peak power \nrelative to peak power at {session_baseline}"

            elif normalized_to_session_baseline == "no":
                y_parameter = "beta_peak_power"
                y_label = f"peak power"

        elif peak_parameter == "center_frequency":
            if normalized_to_session_baseline == "yes":
                y_parameter = f"peak_cf_rel_to_{session_baseline}"
                y_label = f"peak center frequency \nrelative to center frequency at {session_baseline}"

            if normalized_to_session_baseline == "no":
                y_parameter = "beta_center_frequency"
                y_label = "peak center frequency"

        ###################### plot violinplot and scatter ######################
        fig = plt.figure()
        ax = fig.add_subplot()

        sns.violinplot(data=group_dataframe, x="session", y=y_parameter, ax=ax, palette="pastel")

        sns.stripplot(
            data=group_dataframe,
            x="session",
            y=y_parameter,
            size=6,
            alpha=0.4,
            hue="subject_hemisphere",
            palette="mako",
            ax=ax,
        )

        # statistical test: doesn't work if groups have different sample size
        if session_baseline == "postop":
            pairs = list(combinations([0, 3, 12, 18], 2))
            num_sessions = [0, 3, 12, 18]

        elif session_baseline == "fu3m":
            pairs = list(combinations([3, 12, 18], 2))
            num_sessions = [3, 12, 18]

        # pairs = list(combinations(num_sessions, 2))

        annotator = Annotator(ax, pairs, data=group_dataframe, x='session', y=y_parameter)
        annotator.configure(test='Mann-Whitney', text_format='star')  # or t-test_ind ??
        annotator.apply_and_annotate()

        plt.title(
            f"BSSu channels in {group} group with highest peak power in beta band (13-35 Hz) \nduring {session_baseline} recording"
        )
        plt.ylabel(y_label)
        plt.xlabel("session")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                figures_path,
                f"MINUS_fooof_highest_beta_peak_{group}_from_{session_baseline}_{peak_parameter}_normalized_{normalized_to_session_baseline}.png",
            ),
            bbox_inches="tight",
        )

        print(
            "figure: ",
            f"MINUS_fooof_highest_beta_peak_{group}_from_{session_baseline}_{peak_parameter}_normalized_{normalized_to_session_baseline}.png",
            "\nwritten in: ",
            figures_path,
        )

        ##################### DESCRIPTION OF EACH SESSION GROUP #####################
        # describe each group
        group_description = {}

        for ses in num_sessions:
            session_group = group_dataframe.loc[group_dataframe.session == ses]
            session_group = np.array(session_group[y_parameter].values)

            description = scipy.stats.describe(session_group)

            group_description[f"{ses}"] = description

        description_results = pd.DataFrame(group_description)
        description_results.rename(
            index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
            inplace=True,
        )
        description_results = description_results.transpose()
        description_results_copy = description_results.copy()
        description_results_copy["channel_group"] = group

        results_df = pd.concat([results_df, description_results_copy])

    return results_df


def fooof_count_rank1_or_2(
    session_baseline: str,
):
    """
    Input:
        - session_baseline=str, "postop" or "fu3m"

    This function counts
    """

    results_path = findfolders.get_local_path(folder="GroupResults")

    normalized_peak_to_baseline = fooof_rank1_baseline_beta_peak(session_baseline=session_baseline)

    if session_baseline == "postop":
        session = [0, 3, 12, 18]

    elif session_baseline == "fu3m":
        session = [3, 12, 18]

    channel_groups = ["ring", "segm_inter", "segm_intra"]

    count_dict = {}

    for ses in session:
        session_dataframe = normalized_peak_to_baseline.loc[normalized_peak_to_baseline.session == ses]

        for group in channel_groups:
            if group == "ring":
                channel_list = ["01", "12", "23"]

            elif group == "segm_inter":
                channel_list = ["1A2A", "1B2B", "1C2C"]

            elif group == "segm_intra":
                channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

            # get only the channels within a channel group
            group_dataframe = session_dataframe.loc[session_dataframe["bipolar_channel"].isin(channel_list)]

            # check if there are rows in the dataframe
            if group_dataframe["rank_beta_power"].count() == 0:
                total_count = 0

            else:
                # get percentage how often the rank 1 baseline channels stays rank 1 in 3, 12 and 18 MFU
                total_count = group_dataframe[
                    "rank_beta_power"
                ].count()  # number of channels from one session and within a channel group

            # number how often rank 1.0 or rank 2.0
            if 1.0 not in group_dataframe.rank_beta_power.values:
                rank_1_count = 0

            else:
                rank_1_count = group_dataframe["rank_beta_power"].value_counts()[1.0]

            if 2.0 not in group_dataframe.rank_beta_power.values:
                rank_2_count = 0

            else:
                rank_2_count = group_dataframe["rank_beta_power"].value_counts()[2.0]

            # percentage how often channel stays rank 1 or 2
            rank_1_percentage = rank_1_count / total_count
            rank_1_or_2_percentage = (
                rank_1_count + rank_2_count
            ) / total_count  # segm_inter und ring groups only have 3 ranks so not very much info...

            rank_1_proportion = f"{rank_1_count} / {total_count}"
            rank_1_or_2_proportion = f"{rank_1_count + rank_2_count} / {total_count}"

            # save in dict
            count_dict[f"{ses}_{group}"] = [
                ses,
                group,
                total_count,
                rank_1_count,
                rank_2_count,
                rank_1_percentage,
                rank_1_or_2_percentage,
                rank_1_proportion,
                rank_1_or_2_proportion,
            ]

    count_dataframe = pd.DataFrame(count_dict)
    count_dataframe.rename(
        index={
            0: "session",
            1: "channel_group",
            2: "total_count_of_channels",
            3: "count_rank_1",
            4: "count_rank_2",
            5: "percentage_rank_1",
            6: "percentage_rank_1_or_2",
            7: "proportion_staying_rank_1",
            8: "proportion_staying_rank_1_or_2",
        },
        inplace=True,
    )
    count_dataframe = count_dataframe.transpose()

    return count_dataframe


def highest_beta_channels_fooof(fooof_spectrum: str, highest_beta_session: str):
    """
    Load the file "fooof_model_group_data.json"
    from the group result folder

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

    1) calculate beta average for each channel and rank within 1 stn, 1 session and 1 channel group

    2) rank beta averages and only select the channels with rank 1.0

    Output highest_beta_df
        - containing all stns, all sessions, all channels with rank 1.0 within their channel group

    """

    # load the group dataframe
    fooof_group_result = loadResults.load_group_fooof_result()

    # create new column: first duplicate column fooof power spectrum, then apply calculation to each row -> average of indices [13:36] so averaging the beta range
    fooof_group_result_copy = fooof_group_result.copy()

    if fooof_spectrum == "periodic_spectrum":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_power_spectrum"]

    elif fooof_spectrum == "periodic_plus_aperiodic":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["periodic_plus_aperiodic_power_log"]

    elif fooof_spectrum == "periodic_flat":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_periodic_flat"]

    fooof_group_result_copy["beta_average"] = fooof_group_result_copy["beta_average"].apply(
        lambda row: np.mean(row[13:36])
    )

    ################################ WRITE DATAFRAME ONLY WITH HIGHEST BETA CHANNELS PER STN | SESSION | CHANNEL_GROUP ################################
    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    stn_unique = fooof_group_result_copy.subject_hemisphere.unique().tolist()

    beta_rank_df = pd.DataFrame()

    for stn in stn_unique:
        stn_df = fooof_group_result_copy.loc[fooof_group_result_copy.subject_hemisphere == stn]

        for ses in sessions:
            # check if session exists
            if ses not in stn_df.session.values:
                continue

            else:
                stn_ses_df = stn_df.loc[stn_df.session == ses]  # df of only 1 stn and 1 session

            for group in channel_group:
                if group == "ring":
                    channels = ['01', '12', '23']

                elif group == "segm_inter":
                    channels = ["1A2A", "1B2B", "1C2C"]

                elif group == "segm_intra":
                    channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                group_comp_df = stn_ses_df.loc[
                    stn_ses_df["bipolar_channel"].isin(channels)
                ].reset_index()  # df of only 1 stn, 1 session and 1 channel group

                # rank beta average of channels within one channel group
                group_comp_df_copy = group_comp_df.copy()
                group_comp_df_copy["beta_rank"] = group_comp_df_copy["beta_average"].rank(ascending=False)

                # save to ranked_beta_df
                beta_rank_df = pd.concat([beta_rank_df, group_comp_df_copy])

    # depending on input: keep only rank 1.0 or keep postop rank 1 or 3MFU rank 1 channel
    if highest_beta_session == "highest_each_session":
        # only keep the row with beta rank 1.0
        highest_beta_df = beta_rank_df.loc[beta_rank_df.beta_rank == 1.0]

    elif highest_beta_session == "highest_postop":
        highest_beta_df = pd.DataFrame()
        # for each stn get channel name of beta rank 1 in postop and select the channels for the other timepoints
        for stn in stn_unique:
            stn_data = beta_rank_df.loc[beta_rank_df.subject_hemisphere == stn]

            for ses in sessions:
                # check if postop exists
                if "postop" not in stn_data.session.values:
                    continue

                elif ses not in stn_data.session.values:
                    continue

                else:
                    postop_rank1_channels = stn_data.loc[stn_data.session == "postop"]
                    postop_rank1_channels = postop_rank1_channels.loc[postop_rank1_channels.beta_rank == 1.0]

                    stn_ses_data = stn_data.loc[stn_data.session == ses]

                for group in channel_group:
                    if group == "ring":
                        channels = ['01', '12', '23']

                    elif group == "segm_inter":
                        channels = ["1A2A", "1B2B", "1C2C"]

                    elif group == "segm_intra":
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index()

                    # get channel name of rank 1 channel in postop in this channel group
                    postop_1_row = postop_rank1_channels.loc[postop_rank1_channels["bipolar_channel"].isin(channels)]
                    postop_1_channelname = postop_1_row.bipolar_channel.values[0]

                    # select only this channel in all the other sessions
                    selected_rows = group_data.loc[group_data.bipolar_channel == postop_1_channelname]
                    highest_beta_df = pd.concat([highest_beta_df, postop_1_row, selected_rows])

        # drop index columns
        # drop duplicated postop rows
        highest_beta_df = highest_beta_df.drop(columns=["level_0", "index"])
        highest_beta_df = highest_beta_df.drop_duplicates(
            keep="first", subset=["subject_hemisphere", "session", "bipolar_channel"]
        )

    elif highest_beta_session == "highest_fu3m":
        highest_beta_df = pd.DataFrame()
        # for each stn get channel name of beta rank 1 in postop and select the channels for the other timepoints
        for stn in stn_unique:
            stn_data = beta_rank_df.loc[beta_rank_df.subject_hemisphere == stn]

            for ses in sessions:
                # # if session is postop, continue, because we´re only interested in follow ups here
                # if ses == "postop":
                #     continue

                # check if fu3m exists
                if "fu3m" not in stn_data.session.values:
                    continue

                elif ses not in stn_data.session.values:
                    continue

                else:
                    fu3m_rank1_channels = stn_data.loc[stn_data.session == "fu3m"]
                    fu3m_rank1_channels = fu3m_rank1_channels.loc[fu3m_rank1_channels.beta_rank == 1.0]

                    stn_ses_data = stn_data.loc[stn_data.session == ses]

                for group in channel_group:
                    if group == "ring":
                        channels = ['01', '12', '23']

                    elif group == "segm_inter":
                        channels = ["1A2A", "1B2B", "1C2C"]

                    elif group == "segm_intra":
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index()

                    # get channel name of rank 1 channel in fu3m in this channel group
                    fu3m_1_row = fu3m_rank1_channels.loc[fu3m_rank1_channels["bipolar_channel"].isin(channels)]
                    fu3m_1_channelname = fu3m_1_row.bipolar_channel.values[0]

                    # select only this channel in all the other sessions
                    selected_rows = group_data.loc[group_data.bipolar_channel == fu3m_1_channelname]
                    highest_beta_df = pd.concat([highest_beta_df, fu3m_1_row, selected_rows])

        # drop index columns
        # drop duplicated postop rows
        highest_beta_df = highest_beta_df.drop(columns=["level_0", "index"])
        highest_beta_df = highest_beta_df.drop_duplicates(
            keep="first", subset=["subject_hemisphere", "session", "bipolar_channel"]
        )

    return highest_beta_df


def fooof_mixedlm_highest_beta_channels(
    fooof_spectrum: str, highest_beta_session: str, data_to_fit: str, incl_sessions: list, shape_of_model: str
):
    """

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency"

        - incl_sessions: [0,3] or [3,12,18] o [0,3,12,18]

        - shape_of_model: e.g. "straight", "curved", "asymptotic"


    Load the dataframe with highest beta channels in a given baseline session

    ASYMPTOTIC LINE
    lme model: smf.mixedlm(f"{data_to_fit} ~ session + session_asymptotic", data=data_analysis, groups=data_analysis["group"], re_formula="1")
        - this is a mixed linear effects model
        - dependent variable = data_to_fit e.g. beta average
        - independent variable = predictor = session + session_asymptotic
        - groups = electrode ID
        - re_formula = "1" specifying a random intercept model, assuming the same effect of the predictor across groups but different intercepts for each group

    The model will fit a curved line, the sum of a linear function (ax + b) and an asymptotic function a * (x / (1 + x)) + b
        y = a1 * x + a2 * (x / (1 + x))  + b

        - a1 = coef of the linear model (output as Coef. of session)
        - a2 = coef of the asymptotic model (output as Coef. of session_asymptotic)
        - x = continuous range from 0 to 18 (min and max session)
        - b = model intercept (output as Coef. of Intercept)


    CURVED LINE
    lme model: smf.mixedlm(f"{data_to_fit} ~ session + session_sq", data=data_analysis, groups=data_analysis["group"], re_formula="1")
        - this is a mixed linear effects model
        - dependent variable = data_to_fit e.g. beta average
        - independent variable = predictor = session + session squared
        - groups = electrode ID
        - re_formula = "1" specifying a random intercept model, assuming the same effect of the predictor across groups but different intercepts for each group

    The model will fit a curved line, the sum of a linear function (ax + b) and an exponential function (ax^2 + b)
        y = a1 * x + a2 * x**2  + b

        - a1 = coef of the linear model (output as Coef. of session)
        - a2 = coef of the squared model (output as Coef. of session_sq)
        - x = continuous range from 0 to 18 (min and max session)
        - b = model intercept (output as Coef. of Intercept)


    Two figures





    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")
    fontdict = {"size": 25}

    # Load the dataframe with only highest beta channels
    highest_beta_channels = highest_beta_channels_fooof(
        fooof_spectrum=fooof_spectrum, highest_beta_session=highest_beta_session
    )

    le = LabelEncoder()

    # define split array function
    split_array = lambda x: pd.Series(x)

    channel_group = ["ring", "segm_inter", "segm_intra"]

    ring = ['01', '12', '23']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    group_dict = {}  # data with predictions
    sample_size_dict = {}  #
    model_output = {}  # md.fit()

    ############################## create a single dataframe for each channel group with only one highest beta channels per STN ##############################
    for group in channel_group:
        if group == "ring":
            group_df = highest_beta_channels.loc[highest_beta_channels.bipolar_channel.isin(ring)]

        elif group == "segm_inter":
            group_df = highest_beta_channels.loc[highest_beta_channels.bipolar_channel.isin(segm_inter)]

        elif group == "segm_intra":
            group_df = highest_beta_channels.loc[highest_beta_channels.bipolar_channel.isin(segm_intra)]

        # session values have to be integers, add column group with integers for each STN electrode
        group_df_copy = group_df.copy()
        group_df_copy["group"] = le.fit_transform(
            group_df_copy["subject_hemisphere"]
        )  # adds a column "group" with integer values for each subject_hemisphere
        group_df_copy["session"] = group_df_copy.session.replace(
            to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0, 3, 12, 18]
        )

        # split beta peak column into three columns
        group_df_copy[["beta_center_frequency", "beta_peak_power", "beta_band_width"]] = group_df_copy[
            "beta_peak_CF_power_bandWidth"
        ].apply(split_array)
        group_df_copy = group_df_copy.drop(columns=["alpha_peak_CF_power_bandWidth", "gamma_peak_CF_power_bandWidth"])

        group_df_copy = group_df_copy.dropna()

        # only select sessions that are in incl_sessions
        group_df_copy = group_df_copy.loc[group_df_copy.session.isin(incl_sessions)]

        group_dict[group] = group_df_copy

    ############################## perform linear mixed effects model ##############################
    for g, group in enumerate(channel_group):
        data_analysis = group_dict[group]

        # predictor_x = data_analysis.session.values
        data_analysis = data_analysis.copy()
        data_analysis["session_sq"] = data_analysis.session**2
        data_analysis["session_asymptotic"] = data_analysis.session / (1 + data_analysis.session)

        if shape_of_model == "asymptotic":
            md = smf.mixedlm(
                f"{data_to_fit} ~ session + session_asymptotic",
                data=data_analysis,
                groups=data_analysis["group"],
                re_formula="1",
            )

        elif shape_of_model == "curved":
            md = smf.mixedlm(
                f"{data_to_fit} ~ session + session_sq",
                data=data_analysis,
                groups=data_analysis["group"],
                re_formula="1",
            )

        elif shape_of_model == "straight":
            md = smf.mixedlm(
                f"{data_to_fit} ~ session", data=data_analysis, groups=data_analysis["group"], re_formula="1"
            )

        # re_formula defining the random effect
        # re_formula = 1 specifying random intercept model, assuming same effect of predictor for all groups
        # re_formula = f"1 + session" specifying random intercept and slope model
        mdf = md.fit()

        # save linear model result
        print(mdf.summary())
        model_output[group] = mdf

        # add predictions column to dataframe
        yp = mdf.fittedvalues
        group_dict[group]["predictions"] = yp

        for ses in incl_sessions:
            ses_data = data_analysis.loc[data_analysis.session == ses]
            count = ses_data.subject_hemisphere.count()

            # save sample size
            sample_size_dict[f"{group}_{ses}mfu"] = [group, ses, count]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(
        index={
            0: "channel_group",
            1: "session",
            2: "count",
        },
        inplace=True,
    )
    sample_size_df = sample_size_df.transpose()

    ############################## plot the observed values and the model ##############################
    fig_1, axes_1 = plt.subplots(3, 1, figsize=(10, 15))
    fig_2, axes_2 = plt.subplots(3, 1, figsize=(10, 15))

    for g, group in enumerate(channel_group):
        data_analysis = group_dict[group]  # this is the dataframe with data
        mdf_group = model_output[group]  # this is md.fit()

        # get the results
        result_part_2 = mdf_group.summary().tables[1]  # part containing model intercept, slope, std.error
        model_intercept = float(result_part_2["Coef."].values[0])
        model_slope = float(result_part_2["Coef."].values[1])

        if shape_of_model == "asymptotic" or "curved":
            model_slope_2 = float(result_part_2["Coef."].values[2])
            # group_variance = float(result_part_2["Coef."].values[3])

        std_error_intercept = float(result_part_2["Std.Err."].values[0])
        # p_val_intercept = float(result_part_2["P>|z|"].values[0])
        # p_val_session = float(result_part_2["P>|z|"].values[1])
        # p_val_session2 = float(result_part_2["P>|z|"].values[2])
        conf_int = mdf_group.conf_int(
            alpha=0.05
        )  # table with confidence intervals for intercept, session, session2 and group var

        # one subplot per channel group
        axes_1[g].set_title(f"{group} channel group", fontdict=fontdict)
        axes_2[g].set_title(f"{group} channel group", fontdict=fontdict)

        ################## plot the result for each electrode ##################
        for id, group_id in enumerate(data_analysis.group.unique()):
            sub_data = data_analysis[data_analysis.group == group_id]

            # axes[g].scatter(sub_data[f"{data_to_fit}"], sub_data["session"] ,color=plt.cm.twilight_shifted(group_id*10)) # color=plt.cm.tab20(group_id)
            # axes[g].plot(sub_data[f"{data_to_fit}"], sub_data["predictions"], color=plt.cm.twilight_shifted(group_id*10))

            axes_1[g].scatter(
                sub_data["session"], sub_data[f"{data_to_fit}"], color=plt.cm.twilight_shifted((id + 1) * 10), alpha=0.3
            )  # color=plt.cm.tab20(group_id)
            # plot the predictions
            # axes[g].plot(sub_data["session"], sub_data["predictions"], color=plt.cm.twilight_shifted((id+1)*10), linewidth=1, alpha=0.5)
            axes_1[g].plot(
                sub_data["session"],
                sub_data[f"{data_to_fit}"],
                color=plt.cm.twilight_shifted((id + 1) * 10),
                linewidth=1,
                alpha=0.3,
            )

        # plot the model regression line
        if 0 in incl_sessions:
            if 18 in incl_sessions:
                x = np.arange(0, 19)

            else:
                x = np.arange(0, 4)

        elif 0 not in incl_sessions:
            x = np.arange(3, 19)

        ################## plot the modeled curved line ##################
        if shape_of_model == "curved":
            y = x * model_slope + x**2 * model_slope_2 + model_intercept  # curved model

        elif shape_of_model == "asymptotic":
            y = x * model_slope + (x / (1 + x)) * model_slope_2 + model_intercept  # asymptotic model

        elif shape_of_model == "straight":
            y = x * model_slope + model_intercept  # straight line

        axes_1[g].plot(x, y, color="k", linewidth=5)
        # linear model: coef*x (=linear) + coef*x^2 (=exponential) + intercept
        # coef defines the slope

        # pred = mdf_group.predict(exog=dict(x=x))
        # calculate the confidence interval
        # cov_params = mdf_group.cov_params()
        # mse = np.mean(mdf_group.resid.values**2)
        # t_value = stats.t.ppf(0.975, df=mdf_group.df_resid)
        # standard_errors = np.sqrt(np.diag(cov_params))
        # lower_bound = y - t_value * standard_errors * np.sqrt(mse)
        # upper_bound = y + t_value * standard_errors * np.sqrt(mse)

        # lower_bound = y + 1.96 *

        # axes[g].plot(prediction_data["session"], prediction_data["mean_yp"], color="k", linewidth=5)
        # axes[g].fill_between(prediction_data["session"], prediction_data["mean_yp"]-prediction_data["sem_yp"], prediction_data["mean_yp"]+prediction_data["sem_yp"], color='lightgray', alpha=0.5)
        # axes_1[g].fill_between(x, lower_bound, upper_bound, color="k", linewidth=5, alpha=0.3)

        ################### plot the residuals against predictions ##################
        # number of residuals equals the number of observations (number of channels I input)
        resid = mdf_group.resid
        predicted_values = mdf_group.fittedvalues
        axes_2[g].scatter(predicted_values, resid, color="k", alpha=0.2)
        axes_2[g].axhline(y=0, color="red", linestyle="--")

    for ax in axes_1:
        ax.set_ylabel(f"{data_to_fit}", fontsize=25)
        ax.set_xlabel("months post-surgery", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_1.suptitle(f"Linear mixed effects model: {highest_beta_session} beta channels", fontsize=30)
    fig_1.subplots_adjust(wspace=0, hspace=0)

    fig_1.tight_layout()
    fig_1.savefig(
        os.path.join(
            figures_path,
            f"lme_{shape_of_model}_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.png",
        ),
        bbox_inches="tight",
    )
    fig_1.savefig(
        os.path.join(
            figures_path,
            f"lme_{shape_of_model}_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    print("figure: ", f"lme_{data_to_fit}_{highest_beta_session}_beta_channels.png", "\nwritten in: ", figures_path)

    for ax in axes_2:
        ax.set_xlabel("Predicted Values", fontsize=25)
        ax.set_ylabel("Residuals", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_2.suptitle(f"Linear mixed effects model residuals: {highest_beta_session} beta channels", fontsize=30)
    fig_2.subplots_adjust(wspace=0, hspace=0)
    fig_2.tight_layout()
    fig_2.savefig(
        os.path.join(
            figures_path,
            f"lme_{shape_of_model}_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.png",
        ),
        bbox_inches="tight",
    )
    fig_2.savefig(
        os.path.join(
            figures_path,
            f"lme_{shape_of_model}_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    # save results
    mdf_result_filepath = os.path.join(
        results_path,
        f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle",
    )
    with open(mdf_result_filepath, "wb") as file:
        pickle.dump(model_output, file)

    print(
        "file: ",
        f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle",
        "\nwritten in: ",
        results_path,
    )

    return {
        "group_dict": group_dict,
        "sample_size_df": sample_size_df,
        "conf_int": conf_int,
        "model_output": model_output,
        "md": md,
    }


def fooof_ploynomial_regression_model_highest_beta_channels(
    fooof_spectrum: str, highest_beta_session: str, data_to_fit: str, incl_sessions: list
):
    """

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency"

        - incl_sessions: [0,3] or [3,12,18] o [0,3,12,18]


    Load the dataframe with highest beta channels in a given baseline session



    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")
    fontdict = {"size": 25}

    # Load the dataframe with only highest beta channels
    highest_beta_channels = highest_beta_channels_fooof(
        fooof_spectrum=fooof_spectrum, highest_beta_session=highest_beta_session
    )

    le = LabelEncoder()

    # define split array function
    split_array = lambda x: pd.Series(x)

    channel_group = ["ring", "segm_inter", "segm_intra"]

    ring = ['01', '12', '23']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    group_dict = {}
    mdf_result = {}
    prediction_result = {}
    result_dict = {}

    ############################## create a single dataframe for each channel group with only one highest beta channels per STN ##############################
    for group in channel_group:
        if group == "ring":
            group_df = highest_beta_channels.loc[highest_beta_channels.bipolar_channel.isin(ring)]

        elif group == "segm_inter":
            group_df = highest_beta_channels.loc[highest_beta_channels.bipolar_channel.isin(segm_inter)]

        elif group == "segm_intra":
            group_df = highest_beta_channels.loc[highest_beta_channels.bipolar_channel.isin(segm_intra)]

        # session values have to be integers, add column group with integers for each STN electrode
        group_df_copy = group_df.copy()
        group_df_copy["group"] = le.fit_transform(group_df_copy["subject_hemisphere"])
        group_df_copy["session"] = group_df_copy.session.replace(
            to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0, 3, 12, 18]
        )

        # split beta peak column into three columns
        group_df_copy[["beta_center_frequency", "beta_peak_power", "beta_band_width"]] = group_df_copy[
            "beta_peak_CF_power_bandWidth"
        ].apply(split_array)
        group_df_copy = group_df_copy.drop(columns=["alpha_peak_CF_power_bandWidth", "gamma_peak_CF_power_bandWidth"])

        group_df_copy = group_df_copy.dropna()

        # only select sessions that are in incl_sessions
        group_df_copy = group_df_copy.loc[group_df_copy.session.isin(incl_sessions)]

        group_dict[group] = group_df_copy

    ############################## perform linear mixed effects model ##############################
    for g, group in enumerate(channel_group):
        data_analysis = group_dict[group]
        predictor_x = data_analysis.session.values
        outcome_y = data_analysis[f"{data_to_fit}"]

        predictor_polynomial = sm.add_constant(
            np.column_stack([predictor_x, predictor_x**2])
        )  # adjust the degree as needed
        # np.column_stack: creates a 2D array with stacked x values and a second column with stacked x**2 values
        # np.add_constant: adds a column with ones as a constant term, this allows the model to estimate an intercept which represents the predicted outcome value when the predictor variable is zero

        # fit the model
        md = sm.OLS(outcome_y, predictor_polynomial)
        mdf = md.fit()

        # save linear model result
        print(mdf.summary())
        mdf_result[group] = mdf

        result_dict[group] = [
            group,
            mdf.aic,
            mdf.bic,
            mdf.rsquared,
            mdf.fvalue,  # f-statistic
            mdf.f_pvalue,  # prob (F-statistic)
            mdf.pvalues[0],  # pvalues const
            mdf.pvalues[1],  # pvalues x1 = for linear term for predictor
            mdf.pvalues[2],  # pvalues x2 = for squared term for predictor
            mdf.params[0],  # coefficient const
            mdf.params[1],  # coefficient x1
            mdf.params[2],  # coefficient x2
        ]

        # add predictions column to dataframe
        yp = mdf.fittedvalues
        group_dict[group]["predictions"] = yp

        for ses in incl_sessions:
            ses_data = data_analysis.loc[data_analysis.session == ses]
            count = ses_data.subject_hemisphere.count()

            # get mean of predictions
            mean_yp = np.mean(ses_data.predictions)
            std_yp = np.std(ses_data.predictions)
            sem_yp = stats.sem(ses_data.predictions)

            # save sample size
            prediction_result[f"{group}_{ses}mfu"] = [group, ses, count, mean_yp, std_yp, sem_yp]

    prediction_result_df = pd.DataFrame(prediction_result)
    prediction_result_df.rename(
        index={0: "channel_group", 1: "session", 2: "count", 3: "mean_yp", 4: "std_yp", 5: "sem_yp"}, inplace=True
    )
    prediction_result_df = prediction_result_df.transpose()

    result_dict_df = pd.DataFrame(result_dict)
    result_dict_df.rename(
        index={
            0: "channel_group",
            1: "aic",
            2: "bic",
            3: "rsquared",
            4: "fvalue",
            5: "f_pvalue",
            6: "p_value_const",
            7: "p_value_linear",
            8: "p_value_squared",
            9: "coef_const",
            10: "coef_linear",
            11: "coef_squared",
        },
        inplace=True,
    )
    result_dict_df = result_dict_df.transpose()

    ############################## PLOT THE RESULT ##############################
    fig_1, axes_1 = plt.subplots(3, 1, figsize=(10, 15))
    fig_2, axes_2 = plt.subplots(3, 1, figsize=(10, 15))

    for g, group in enumerate(channel_group):
        data_analysis = group_dict[group]

        mdf_group = mdf_result[group]  # = md.fit()

        ############################## Fig 1 : plot the polynomial regression model ##############################
        ############################## Fig 2 : plot the residuals (difference between observed and predicted outcomes) ##############################
        # one subplot per channel group
        axes_1[g].set_title(f"{group} channel group", fontdict=fontdict)
        axes_2[g].set_title(f"{group} channel group", fontdict=fontdict)

        # plot the real observed values for each electrode
        for id, group_id in enumerate(data_analysis.group.unique()):
            sub_data = data_analysis[data_analysis.group == group_id]

            axes_1[g].scatter(
                sub_data["session"], sub_data[f"{data_to_fit}"], color=plt.cm.twilight_shifted((id + 1) * 10), alpha=0.3
            )  # color=plt.cm.tab20(group_id)
            axes_1[g].plot(
                sub_data["session"],
                sub_data[f"{data_to_fit}"],
                color=plt.cm.twilight_shifted((id + 1) * 10),
                linewidth=1,
                alpha=0.3,
            )

        # plot the model regression line
        if 0 in incl_sessions:
            if 18 in incl_sessions:
                x = np.arange(0, 19)
                x = sm.add_constant(np.column_stack([x, x**2]))

            else:
                x = np.arange(0, 4)
                x = sm.add_constant(np.column_stack([x, x**2]))

        elif 0 not in incl_sessions:
            x = np.arange(3, 19)
            x = sm.add_constant(np.column_stack([x, x**2]))

        # get prediction values
        pred = mdf_group.get_prediction(x)
        y_predict = pred.predicted_mean
        # y_predict = mdf_group.predict(x)
        conf_int = pred.conf_int(alpha=0.05)  # 95% confidence interval

        # from 0 to 18
        x_range = np.arange(0, 19)

        # plot the curved model line
        axes_1[g].plot(x_range, y_predict, color="k", linewidth=5)
        axes_1[g].fill_between(x_range.flatten(), conf_int[:, 0], conf_int[:, 1], color="lightgray", alpha=0.5)

        # plot the residuals: number of residuals equals the number of observations (number of channels I input)
        resid = mdf_group.resid
        predicted_values = mdf_group.fittedvalues
        axes_2[g].scatter(predicted_values, resid, color="k", alpha=0.2)
        axes_2[g].axhline(y=0, color="red", linestyle="--")

    for ax in axes_1:
        ax.set_ylabel(f"{data_to_fit}", fontsize=25)
        ax.set_xlabel("months post-surgery", fontsize=25)
        ax.set_xlim([-0.5, 19])

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_1.suptitle(f"Polynomial regression model: {highest_beta_session} beta channels", fontsize=30)
    fig_1.subplots_adjust(wspace=0, hspace=0)
    fig_1.tight_layout()
    fig_1.savefig(
        os.path.join(
            figures_path, f"ols_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.png"
        ),
        bbox_inches="tight",
    )
    fig_1.savefig(
        os.path.join(
            figures_path, f"ols_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.svg"
        ),
        bbox_inches="tight",
        format="svg",
    )

    for ax in axes_2:
        ax.set_xlabel("Predicted Values", fontsize=25)
        ax.set_ylabel("Residuals", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_2.suptitle(
        f"Polynomial regression model ({data_to_fit}) residuals: {highest_beta_session} beta channels", fontsize=30
    )
    fig_2.subplots_adjust(wspace=0, hspace=0)
    fig_2.tight_layout()
    fig_2.savefig(
        os.path.join(
            figures_path,
            f"ols_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.png",
        ),
        bbox_inches="tight",
    )
    fig_2.savefig(
        os.path.join(
            figures_path,
            f"ols_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    print("figure: ", f"ols_{data_to_fit}_{highest_beta_session}_beta_channels.png", "\nwritten in: ", figures_path)

    # save results
    mdf_result_filepath = os.path.join(
        results_path, f"fooof_ols_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle"
    )
    with open(mdf_result_filepath, "wb") as file:
        pickle.dump(mdf_result, file)

    print(
        "file: ",
        f"fooof_ols_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle",
        "\nwritten in: ",
        results_path,
    )

    return {
        "group_dict": group_dict,
        "mdf_result": mdf_result,
        "mdf": mdf,
        "prediction_result_df": prediction_result_df,
        "ypredict": y_predict,
        "x": x,
        "predictor_polynomial": predictor_polynomial,
    }
