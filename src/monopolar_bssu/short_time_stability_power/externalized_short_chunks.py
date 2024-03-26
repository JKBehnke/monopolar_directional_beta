""" Analysis of 20 sec chunks of 2 min externalized recordings """


import os
import pickle

import fooof
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof.plts.spectra import plot_spectrum

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io_externalized as io_externalized
from ..utils import loadResults as loadResults
from ..utils import sub_recordings_dict as sub_recordings_dict
from ..utils import externalized_lfp_preprocessing as externalized_lfp_preprocessing
from ..externalized_lfp import feats_ssd as feats_ssd

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

# patient_metadata = load_data.load_patient_metadata_externalized()
PATIENT_METADATA = io_externalized.load_excel_data(filename="patient_metadata")
HEMISPHERES = ["Right", "Left"]
DIRECTIONAL_CONTACTS = ["1A", "1B", "1C", "2A", "2B", "2C"]
BSSU_CHANNELS = [
    "01",
    "02",
    "03",
    "12",
    "13",
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

FILTER = {
    "notch_and_band_pass_filtered": "filtered_lfp_250Hz",
    "unfiltered": "lfp_resampled_250Hz",
    "only_high_pass_filtered": "only_high_pass_lfp_250Hz",
}

