""" Illustrate how many maximal beta contacts were clinically active at 12 months """

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



def barplot_structure_maximal_beta_contacts_active(
    bssu_version: str,
    clinical_session: str,
    percept_session: str
):
    """
    Barplot of the rank validation method: use for comparing externalized_fooof to the three percept methods
    3 methods: "JLB_directional", "euclidean_directional", "detec_strelow_contacts"

    Input:
        - value_to_plot: str e.g.
            "percentage_at_least_one_same_contact_rank_1_and_2" must be a column in the sample size result Excel file
            "percentage_both_contacts_matching",
            "percentage_same_rank_1"
            
        - clinical_session: "fu12m"
        - percept_session: "fu12m"
        - new_reference: "one_to_zero_two_to_three"

        - bssu_version: "percept" or "externalized"

    """
    questions = ["percentage_at_least_one_same_contact_rank_1_and_2",
                 "percentage_both_contacts_matching",
                 "percentage_same_rank_1"]
    
    methods_to_compare = [
        "externalized_fooof",
        "euclidean_directional",
        "JLB_directional",
        "detec_strelow_contacts",
        ]
    
    data_result_dict = {}
    data_result_dict["Questions"] = questions 

    if bssu_version == "percept":
        external_extension = ""

    elif bssu_version == "externalized":
        external_extension = "_externalized_bssu"

    ext_fooof_re_ref = "one_to_zero_two_to_three_"
    
    for method in methods_to_compare:

        if method == "externalized_fooof":
            method_name = method
        
        else:
            method_name = f"{method}{external_extension}"

        # if method == "detec_strelow_contacts":
        #     comparison_methods = f"externalized_fooof_{method_name}"
        
        # else:
        comparison_methods = f"best_clinical_contacts_{method_name}"

        # load the comparison DF 
        load_data = monopol_comparison_tests.load_comparison_result_DF(
            method_comparison= comparison_methods, 
            comparison_file="rank", 
            clinical_session=clinical_session, 
            percept_session=percept_session, 
            fooof_version="v2", 
            bssu_version=bssu_version,
            new_reference="one_to_zero_two_to_three"
        )

        # extract data and save into group dict
        questions_results = [] # list per method with three values (questions 1-3)
        for q in questions:
            
            q_result = load_data[q].values[0]
            questions_results.append(q_result)
        

        data_result_dict[method] = questions_results

    # transform final dictionary to Dataframe
    data_results_df = pd.DataFrame(data_result_dict)

    return data_results_df

def barplot_method_validation_maximal_contacts(
    bssu_version: str,
    clinical_session: str,
    percept_session: str
):
    """ 
    PLOT BARPLOT:
        - For each method plot three bars in different colors: 
            answering the questions: ["percentage_at_least_one_same_contact_rank_1_and_2",
                 "percentage_both_contacts_matching",
                 "percentage_same_rank_1"]

        - Validation of methods (percept or externalized) vs. externalized_fooof ("real" beta)
    """

    loaded_data = barplot_structure_maximal_beta_contacts_active(
        bssu_version=bssu_version,
        clinical_session=clinical_session,
        percept_session=percept_session
    )
    # rows=questions, columns=methods

    # structure data for seaborn
    data_long = pd.melt(loaded_data, id_vars='Questions', var_name='method', value_name='Value')

    # Plot each bar
    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    sns.barplot(x='method', y='Value', hue='Questions', data=data_long)

    plt.ylim(0, 1)

    # Add labels and title
    plt.xlabel('Method')
    plt.ylabel('Percentage of hemispheres')
    plt.title(f'Hemispheres with clinically active maximal beta contacts, {bssu_version}')

    fig.tight_layout()

    io_externalized.save_fig_png_and_svg(path=GROUP_FIGURES_PATH,
                                         filename=f"Barplot_maximal_beta_contacts_matching_{bssu_version}_bssu_{percept_session}_vs_{clinical_session}_active_contacts",
                                         figure=fig)