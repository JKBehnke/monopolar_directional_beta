import os
import numpy as np
import pandas as pd
import sys


from ..utils import loadResults as loadResults


def find_project_folder():
    """
    find_project_folder is a function to find the folder "PyPerceive_Project" on your local computer

    Return: tuple[str, str] -> project_path, data_path
    to only use one str use _ -> example: project_folder, _ = find_project_folder()
    """

    # from the cwd get path to PyPerceive_Project (=Git Repository)
    jennifer_user_path = os.getcwd()
    while jennifer_user_path[-14:] != "jenniferbehnke":
        jennifer_user_path = os.path.dirname(jennifer_user_path)

    project_path = os.path.join(
        jennifer_user_path,
        "Dropbox",
        "work",
        "ResearchProjects",
        "BetaSenSightLongterm",
    )

    results_path = os.path.join(project_path, "results")

    return project_path, results_path


def get_onedrive_path(folder: str = "onedrive", sub: str = None):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'Percept_Data_structured', 'sourcedata']
    """

    folder_options = ["onedrive", "sourcedata"]

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(
            f"given folder: {folder} is incorrect, " f"should be {folder_options}"
        )

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != "Users":
        path = os.path.dirname(path)  # path is now leading to Users/username

    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [
        f
        for f in os.listdir(path)
        if np.logical_and("onedrive" in f.lower(), "charit" in f.lower())
    ]

    path = os.path.join(path, onedrive_f[0])  # path is now leading to Onedrive folder

    # add the folder DATA-Test to the path and from there open the folders depending on input folder
    datapath = os.path.join(path, "Percept_Data_structured")
    if folder == "onedrive":
        return datapath

    elif folder == "sourcedata":
        return os.path.join(datapath, "sourcedata")

    # elif folder == 'results': # must be data or figures
    #     return os.path.join(datapath, 'results')

    # elif folder == "raw_perceive": # containing all relevant perceive .mat files
    #     return os.path.join(datapath, "sourcedata", f"sub-{sub}", "raw_perceive")


def get_onedrive_path_mac(folder: str = "onedrive", sub: str = None):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'Percept_Data_structured', 'sourcedata']
    """

    folder_options = ["onedrive", "sourcedata"]

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(
            f"given folder: {folder} is incorrect, " f"should be {folder_options}"
        )

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != "Users":
        path = os.path.dirname(path)  # path is now leading to Users/username

    # get the onedrive folder containing "charit" and add it to the path

    path = os.path.join(path, "Charité - Universitätsmedizin Berlin")

    # onedrive_f = [
    #     f for f in os.listdir(path) if np.logical_and(
    #         'onedrive' in f.lower(),
    #         'shared' in f.lower())
    #         ]
    # print(onedrive_f)

    # path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder

    # add the folder DATA-Test to the path and from there open the folders depending on input folder
    datapath = os.path.join(
        path, "AG Bewegungsstörungen - Percept - Percept_Data_structured"
    )
    if folder == "onedrive":
        return datapath

    elif folder == "sourcedata":
        return os.path.join(datapath, "sourcedata")

    # elif folder == 'results': # must be data or figures
    #     return os.path.join(datapath, 'results')

    # elif folder == "raw_perceive": # containing all relevant perceive .mat files
    #     return os.path.join(datapath, "sourcedata", f"sub-{sub}", "raw_perceive")


############## PyPerceive Repo: add to dev, after pulling ##############
# check if 'Charité - Universitätsmedizin Berlin' is in directory
# if 'Charité - Universitätsmedizin Berlin' in os.listdir(path):

#     path = os.path.join(path, 'Charité - Universitätsmedizin Berlin')

#     # add the folder DATA-Test to the path and from there open the folders depending on input folder
#     datapath = os.path.join(path, 'AG Bewegungsstörungen - Percept - Percept_Data_structured')
#     if folder == 'onedrive':
#         return datapath

#     elif folder == 'sourcedata':
#         return os.path.join(datapath, 'sourcedata')

# else:
#     # get the onedrive folder containing "onedrive" and "charit" and add it to the path
#     onedrive_f = [
#         f for f in os.listdir(path) if np.logical_and(
#             'onedrive' in f.lower(),
#             'charit' in f.lower())
#             ]

#     path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder


#     # add the folder DATA-Test to the path and from there open the folders depending on input folder
#     path = os.path.join(path, 'Percept_Data_structured')
#     if folder == 'onedrive':

#         assert os.path.exists(path), f'wanted path ({path}) not found'

#         return path

#     elif folder == 'sourcedata':

#         path = os.path.join(path, 'sourcedata')
#         if sub: path = os.path.join(path, f'sub-{sub}')

#         assert os.path.exists(path), f'wanted path ({path}) not found'

#         return path


def get_onedrive_path_externalized_bids(folder: str = "onedrive", sub: str = None):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'sourcedata', 'rawdata', 'derivatives',
        'sourcedata_sub', 'rawdata_sub',
        ]
    """

    folder_options = [
        "onedrive",
        "sourcedata",
        "rawdata",
        "derivatives",
        "sourcedata_sub",
        "rawdata_sub",
    ]

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(
            f"given folder: {folder} is incorrect, " f"should be {folder_options}"
        )

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != "Users":
        path = os.path.dirname(path)  # path is now leading to Users/username

    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [
        f
        for f in os.listdir(path)
        if np.logical_and("onedrive" in f.lower(), "charit" in f.lower())
    ]

    path = os.path.join(path, onedrive_f[0])  # path is now leading to Onedrive folder

    # add the BIDS folder to the path and from there open the folders depending on input folder
    datapath = os.path.join(path, "BIDS_01_Berlin_Neurophys")
    if folder == "onedrive":
        return datapath

    elif folder == "sourcedata":
        return os.path.join(datapath, "sourcedata")

    elif folder == "rawdata":
        return os.path.join(datapath, "rawdata")

    elif folder == "derivatives":
        return os.path.join(datapath, "derivatives")

    elif folder == "sourcedata_sub":
        return os.path.join(datapath, "sourcedata", f"sub-{sub}")

    elif folder == "rawdata_sub":
        local_path = get_monopolar_project_path(folder="data")
        patient_metadata = pd.read_excel(
            os.path.join(local_path, "patient_metadata.xlsx"),
            keep_default_na=True,
            sheet_name="patient_metadata",
        )

        # change column "patient_ID" to strings
        patient_metadata["patient_ID"] = patient_metadata.patient_ID.astype(str)

        sub_BIDS_ID = patient_metadata.loc[
            patient_metadata.patient_ID == sub
        ]  # row of subject

        # check if the subject has a BIDS key
        if pd.isna(sub_BIDS_ID.BIDS_key.values[0]):
            print(f"The subject {sub} has no BIDS key yet.")
            return "no BIDS key"

        else:
            sub_BIDS_ID = sub_BIDS_ID.BIDS_key.values[0]  # externalized ID

            sub_folders = os.listdir(os.path.join(datapath, "rawdata"))
            # check if externalized ID is in the directory
            folder_name = []
            for folder in sub_folders:
                if sub_BIDS_ID in folder:
                    folder_name.append(folder)

            # check if the corresponding BIDS key has a folder in the directory
            if len(folder_name) == 0:
                print(f"The subject {sub} has no BIDS folder yet in {datapath}")

            else:
                sub_path = os.path.join(datapath, "rawdata", folder_name[0])
                return sub_path


def get_local_path(folder: str, sub: str = None):
    """
    find_project_folder is a function to find the folder "Longterm_beta_project" on your local computer

    Input:
        - folder: str
            'Research': path to Research folder
            'Longterm_beta_project': path to Project folder
            'GroupResults': path to results folder, without going in subject level
            'results': subject folder of results
            'GroupFigures': path to figures folder, without going in subject level
            'figures': figure folder of results

        - sub: str, e.g. "029"


    """

    folder_options = [
        "Project",
        "GroupResults",
        "results",
        "GroupFigures",
        "figures",
        "data",
    ]

    # Error checking, if folder input is in folder options
    # if folder.lower() not in folder_options:
    # raise ValueError(
    #     f'given folder: {folder} is incorrect, '
    #     f'should be {folder_options}')

    # from your cwd get the path and stop at 'Users'
    jennifer_user_path = os.getcwd()
    while jennifer_user_path[-14:] != "jenniferbehnke":
        jennifer_user_path = os.path.dirname(jennifer_user_path)

    project_path = os.path.join(
        jennifer_user_path,
        "Dropbox",
        "work",
        "ResearchProjects",
        "BetaSenSightLongterm",
    )

    # add the folder to the path and from there open the folders depending on input folder
    if folder == "Project":
        return project_path

    elif folder == "GroupResults":
        return os.path.join(project_path, "results")

    elif folder == "results":
        return os.path.join(project_path, "results", f"sub-{sub}")

    elif folder == "GroupFigures":
        return os.path.join(project_path, "figures")

    elif folder == "figures":
        return os.path.join(project_path, "figures", f"sub-{sub}")

    elif folder == "data":
        return os.path.join(project_path, "data")


def get_monopolar_project_path(folder: str, sub: str = None):
    """
    find_project_folder is a function to find the folder "Longterm_beta_project" on your local computer

    Input:
        - folder: str
            'Research': path to Research folder
            'Longterm_beta_project': path to Project folder
            'GroupResults': path to results folder, without going in subject level
            'results': subject folder of results
            'GroupFigures': path to figures folder, without going in subject level
            'figures': subject folder of figures

        - sub: str, e.g. "EL001" or "L010"


    """

    folder_options = [
        "Project",
        "GroupResults",
        "results",
        "GroupFigures",
        "figures",
        "data",
        "data_sub",
    ]

    # Error checking, if folder input is in folder options
    # if folder.lower() not in folder_options:
    # raise ValueError(
    #     f'given folder: {folder} is incorrect, '
    #     f'should be {folder_options}')

    # from your cwd get the path and stop at 'Users'
    jennifer_user_path = os.getcwd()
    while jennifer_user_path[-14:] != "jenniferbehnke":
        jennifer_user_path = os.path.dirname(jennifer_user_path)

    project_path = os.path.join(
        jennifer_user_path,
        "Dropbox",
        "work",
        "ResearchProjects",
        "Monopolar_power_estimation",
    )

    # add the folder to the path and from there open the folders depending on input folder
    if folder == "Project":
        return project_path

    elif folder == "GroupResults":
        return os.path.join(project_path, "results")

    elif folder == "results":
        return os.path.join(project_path, "results", f"sub-{sub}")

    elif folder == "GroupFigures":
        return os.path.join(project_path, "figures")

    elif folder == "figures":
        return os.path.join(project_path, "figures", f"sub-{sub}")

    elif folder == "data":
        return os.path.join(project_path, "data")

    elif folder == "data_sub":
        patient_metadata = pd.read_excel(
            os.path.join(project_path, "data", "patient_metadata.xlsx"),
            keep_default_na=True,
            sheet_name="patient_metadata",
        )

        # change column "patient_ID" to strings
        patient_metadata["patient_ID"] = patient_metadata.patient_ID.astype(str)

        sub_externalized_ID = patient_metadata.loc[
            patient_metadata.patient_ID == sub
        ]  # row of subject
        sub_externalized_ID = sub_externalized_ID.externalized_ID.values[
            0
        ]  # externalized ID

        sub_folders = os.listdir(os.path.join(project_path, "data", "externalized_lfp"))
        # check if externalized ID is in the directory
        for folder in sub_folders:
            if sub_externalized_ID in folder:
                folder_name = folder

        # if folder_name not in locals():
        #     print(f"subject {sub} not in data. Check, if this subject has a folder in data")

        sub_path = os.path.join(project_path, "data", "externalized_lfp", folder_name)
        return sub_path


def get_patterned_dbs_project_path(folder: str, sub: str = None):
    """ """
    # from your cwd get the path and stop at 'Users'
    jennifer_user_path = os.getcwd()
    while jennifer_user_path[-14:] != "jenniferbehnke":
        jennifer_user_path = os.path.dirname(jennifer_user_path)

    project_path = os.path.join(
        jennifer_user_path,
        "Dropbox",
        "work",
        "ResearchProjects",
        "Patterned_stimulation_project",
    )

    folder_paths = {
        "Project": project_path,
        "GroupResults": os.path.join(project_path, "results"),
        "results": os.path.join(project_path, "results", f"sub-{sub}"),
        "GroupFigures": os.path.join(project_path, "figures"),
        "figures": os.path.join(project_path, "figures", f"sub-{sub}"),
        "data": os.path.join(project_path, "data"),
        "sub_data": os.path.join(project_path, "data", f"sub-{sub}"),
    }

    # add the folder to the path and from there open the folders depending on input folder
    return folder_paths[folder]


def chdir_repository(repository: str):
    """
    repository: "Py_Perceive", "meet", "BetaSenSightLongterm"

    """

    #######################     USE THIS DIRECTORY FOR IMPORTING PYPERCEIVE REPO  #######################

    # create a path to the BetaSenSightLongterm folder
    # and a path to the code folder within the BetaSenSightLongterm Repo
    jennifer_user_path = os.getcwd()
    while jennifer_user_path[-14:] != "jenniferbehnke":
        jennifer_user_path = os.path.dirname(jennifer_user_path)

    repo_dict = {
        "Py_Perceive": os.path.join(
            jennifer_user_path, "code", "PyPerceive_project", "PyPerceive", "code"
        ),
        "meet": os.path.join(jennifer_user_path, "code", "meet_repository", "meet"),
        "BetaSenSightLongterm": os.path.join(
            jennifer_user_path, "code", "BetaSenSightLongterm", "BetaSenSightLongterm"
        ),
        "monopolar_directional_beta": os.path.join(
            jennifer_user_path,
            "code",
            "Monopolar_power_estimation",
            "monopolar_directional_beta",
        ),
    }

    # directory to PyPerceive code folder
    project_path = repo_dict[repository]
    sys.path.append(project_path)

    # # change directory to PyPerceive code path within BetaSenSightLongterm Repo
    os.chdir(project_path)

    return os.getcwd()
