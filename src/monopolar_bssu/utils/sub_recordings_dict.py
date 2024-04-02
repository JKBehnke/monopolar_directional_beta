""" Subjects with externalized or postop Percept recordings """

sub_bids_id_dict = {
    "024": "noBIDS24",
    "025": "L001",
    "028": "noBIDS28", # no beta in externalized both hemispheres
    "029": "noBIDS29", # very low beta externalized RSTN, Artifacts in LSTN, outlier for CV
    "030": "EL007", # no beta LSTN ext
    "031": "L002", # no Med OFF in externalized
    "032": "L003", # no beta LSTN ext
    "047": "L007",
    "048": "noBIDS48", # 2C fehlt Right, no beta
    "049": "noBIDS49",
    "052": "EL014", # no beta LSTN ext half of the recording thick, the other thin. Not sure what this is
    "056": "noBIDS56",
    "059": "EL016",
    "061": "L010", # no beta RSTN ext
    "064": "L012",
    "067": "EL017",
    "069": "L013", # 69 L 01 error: window is longer than input signal
    "071": "L014", # no beta RSTN ext
    "072": "L015",
    "075": "EL019",
    "077": "L016",
    "079": "L017", # no beta RSTN ext
    "080": "EL021",

    # n=22

}

# no beta either in externalized or percept -> exclude for comparison externalized vs percept
EXCLUDED_NO_BETA_EXT_OR_PERCEPT = [
    "028_Right",
    "028_Left",
    "029_Right",
    "029_Left",
    "030_Left",
    "032_Right",
    "032_Left",
    "048_Right",
    "048_Left",
    "049_Right",
    "049_Left",
    "052_Left",
    "056_Left",
    "061_Right",
    "071_Right",
    "072_Left",
    "075_Right",
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
    "069_Left", # zu viele Bewegungsartifakte, too short for 20sec analysis
    "071_Right",
    "072_Right",
    #"072_Left",
    "075_Right", # beta there initially but gone after re-ref
    "077_Left",
]

# no beta only in percept postop -> exclude for comparison percept postop vs percept postop
EXCLUDED_NO_BETA_PERCEPT = [
    "029_Right",
    "029_Left",
    "032_Right",
    "048_Right",
    "048_Left",
    "049_Left",
    "056_Left",
    "061_Right",
    "071_Right",
    "075_Right",
]

# watch out, there's another exclude patients function in the monopol_method_comparison file!
def exclude_patients(rec:str):
    """
    Input:
        rec: str, "externalized", "percept", "ext_or_percept"
    Exclude patients with bad recordings or no beta
        - 069_Left: recording too short because too many artifacts were taken out
    """

    if rec == "externalized":
        exclude_patients_list = EXCLUDED_NO_BETA_EXT
    
    elif rec == "percept":
        exclude_patients_list = EXCLUDED_NO_BETA_PERCEPT
    
    elif rec == "ext_or_percept":
        exclude_patients_list = EXCLUDED_NO_BETA_EXT_OR_PERCEPT

    return exclude_patients_list




def get_bids_id_from_sub(sub:str):
    """
    """
    return sub_bids_id_dict[sub]

def get_sub_from_bids_id(bids_id:str):
    """
    """
    return [sub for sub, bids in sub_bids_id_dict.items() if bids == bids_id][0]

def get_bids_id_all_list():
    """
    """
    return list(sub_bids_id_dict.values())