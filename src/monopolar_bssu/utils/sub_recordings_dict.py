""" Subjects with externalized or postop Percept recordings """

sub_bids_id_dict = {
    "024": "noBIDS24",
    "025": "L001",
    "028": "noBIDS28",
    "029": "noBIDS29",
    "030": "EL007",
    "031": "L002", # no figures
    "032": "L003",
    "047": "L007",
    "048": "noBIDS48", # 2C fehlt Right
    "049": "noBIDS49",
    "052": "EL014", # half of the recording thick, the other thin. Not sure what this is
    "056": "noBIDS56",
    "059": "EL016",
    "061": "L010",
    "064": "L012",
    "067": "EL017",
    "069": "L013", # 69 L 01 error: window is longer than input signal
    "071": "L014",
    "072": "L015",
    "075": "EL019",
    "077": "L016",
    "079": "L017",
    "080": "EL021",

}


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