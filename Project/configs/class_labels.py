"""
Class Labels Configuration
--------------------------
Centralized class label definitions for EuroSAT (10 classes) and UC Merced (21 classes).
"""

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

UC_MERCED_CLASSES = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
]


def get_class_labels(dataset_name: str = "eurosat") -> list[str]:
    """
    Retrieve class label list for the specified dataset name.

    Args:
        dataset_name (str): 'eurosat' or 'uc_merced'.

    Returns:
        list[str]: Class label strings.
    """
    key = str(dataset_name).lower().strip()
    if key in ["uc_merced", "ucmerced", "uc-merced", "uc_merced_model"]:
        return UC_MERCED_CLASSES
    return EUROSAT_CLASSES
