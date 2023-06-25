from pathlib import Path

BASE_PATH = Path(__file__).parents[1]
DAT_PATH = BASE_PATH / "data"
PPP_PATH = DAT_PATH / "ppp"
L2_PATH = DAT_PATH / "l2"
CENSUS_PATH = DAT_PATH / "census"
CW_PATH = DAT_PATH / "crosswalks"
FINAL_PATH = DAT_PATH / "final"
RETH_PATH = FINAL_PATH / "rethnicity"
ETHNICOLR_PATH = FINAL_PATH / "ethnicolr"
SURGEO_PATH = FINAL_PATH / "surgeo"
