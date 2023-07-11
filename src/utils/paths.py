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
DIST_PATH = DAT_PATH / "distributions"
BAYES_PATH = FINAL_PATH / "bayes"
HMDA_PATH = DAT_PATH / "hmda"

PAPER_PATH = Path(__file__).parents[2] / "paper"
TBL_PATH = PAPER_PATH / "tables"
FIG_PATH = PAPER_PATH / "figures"
