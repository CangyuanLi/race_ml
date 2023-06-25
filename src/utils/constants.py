import string

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RACES = ("asian", "black", "hispanic", "white")
RACES_DICT = {r: i for i, r in enumerate(RACES)}

VALID_NAME_CHARS = f"{string.ascii_lowercase} '-"
VALID_NAME_CHARS_DICT = {c: i for i, c in enumerate(VALID_NAME_CHARS, start=1)}
VALID_NAME_CHARS_LEN = len(VALID_NAME_CHARS) + 1
