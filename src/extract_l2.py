import shutil
from pathlib import Path

import pandas as pd
import tqdm
from utils.paths import L2_PATH


def read_file(f):
    shutil.unpack_archive(f, "tmp")

    for file in Path("tmp").glob("*"):
        if (
            file.suffix == ".tab"
            and "DEMOGRAPHIC" in file.stem
            and "__MACOSX" not in file.as_posix()
        ):
            file = file
            break

    cols = {
        "Voters_FirstName",
        "Voters_MiddleName",
        "Voters_LastName",
        "Voters_NameSuffix",
        "Residence_Addresses_State",
        "Residence_Addresses_Zip",
        "Residence_Addresses_ZipPlus4",
        "Residence_Addresses_CensusTract",
        "Residence_Addresses_CensusBlockGroup",
        "Residence_Addresses_CensusBlock",
        "Ethnic_Description",
        "EthnicGroups_EthnicGroup1Desc",
        "CountyEthnic_LALEthnicCode",
        "CountyEthnic_Description",
        "Voters_FIPS",
        "County",
    }

    csv = pd.read_csv(file, sep="\t", usecols=cols, encoding="latin-1", dtype=str)
    csv.to_parquet(L2_PATH / f"intermediate/{file.name}.parquet", index=False)

    shutil.rmtree("tmp")


def main():
    files = list((L2_PATH / "raw").glob("*.zip"))
    for file in tqdm.tqdm(files):
        read_file(file)


if __name__ == "__main__":
    main()
