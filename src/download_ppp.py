# Imports

import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup

from utils.paths import PPP_PATH

# Globals

PPP_DATA_URL = "https://data.sba.gov/dataset/ppp-foia"


def get_ppp_data_links():
    soup = BeautifulSoup(requests.get(PPP_DATA_URL).content, features="html.parser")
    resource_list: list[BeautifulSoup] = (
        soup.find("section", {"id": "dataset-resources"})
        .find("ul", {"class": "resource-list"})
        .find_all("li", {"class": "resource-item"})
    )

    base_url = "https://data.sba.gov/dataset/8aa276e2-6cab-4f86-aca4-a7dde42adf24"
    url_list = []
    for resource in resource_list:
        data = resource.find("a")
        title = data["title"]

        if "public" in title:
            url = "/".join(data["href"].split("/")[-2:])
            complete_url = f"{base_url}/{url}/download/{title}"
            url_list.append(complete_url)

    return url_list


def _read_data(url: str) -> pd.DataFrame:
    cols = [
        "borrowername",
        "race",
        "ethnicity",
        "borrowerstate",
        "borrowerzip",
        "businesstype",
        "dateapproved",
    ]
    df = pd.read_csv(
        url,
        usecols=lambda x: x.lower() in cols,
    )
    df.columns = df.columns.str.lower()

    return df


def read_all_data(urls: list[str]) -> pd.DataFrame:
    dfs = []
    for url in tqdm.tqdm(urls):
        dfs.append(_read_data(url))

    return pd.concat(dfs)


def main():
    url_list = get_ppp_data_links()
    df = read_all_data(url_list)
    df.to_parquet(PPP_PATH / "ppp_raw.parquet")


if __name__ == "__main__":
    main()
