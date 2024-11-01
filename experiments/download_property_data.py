import argparse
import os
from datetime import datetime

import wget
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf


URL = "prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
YEARS = np.arange(1995, 2023)


def get_latitudes_longitudes(postcodes: str) -> tuple:
    try:
        response = requests.post(
            "https://api.postcodes.io/postcodes",
            json={"postcodes": list(postcodes)},
        ).json()

    except Exception:
        print("Error with post request, returning Nones...")
        return [None] * len(postcodes), [None] * len(postcodes)

    if response["status"] != 200:
        print(f"Response status {response['status']}, returning Nones...")
        return [None] * len(postcodes), [None] * len(postcodes)

    latitudes = [
        result["result"]["latitude"] if result["result"] is not None else None
        for result in response["result"]
    ]
    longitudes = [
        result["result"]["longitude"] if result["result"] is not None else None
        for result in response["result"]
    ]

    return latitudes, longitudes


def add_coordinates_to_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with postcodes of type str only
    dataframe = dataframe[dataframe.iloc[:, 3].apply(type) == str]
    postcodes = dataframe.iloc[:, 3].values

    # Split postcodes into 200 chunks to avoid overloading the API
    postcode_splits = np.array_split(postcodes, len(postcodes) // 100 + 1)

    latitudes = []
    longitudes = []

    for split in tqdm(
        postcode_splits,
        desc="Getting latitudes and longitudes",
    ):
        lat, lon = get_latitudes_longitudes(split)
        latitudes.extend(lat)
        longitudes.extend(lon)

    dataframe[len(dataframe.columns)] = latitudes
    dataframe[len(dataframe.columns)] = longitudes

    return dataframe


# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    type=str,
    default="./_data",
    help="Directory to download data to.",
)

parser.add_argument(
    "--small-dataset",
    action="store_true",
    default=False,
)

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

if not os.path.exists(f"{args.data_dir}/raw"):
    os.makedirs(f"{args.data_dir}/raw")

if not os.path.exists(f"{args.data_dir}/processed"):
    os.makedirs(f"{args.data_dir}/processed")

# Download data
for year in YEARS:
    if os.path.exists(f"{args.data_dir}/raw/pp-{year}.csv"):
        print(f"Data for {year} already exists, skipping...")
        continue

    print(f"\nDownloading data for {year}...")
    wget.download(f"http://{URL}/pp-{year}.csv", out=f"{args.data_dir}/raw")

print("\nFinished getting data, loading...")
dataframes = [
    pd.read_csv(f"{args.data_dir}/raw/pp-{year}.csv", header=None)
    for year in tqdm(YEARS)
]

print("\nFinished loading, processing...")
dataframes = [
    add_coordinates_to_dataframe(
        dataframe[::10] if args.small_dataset else dataframe
    )
    for dataframe in tqdm(dataframes, desc="Processing dataframes")
]

dataframes = [dataframe.dropna(subset=[16, 17]) for dataframe in dataframes]

print("\nFinished processing, saving individual dataframes...")
name = "small" if args.small_dataset else "all"
for dataframe, year in zip(dataframes, YEARS):
    dataframe.to_csv(
        f"{args.data_dir}/processed/{name}-{year}.csv",
        index=False,
    )

print("\nSaving aggregate dataframe...")
dataframe = pd.concat(dataframes)
dataframe.to_csv(f"{args.data_dir}/processed/inter-{name}.csv", index=False)


# Read dates and add integer representing number of days since earliest day
print("\nAdding days since first day...")
d0 = datetime.strptime(min(dataframe.iloc[:, 2]), "%Y-%m-%d %H:%M")
days = dataframe.iloc[:, 2].map(
    lambda d: (datetime.strptime(d, "%Y-%m-%d %H:%M") - d0).days
)

# Creating postprocessed dataframe
dataframe = pd.DataFrame(
    data={
        "lat": dataframe.iloc[:, -2].astype(np.float32),
        "lon": dataframe.iloc[:, -1].astype(np.float32),
        "price": np.log10(dataframe.iloc[:, 1]).astype(np.float32),
        "type": dataframe.iloc[:, 4].astype(str),
        "new": dataframe.iloc[:, 5].astype(str),
        "lease": dataframe.iloc[:, 6].astype(str),
        "day": days,
        "date": dataframe.iloc[:, 2].astype(str),
        "postcode": dataframe.iloc[:, 3].astype(str),
    },
)
dataframe.to_csv(f"{args.data_dir}/processed/{name}.csv", index=False)


ax = plt.axes(projection=ccrs.Mercator())
ax.scatter(
    dataframe.iloc[:, -1],
    dataframe.iloc[:, -2],
    transform=ccrs.PlateCarree(),
)
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
plt.savefig("tmp.pdf")
