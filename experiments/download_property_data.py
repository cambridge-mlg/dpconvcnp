import argparse
import os
from multiprocessing import Pool

import wget
import pandas as pd
import numpy as np
from tqdm import tqdm
import pgeocode

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf


URL = "prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
YEARS = np.arange(1995, 2023)


def get_latitude_longitude(postcode: str) -> tuple:
    nomi = pgeocode.Nominatim("gb")
    location = nomi.query_postal_code(postcode)
    return (location.latitude, location.longitude)


def add_coordinates_to_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    nomi = pgeocode.Nominatim("gb")

    # Keep rows with postcodes of type str only
    dataframe = dataframe[dataframe.iloc[:, 3].apply(type) == str]

    postcodes = dataframe.iloc[:, 3].values

    with Pool() as pool:
        latitudes, longitudes = zip(
            *pool.map(get_latitude_longitude, postcodes)
        )

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

print("\nFinished downloading data, preprocessing...")
dataframes = [
    add_coordinates_to_dataframe(
        pd.read_csv(f"{args.data_dir}/raw/pp-{year}.csv", header=None)
    )
    for year in tqdm(YEARS)
]

for dataframe, year in zip(dataframes, YEARS):
    dataframe.to_csv(
        f"{args.data_dir}/processed/{year}.csv",
        index=False,
    )

dataframe = pd.concat(dataframes)
dataframe.to_csv(f"{args.data_dir}/processed/all.csv", index=False)

ax = plt.axes(projection=ccrs.Mercator())
ax.scatter(dataframe.iloc[:, -2], dataframe.iloc[:, -1], transform=ccrs.PlateCarree())
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
plt.savefig("tmp.pdf")