import argparse
import os

import wget

URL = "prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"

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

# Download data
for year in range(1995, 2022):
    print(f"\nDownloading data for {year}...")
    wget.download(f"http://{URL}/pp-{year}.csv", out=args.data_dir)
