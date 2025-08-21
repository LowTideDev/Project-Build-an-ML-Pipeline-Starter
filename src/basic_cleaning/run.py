#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description="Basic data cleaning for NYC Airbnb dataset")

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Input W&B artifact to read, e.g. 'sample.csv:latest'",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name for the cleaned dataset artifact to create, e.g. 'clean_sample.csv'",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Artifact type to assign in W&B, e.g. 'clean_data'",
    )
    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Short description of the cleaned dataset contents",
    )
    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum allowed price; rows below this are dropped",
    )
    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum allowed price; rows above this are dropped",
    )

    return parser.parse_args()


# DO NOT MODIFY (except to use parse_args() result)
def go(args):
    run = wandb.init(job_type="basic_cleaning", save_code=True)
    run.config.update(vars(args))

    # Download input artifact. This also tracks that we used it.
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Price outlier filter
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Parse dates
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"])

    # ⚠️ NYC geo filter — enable LATER for the “successful failure” part
    # idx_geo = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    # df = df[idx_geo].copy()

    # Save cleaned file
    out_csv = "clean_sample.csv"
    df.to_csv(out_csv, index=False)

    # Log the cleaned data as an artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(out_csv)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    go(args)
