"""
Basic data cleaning. A cleaned dataset will
be uploaded to W&B serving as an input
artifact for consecutive pipeline components.
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    A function to clean a sample dataset
    downloaded from W&B.
    
    Input:
        - args: command line arguments
    Output:
        - None
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    
    logger.info("Downloading artifact...")
    artifact_pth = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_pth)

    logger.info("Removing outliers...")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Adjusting data formats...")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Saving cleaned dataset...")
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="cleaned_data",
        description="Cleaned dataset",
    )

    logger.info("Uploading dataset artifact to W&B...")
    cwd = os.getcwd()
    artifact.add_file(os.path.join(cwd, args.output_artifact))
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        help="The name for the output artifact",
        type=str,
        required=True
    )

    parser.add_argument(
        "--min_price",
        help="The minimum price to consider",
        type=float,
        required=True
    )

    parser.add_argument(
        "--max_price",
        help="The maximum price to consider",
        type=float,
        required=True
    )

    args = parser.parse_args()

    go(args)