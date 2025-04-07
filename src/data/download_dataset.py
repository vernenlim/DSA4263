import os
import subprocess


def download_kaggle_dataset(dataset_slug: str, output_dir: str):
    """
    Download a dataset from Kaggle using the Kaggle CLI.

    Args:
        dataset_slug (str): The Kaggle dataset slug (e.g., "rupakroy/online-payments-fraud-detection-dataset")
        output_dir (str): Path to the output directory to save files
    """
    print(f"Downloading dataset '{dataset_slug}' to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_slug,
                "-p",
                output_dir,
                "--unzip",
            ],
            check=True,
        )
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print("Error downloading dataset:", e)


def main():
    raw_data_dir = os.path.join("Dataset")
    dataset = "rupakroy/online-payments-fraud-detection-dataset"
    download_kaggle_dataset(dataset, raw_data_dir)


if __name__ == "__main__":
    main()
