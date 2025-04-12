import os
import subprocess


def download_kaggle_dataset(
    dataset_slug: str, output_dir: str, renamed_filename: str = None
):
    """
    Download a dataset from Kaggle using the Kaggle CLI and optionally rename it.

    Args:
        dataset_slug (str): The Kaggle dataset slug (e.g., "rupakroy/online-payments-fraud-detection-dataset")
        output_dir (str): Path to the output directory to save files
        renamed_filename (str): Optional new filename (without extension), e.g., 'DSA4263_data_raw'
    """
    print(f"Downloading dataset '{dataset_slug}' to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download and unzip using Kaggle CLI
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
    dataset_folder = os.path.join("Dataset")
    dataset_slug = "rupakroy/online-payments-fraud-detection-dataset"

    download_kaggle_dataset(dataset_slug, dataset_folder)


if __name__ == "__main__":
    main()
