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

        # Rename the downloaded CSV if needed
        if renamed_filename:
            for fname in os.listdir(output_dir):
                if fname.endswith(".csv"):
                    original_path = os.path.join(output_dir, fname)
                    renamed_path = os.path.join(output_dir, f"{renamed_filename}.csv")
                    os.rename(original_path, renamed_path)
                    print(f"Renamed '{fname}' to '{renamed_filename}.csv'")
                    break

    except subprocess.CalledProcessError as e:
        print("Error downloading dataset:", e)


def main():
    dataset_folder = os.path.join("Dataset")
    dataset_slug = "rupakroy/online-payments-fraud-detection-dataset"
    renamed_filename = "DSA4263_data_raw"

    download_kaggle_dataset(dataset_slug, dataset_folder, renamed_filename)


if __name__ == "__main__":
    main()
