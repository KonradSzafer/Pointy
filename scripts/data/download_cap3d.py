import os
import logging
import subprocess

from src.paths import PATHS

logger = logging.getLogger(__name__)


DATA_DIR = PATHS.data + "cap3d/"
DATASET_URL = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/"
CSV_FILENAME_1 = "Cap3D_automated_Objaverse_full.csv"
CSV_FILENAME_2 = "Cap3D_automated_Objaverse_no3Dword.csv"
FILENAMES = [
    "PointCloud_pt_zips/compressed_pcs_pt_00.zip",  # 44.0 GB - 200k
    "PointCloud_pt_zips/compressed_pcs_pt_01.zip",  # 44.0 GB - 200k
    "PointCloud_pt_zips/compressed_pcs_pt_02.zip",  # 44.0 GB - 200k
    "PointCloud_pt_zips/compressed_pcs_pt_03.zip",  # 13.5 GB - 61577
    "PointCloud_pt_zips/compressed_pcs_pt_04.zip",  # 28.3 GB - 123573
    "PointCloud_pt_zips/compressed_pcs_pt_05.zip",  # 40.4 GB - 200k
    "PointCloud_pt_zips/compressed_pcs_pt_06.zip",  # 4.36 GB - 21632
]


def get_num_files() -> int:
    return len(os.listdir(f"{DATA_DIR}Cap3D_pcs_pt/"))


def download_file(url: str, output_path: str) -> None:
    if os.path.exists(output_path):
        logger.info(f"File already exists at {output_path}. Skipping download.")
        return
    
    subprocess.run(
        [
            "curl",
            "--retry", "5",
            "--connect-timeout", "30",
            "-L",
            "-o", output_path,
            url,
        ],
        check=True,
    )


def unzip_file(filename: str, output_dir: str) -> None:
    subprocess.run(
        [
            "unzip",
            "-q",  # Quiet mode
            filename,
            "-d",
            output_dir,
        ],
        input="A",
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    logger.info(f"Downloading Cap3D dataset to {DATA_DIR}")
    os.makedirs(DATA_DIR + "Cap3D_pcs_pt/", exist_ok=True)
    
    # Create the data directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Download annotation files
    download_file(DATASET_URL + CSV_FILENAME_1, DATA_DIR + CSV_FILENAME_1)
    download_file(DATASET_URL + CSV_FILENAME_2, DATA_DIR + CSV_FILENAME_2)

    # Download point cloud archives
    num_files_initial = get_num_files()

    for filename in FILENAMES:
        # Download the file
        url = DATASET_URL + filename
        output_filename = DATA_DIR + os.path.basename(filename)  # Without the subdir
        logger.info(f"Downloading {url} to {output_filename}")
        download_file(url, output_filename)

        # Extract the zip file
        logger.info(f"Extracting {output_filename}")
        unzip_file(output_filename, DATA_DIR)

        # Log num files
        num_files = get_num_files()
        delta = num_files - num_files_initial
        logger.info(f"Total files: {num_files} delta: {delta}")
        num_files_initial = num_files
        
        # Remmove the zip file
        if delta > 0:
            os.remove(output_filename)

    # Log final num files
    logger.info(f"Total files: {num_files_initial}")
