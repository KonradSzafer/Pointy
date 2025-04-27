import os
import sys
from pathlib import Path

import wget

from src.paths import PATHS


def download_object_dataset_h5():
    base_url = "https://hkust-vgd.ust.hk/scanobjectnn/"
    filenames = [
        "h5_files.zip"
    ]
    
    for filename in filenames:
        url = base_url + filename
    
        output_dir = Path(PATHS.data).resolve() / "scanobjectnn"
        output_path = output_dir / filename
        os.makedirs(output_dir, exist_ok=True)

        if not output_path.exists():
            print(f"Downloading to: {output_path}")
            wget.download(url, out=str(output_path))
            print(f"\nDownloaded file saved as {output_path}")
            
        # Unzip the downloaded file
        unzip_cmd = f"unzip {output_path} -d {output_dir}"
        print(f"Unzipping file with command: {unzip_cmd}")
        os.system(unzip_cmd)


if __name__ == "__main__":
    download_object_dataset_h5()
