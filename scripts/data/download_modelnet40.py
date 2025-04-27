from huggingface_hub import snapshot_download

from src.paths import PATHS


if __name__ == "__main__":
    snapshot_download(
        repo_id="user/modelnet40_preprocessed",
        repo_type="dataset",
        local_dir=PATHS.data + "modelnet40_preprocessed",
    )
    snapshot_download(
        repo_id="user/modelnet40_preprocessed_high_res",
        repo_type="dataset",
        local_dir=PATHS.data + "modelnet40_preprocessed_high_res",
    )
