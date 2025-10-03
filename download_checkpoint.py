from huggingface_hub import snapshot_download

# Download the entire model repository
snapshot_download(
    repo_id="MONAI/Llama3-VILA-M3-3B",
    local_dir="./Llama3-VILA-M3-3B",  # Local directory to save to
    local_dir_use_symlinks=False      # Download actual files, not symlinks
)
