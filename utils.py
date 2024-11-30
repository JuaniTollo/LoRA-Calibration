# utils.py

import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
# utils.py

import os
from huggingface_hub import snapshot_download

def ensure_tokenizer(tokenizer_path="./tokenizer/tokenizer.json"):
    """
    Checks if the tokenizer exists at the specified path.
    If not, downloads it from the Hugging Face Hub.
    """
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found. Downloading...")
        # Replace 'repo_id' with your tokenizer repository ID
        repo_id = "juantollo/tokenizer_phi_models"
        local_dir = os.path.dirname(tokenizer_path)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            use_auth_token=True  # Set to True if authentication is required
        )
        print("Tokenizer downloaded successfully.")
    else:
        print("Tokenizer found at:", tokenizer_path)
