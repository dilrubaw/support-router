"""Eğitilmiş ağırlıkları (artifacts/) Hugging Face Hub'a yükler. Önce: huggingface-cli login"""
from huggingface_hub import HfApi

import config

if __name__ == "__main__":
    api = HfApi()
    api.create_repo(config.HF_REPO_ID, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(config.ARTIFACTS), repo_id=config.HF_REPO_ID,
                      repo_type="model")
    print(f"Yüklendi: https://huggingface.co/{config.HF_REPO_ID}")
