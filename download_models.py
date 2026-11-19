from huggingface_hub import snapshot_download

import config


def download():
    snapshot_download(repo_id=config.HF_REPO_ID, repo_type="model",
                      local_dir=str(config.ARTIFACTS))
    print(f"İndirildi: {config.ARTIFACTS}")


if __name__ == "__main__":
    download()
