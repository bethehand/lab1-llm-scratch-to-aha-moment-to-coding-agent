#!/usr/bin/env python3
"""Upload one folder to a Hugging Face model repository (called by upload_checkpoints.sh).

    python3 experiments/hf_upload.py <repo_id> <folder> [commit message]

Always talks to the official endpoint: download mirrors such as hf-mirror.com must never receive the token.

On slow long-haul links every single TCP connection gets squeezed to ~100 KB/s, so parallel connections are what helps:
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1   upload the LFS file in parallel parts with hf_transfer
    HFT_MAX_FILES=16                                    cap on parallel connections (huggingface_hub hard-codes 128,
                                                        which floods a shared uplink and gains nothing once it is full)
    HFT_MAX_RETRIES=50                                  retries per part (huggingface_hub hard-codes 5; one part failing
                                                        5 times aborts the whole file and a new attempt starts from zero)
    HF_UPLOAD_ATTEMPTS=3                                whole-folder attempts before giving up
"""
import os, sys, time, functools

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
CAP = max(1, int(os.environ.get("HFT_MAX_FILES", "16")))
RETRIES = max(5, int(os.environ.get("HFT_MAX_RETRIES", "50")))
ATTEMPTS = max(1, int(os.environ.get("HF_UPLOAD_ATTEMPTS", "3")))

try:
    import hf_transfer
    _orig = hf_transfer.multipart_upload

    @functools.wraps(_orig)                     # keeps the signature visible, so huggingface_hub still passes its progress callback
    def _capped(*args, **kwargs):
        kwargs["max_files"] = min(int(kwargs.get("max_files", CAP)), CAP)
        kwargs["parallel_failures"] = min(int(kwargs.get("parallel_failures", CAP - 1)), max(CAP - 1, 0))
        kwargs["max_retries"] = max(int(kwargs.get("max_retries", RETRIES)), RETRIES)
        return _orig(*args, **kwargs)

    hf_transfer.multipart_upload = _capped     # huggingface_hub imports it from the module at call time
except ImportError:
    pass

from huggingface_hub import HfApi

repo, folder = sys.argv[1], sys.argv[2]
message = sys.argv[3] if len(sys.argv) > 3 else f"upload {os.path.basename(os.path.normpath(folder))}"
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
for attempt in range(1, ATTEMPTS + 1):
    t0 = time.time()
    try:
        info = api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model", commit_message=message)
        print(f"uploaded {folder} -> {info} (attempt {attempt}, {time.time() - t0:.0f} s)", flush=True)
        break
    except Exception as e:                      # network failures on long uploads; a new attempt restarts the LFS file
        print(f"attempt {attempt}/{ATTEMPTS} failed after {time.time() - t0:.0f} s: {type(e).__name__}: {e}", flush=True)
        if attempt == ATTEMPTS:
            raise
        time.sleep(60)
