#!/usr/bin/env python3
"""Upload one folder to a Hugging Face model repository (called by upload_checkpoints.sh).

    python3 experiments/hf_upload.py <repo_id> <folder> [commit message]

Always talks to the official endpoint: download mirrors such as hf-mirror.com must never receive the token.

Built for a slow, lossy, intermittently connected link (the lab machine this repository was trained on):
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1   upload the LFS file in parallel parts with hf_transfer
    HFT_MAX_FILES=16          cap on parallel connections (huggingface_hub hard-codes 128, which floods a shared uplink)
    HFT_MAX_RETRIES=50        retries per part (huggingface_hub hard-codes 5; one part failing 5 times aborts the file)
    HF_UPLOAD_ATTEMPTS=10     whole-folder attempts; a new attempt restarts the LFS file from zero
    HF_WAIT_HOURS=12          before each attempt, wait (probing every 2 minutes) for the Hub to become reachable
"""
import os, sys, time, functools

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
CAP = max(1, int(os.environ.get("HFT_MAX_FILES", "16")))
RETRIES = max(5, int(os.environ.get("HFT_MAX_RETRIES", "50")))
ATTEMPTS = max(1, int(os.environ.get("HF_UPLOAD_ATTEMPTS", "10")))
WAIT_S = float(os.environ.get("HF_WAIT_HOURS", "12")) * 3600

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

import requests
from huggingface_hub import HfApi


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def wait_for_hub():
    t0 = time.time(); warned = False
    while True:
        try:
            if requests.head("https://huggingface.co", timeout=15).status_code < 500:
                if warned: log(f"hub reachable again after {time.time() - t0:.0f} s")
                return
        except requests.RequestException:
            pass
        if time.time() - t0 > WAIT_S:
            raise SystemExit(f"hub unreachable for {WAIT_S / 3600:.1f} h, giving up")
        if not warned: log("hub unreachable, probing every 2 minutes"); warned = True
        time.sleep(120)


repo, folder = sys.argv[1], sys.argv[2]
message = sys.argv[3] if len(sys.argv) > 3 else f"upload {os.path.basename(os.path.normpath(folder))}"
api = HfApi()
for attempt in range(1, ATTEMPTS + 1):
    wait_for_hub()
    t0 = time.time()
    try:
        api.create_repo(repo, repo_type="model", exist_ok=True)
        info = api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model", commit_message=message)
        log(f"uploaded {folder} -> {info} (attempt {attempt}, {time.time() - t0:.0f} s)")
        break
    except Exception as e:                      # network failures on long uploads
        log(f"attempt {attempt}/{ATTEMPTS} failed after {time.time() - t0:.0f} s: {type(e).__name__}: {str(e)[:300]}")
        if attempt == ATTEMPTS:
            raise
        time.sleep(60)
