#!/usr/bin/env python3
"""Upload one folder to a Hugging Face model repository (called by upload_checkpoints.sh).

    python3 experiments/hf_upload.py <repo_id> <folder> [commit message]

Always talks to the official endpoint: download mirrors such as hf-mirror.com must never receive the token.

Built for a slow, lossy, intermittently connected link (the lab machine this repository was trained on):
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1   upload the LFS file in parallel parts with hf_transfer
    HFT_MAX_FILES=16          cap on parallel connections per file (huggingface_hub hard-codes 128, which floods a shared uplink)
    HF_FILES_PARALLEL=1       LFS files uploaded at once, each in its own process, largest first (with hf_transfer
                              huggingface_hub otherwise uploads the files one after another); total connections =
                              HF_FILES_PARALLEL x HFT_MAX_FILES. Use with RESHARD in upload_checkpoints.sh.
    HFT_MAX_RETRIES=50        retries per part (huggingface_hub hard-codes 5; one part failing 5 times aborts the file)
    HF_UPLOAD_ATTEMPTS=10     whole-folder attempts; files that finished uploading are skipped on the next attempt
    HF_WAIT_HOURS=12          before each attempt, wait (probing every 2 minutes) for the Hub, and with Xet disabled the
                              S3 accelerate endpoint the LFS parts go to, to become reachable (HF_PROBE_URLS overrides)
"""
import os, sys, time, functools

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
CAP = max(1, int(os.environ.get("HFT_MAX_FILES", "16")))
FILES_PAR = max(1, int(os.environ.get("HF_FILES_PARALLEL", "1")))
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
import huggingface_hub.lfs as _lfs

POST_WAIT_S = float(os.environ.get("HF_POST_WAIT_MIN", "30")) * 60


class _PatientSession:
    """huggingface_hub posts the LFS batch request, the multipart completion and the verify call to the Hub once, with no
    retry. On a link where the Hub drops out while S3 stays up, a shard whose parts are all on S3 is lost because the
    completion call hit an outage. This wrapper retries those POSTs every 30 s for up to HF_POST_WAIT_MIN minutes."""
    def __init__(self, session):
        self._s = session

    def __getattr__(self, name):
        return getattr(self._s, name)

    def post(self, *args, **kwargs):
        t0 = time.time()
        while True:
            try:
                r = self._s.post(*args, **kwargs)
                if r.status_code < 500 or time.time() - t0 > POST_WAIT_S:
                    return r
            except (requests.ConnectionError, requests.Timeout):
                if time.time() - t0 > POST_WAIT_S:
                    raise
            time.sleep(30)


_orig_get_session = _lfs.get_session
_lfs.get_session = lambda: _PatientSession(_orig_get_session())     # lfs.py looks get_session up at call time


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


LFS_EDGE = "https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com"     # where LFS parts go when Xet is disabled
PROBES = [u for u in os.environ.get("HF_PROBE_URLS", "").split(",") if u] or (
    ["https://huggingface.co", LFS_EDGE] if os.environ.get("HF_HUB_DISABLE_XET") == "1" else ["https://huggingface.co"])


def reachable(url):
    try:
        requests.head(url, timeout=15)          # any HTTP answer means the path is up; S3 answers 403/405 to an anonymous HEAD
        return True
    except requests.RequestException:
        return False


def wait_for_hub():
    t0 = time.time(); warned = False
    while True:
        if all(reachable(u) for u in PROBES):
            if warned: log(f"hub and upload endpoint reachable again after {time.time() - t0:.0f} s")
            return
        if time.time() - t0 > WAIT_S:
            raise SystemExit(f"hub unreachable for {WAIT_S / 3600:.1f} h, giving up")
        if not warned: log(f"unreachable: {[u for u in PROBES if not reachable(u)]}, probing every 2 minutes"); warned = True
        time.sleep(120)


def files_of(folder):
    out = []
    for root, _, names in os.walk(folder):
        for n in sorted(names):
            p = os.path.join(root, n)
            out.append((p, os.path.relpath(p, folder).replace(os.sep, "/")))
    return out


def preupload_one(job):                         # runs in a worker process; skips the file if the Hub already has it
    repo_id, path, rel = job
    from huggingface_hub import CommitOperationAdd
    HfApi().preupload_lfs_files(repo_id, additions=[CommitOperationAdd(path_in_repo=rel, path_or_fileobj=path)], repo_type="model")
    return rel


def upload(api, repo_id, folder, message):
    from huggingface_hub import CommitOperationAdd
    files = files_of(folder)
    big = sorted([f for f in files if os.path.getsize(f[0]) > 10 * 1024 * 1024], key=lambda f: -os.path.getsize(f[0]))
    if FILES_PAR > 1 and len(big) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=FILES_PAR, mp_context=mp.get_context("fork")) as ex:
            futures = [ex.submit(preupload_one, (repo_id, p, r)) for p, r in big]
            for i, fut in enumerate(as_completed(futures), 1):
                log(f"  {fut.result()} on the Hub ({i}/{len(big)})")
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=p) for p, r in files]
    return api.create_commit(repo_id=repo_id, operations=ops, commit_message=message, repo_type="model")


repo, folder = sys.argv[1], sys.argv[2]
message = sys.argv[3] if len(sys.argv) > 3 else f"upload {os.path.basename(os.path.normpath(folder))}"
api = HfApi()
for attempt in range(1, ATTEMPTS + 1):
    wait_for_hub()
    t0 = time.time()
    try:
        api.create_repo(repo, repo_type="model", exist_ok=True)
        info = upload(api, repo, folder, message)
        log(f"uploaded {folder} -> {info} (attempt {attempt}, {time.time() - t0:.0f} s)")
        break
    except Exception as e:                      # network failures on long uploads
        cause, depth = e.__cause__, 0           # worker-process errors carry the remote traceback as the cause
        detail = ""
        while cause is not None and depth < 3:
            detail = str(cause).strip().splitlines()[-1][:300] if str(cause).strip() else type(cause).__name__
            cause, depth = cause.__cause__, depth + 1
        log(f"attempt {attempt}/{ATTEMPTS} failed after {time.time() - t0:.0f} s: {type(e).__name__}: {str(e)[:200]}"
            + (f" | cause: {detail}" if detail else ""))
        if attempt == ATTEMPTS:
            raise
        time.sleep(60)
