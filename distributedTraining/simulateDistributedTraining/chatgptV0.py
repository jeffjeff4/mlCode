import multiprocessing as mp
import os
import csv
import pickle
from typing import List, Optional

# -----------------------------
# Checkpoint helpers
# -----------------------------
def ckpt_path(ckpt_dir: str, proc_id: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"trainer_{proc_id}.pkl")

def save_checkpoint(proc_id: int, partial_sum: int, ckpt_dir: str) -> None:
    with open(ckpt_path(ckpt_dir, proc_id), "wb") as f:
        pickle.dump(partial_sum, f)

def load_checkpoint(proc_id: int, ckpt_dir: str) -> Optional[int]:
    path = ckpt_path(ckpt_dir, proc_id)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# -----------------------------
# Data helpers
# -----------------------------
def read_data(csv_file: Optional[str] = None, n_default: int = 1000) -> List[int]:
    """
    If csv_file is provided, it should contain one numeric value per row.
    Otherwise, generate [1..n_default].
    """
    if csv_file is None:
        return list(range(1, n_default + 1))

    data = []
    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            data.append(int(row[0]))
    return data

def shard_data(data: List[int], num_shards: int) -> List[List[int]]:
    sz = len(data)
    base = sz // num_shards
    rem = sz % num_shards
    shards = []
    start = 0
    for i in range(num_shards):
        extra = 1 if i < rem else 0
        end = start + base + extra
        shards.append(data[start:end])
        start = end
    return shards

# -----------------------------
# Worker
# -----------------------------
def worker(proc_id, shard, results, ckpt_dir, resume, fail_now):
    """
    proc_id: process id (rank)
    shard: list[int] assigned to this process
    results: Manager().list shared object to publish partial sum
    ckpt_dir: directory to store checkpoints
    resume: if True, try to load checkpoint first
    fail_now: if True, simulate a crash before doing any work (no checkpoint written)
    """
    # Try resume from checkpoint
    if resume:
        ck = load_checkpoint(proc_id, ckpt_dir)
        if ck is not None:
            print(f"[Trainer {proc_id}] Resume from checkpoint: {ck}")
            results[proc_id] = ck
            return

    # Simulate failure (no checkpoint written)
    if fail_now:
        print(f"[Trainer {proc_id}] Simulated failure!")
        return  # exit early; coordinator will detect missing result

    # Normal compute path
    partial = sum(shard)
    print(f"[Trainer {proc_id}] Computed partial sum: {partial}")

    # Save checkpoint and publish result
    save_checkpoint(proc_id, partial, ckpt_dir)
    results[proc_id] = partial

# -----------------------------
# Coordinator / Runner
# -----------------------------
def run_job(data: List[int], num_trainers: int, ckpt_dir: str,
           resume: bool = False, fail_rank: Optional[int] = None) -> List[Optional[int]]:
    """
    Launch N trainers, each with a shard. Optionally simulate failure of `fail_rank`
    during the first run (no checkpoint). If resume=True, workers will try to reuse
    their checkpoints instead of recomputing.
    Returns a list of partial sums (None for failed/missing ones).
    """
    shards = shard_data(data, num_trainers)
    manager = mp.Manager()
    results = manager.list([None for _ in range(num_trainers)])

    procs = []
    for r in range(num_trainers):
        fail_now = (fail_rank == r) and (not resume)  # only fail on the first attempt
        p = mp.Process(
            target=worker,
            args=(r, shards[r], results, ckpt_dir, resume, fail_now),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    return list(results)

# -----------------------------
# Tests
# -----------------------------
def test_normal(num_trainers: int = 4, n: int = 1000, ckpt_dir: str = "checkpoints_normal"):
    print("\n=== Test 1: Normal run (no failure) ===")
    data = read_data(None, n_default=n)
    expected = sum(data)

    # Clean up old checkpoints if any
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            os.remove(os.path.join(ckpt_dir, f))
    else:
        os.makedirs(ckpt_dir, exist_ok=True)

    partials = run_job(data, num_trainers, ckpt_dir, resume=False, fail_rank=None)
    if any(p is None for p in partials):
        raise RuntimeError("Normal run produced missing results unexpectedly.")

    total = sum(partials)
    print(f"Partials: {partials}")
    print(f"Global sum: {total}, Expected: {expected}")
    assert total == expected, "Mismatch in global sum during normal run!"

def test_failure_recovery(num_trainers: int = 4, n: int = 1000,
                          ckpt_dir: str = "checkpoints_failure", fail_rank: int = 2):
    print("\n=== Test 2: Failure + Resume ===")
    data = read_data(None, n_default=n)
    expected = sum(data)

    # Clean up old checkpoints if any
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            os.remove(os.path.join(ckpt_dir, f))
    else:
        os.makedirs(ckpt_dir, exist_ok=True)

    # First run: simulate failure of one rank → that rank won't write a checkpoint
    partials = run_job(data, num_trainers, ckpt_dir, resume=False, fail_rank=fail_rank)
    print(f"After failure run, partials: {partials}")

    # If any rank failed (None), do a resume: surviving ranks will load ckpts,
    # failed rank will recompute and write its checkpoint
    if any(p is None for p in partials):
        print("Detected missing partials. Resuming from checkpoints...")
        partials = run_job(data, num_trainers, ckpt_dir, resume=True, fail_rank=None)

    total = sum(partials)
    print(f"After resume, partials: {partials}")
    print(f"Global sum: {total}, Expected: {expected}")
    assert total == expected, "Mismatch in global sum after failure recovery!"

# -----------------------------
# Entry (as requested starter)
# -----------------------------
def main():
    # Run both tests
    test_normal(num_trainers=4, n=1000, ckpt_dir="checkpoints_normal")
    test_failure_recovery(num_trainers=4, n=1000, ckpt_dir="checkpoints_failure", fail_rank=2)
    print("\nAll tests passed ✅")

if __name__ == "__main__":
    main()
