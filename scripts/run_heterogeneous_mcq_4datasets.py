"""Fixed seed-42 four-dataset MCQ partition and paper experiment.

Run with the existing fd environment. No validation data, new train/test split,
downsampling, intermediate official-test evaluation or hyperparameter search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DOMAINS = ("medqa", "logiqa", "openbookqa", "sciq")
EXPECTED_TRAIN = dict(zip(DOMAINS, (10177, 7179, 4957, 11668)))
SIZES = (2545, 2544, 2544, 2544, 1795, 1795, 1795, 1794,
         1240, 1239, 1239, 1239, 2917, 2917, 2917, 2917)
RAW = ROOT / "data" / "unified_mcq"
DATA = ROOT / "dataset" / "heterogeneous_mcq_4datasets"
RUNS = ROOT / "exp" / "heterogeneous_mcq_4datasets"
LOGS = ROOT / "logs" / "heterogeneous_mcq_4datasets"
REPORT = ROOT / "reports" / "heterogeneous_mcq_4datasets_seed42_results.md"
MODEL = ROOT / "models" / "models--llama3.2-1B"
TRAINING = ("centralized", "iid_mixed", "dataset_skewed")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict) -> str:
    return row["dataset"] + ":" + str(row["id"])


def id_hash(keys) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocation() -> list[dict[str, int]]:
    """Constrained largest remainder: exact row/client and column/dataset sums."""
    total = sum(SIZES)
    ideal = [[size * EXPECTED_TRAIN[d] / total for d in DOMAINS] for size in SIZES]
    cells = [[math.floor(value) for value in row] for row in ideal]
    row_left = [size - sum(row) for size, row in zip(SIZES, cells)]
    col_left = [EXPECTED_TRAIN[d] - sum(row[j] for row in cells) for j, d in enumerate(DOMAINS)]
    assert sum(row_left) == sum(col_left)
    while sum(row_left):
        choices = [(ideal[i][j] - cells[i][j], -i, -j, i, j)
                   for i in range(16) if row_left[i] for j in range(4) if col_left[j]]
        _, _, _, i, j = max(choices)
        cells[i][j] += 1
        row_left[i] -= 1
        col_left[j] -= 1
    assert not any(row_left) and not any(col_left)
    assert all(sum(row) == size for row, size in zip(cells, SIZES))
    assert all(sum(row[j] for row in cells) == EXPECTED_TRAIN[d] for j, d in enumerate(DOMAINS))
    assert all(all(value > 0 for value in row) for row in cells)
    return [dict(zip(DOMAINS, row)) for row in cells]


def configuration(method: str, smoke: bool = False) -> dict:
    name = "smoke" if smoke else method
    config = dict(
        alg="centralized" if method == "centralized" else "fedit",
        suffix="heterogeneous_mcq_4datasets/" + name,
        device=0, dataset="heterogeneous_mcq_4datasets",
        mcq_train_dir=str(DATA / "train" / ("smoke" if smoke else method)),
        mcq_dataset_dir=str(DATA), mcq_domains=list(DOMAINS),
        model="llama32_1b_base", model_path=str(MODEL), task_type="CAUSAL_LM",
        seed=42, deterministic=True, cn=2 if smoke else (1 if method == "centralized" else 16),
        sr=1.0, rnd=1 if smoke else 10, test_gap=1, mcq=True,
        final_only_test=not smoke, skip_mcq_test=smoke,
        global_test=False, cross_domain_qa=False, round_generation_metrics=False,
        bs=1, grad_accum=8, epoch=1, step=0, lr=1e-4, eval_batch_size=8,
        mode="prototype", lora_rank=8, lora_alpha=32, lora_dropout=0.05,
    )
    return config


def prepare() -> None:
    assert sum(SIZES) == sum(EXPECTED_TRAIN.values()) == 33981
    pools = {}
    sources = {}
    tests = {}
    for domain in DOMAINS:
        source = RAW / domain / "train.jsonl"
        rows = read_jsonl(source)
        assert len(rows) == EXPECTED_TRAIN[domain], (domain, len(rows))
        assert all(row["dataset"] == domain and row["split"] == "train" for row in rows)
        keys = [key(row) for row in rows]
        assert len(set(keys)) == len(keys), f"Repeated unified train ID in {domain}"
        random.Random(42).shuffle(rows)
        pools[domain] = rows
        sources[domain] = {"path": str(source.relative_to(ROOT)), "sha256": sha(source),
                           "train_count": len(rows), "train_id_hash": id_hash(keys)}
        test_source = RAW / domain / "test.jsonl"
        assert test_source.is_file()
        test_target = DATA / "test" / f"{domain}.jsonl"
        test_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test_source, test_target)
        assert sha(test_source) == sha(test_target)
        tests[domain] = {"official_source": str(test_source.relative_to(ROOT)),
                         "source_sha256": sha(test_source),
                         "copied_path": str(test_target.relative_to(ROOT)),
                         "copied_sha256": sha(test_target),
                         "count": sum(1 for _ in test_target.open(encoding="utf-8"))}
    full_keys = {key(row) for rows in pools.values() for row in rows}
    assert len(full_keys) == 33981
    # Dataset-skewed: independent seed-42 shuffle within each source, four slices.
    skewed = [[] for _ in range(16)]
    for di, domain in enumerate(DOMAINS):
        rows = pools[domain]
        start = 0
        for client in range(di * 4, di * 4 + 4):
            skewed[client] = rows[start:start + SIZES[client]]
            start += SIZES[client]
        assert start == len(rows)
    # IID: same shuffled source rows, with deterministic exact integer quotas.
    quotas = allocation()
    mixed = [[] for _ in range(16)]
    for domain in DOMAINS:
        start = 0
        for client in range(16):
            take = quotas[client][domain]
            mixed[client].extend(pools[domain][start:start + take])
            start += take
        assert start == len(pools[domain])
    pooled = [row for domain in DOMAINS for row in pools[domain]]
    partitions = {"centralized": [pooled], "iid_mixed": mixed, "dataset_skewed": skewed}
    manifest = {
        "seed": 42, "train_source": sources, "train_total": 33981,
        "global_train_id_hash": id_hash(full_keys),
        "official_test": tests, "validation_used": False,
        "client_size_vector": list(SIZES),
        "iid_allocation_method": "floor ideals then fill row/column margins by largest remaining fractional priority; deterministic ties by client/dataset index",
        "conditions": {},
        "same_global_train_ids_iid_and_skewed": True,
        "train_union_duplicate_missing_checks": {},
    }
    for method, clients in partitions.items():
        flat = [key(row) for rows in clients for row in rows]
        check = {"rows": len(flat), "unique_ids": len(set(flat)),
                 "duplicate_ids": len(flat) - len(set(flat)),
                 "missing_ids": len(full_keys - set(flat)),
                 "extra_ids": len(set(flat) - full_keys),
                 "train_id_hash": id_hash(flat)}
        assert check == {"rows": 33981, "unique_ids": 33981, "duplicate_ids": 0,
                         "missing_ids": 0, "extra_ids": 0, "train_id_hash": manifest["global_train_id_hash"]}
        manifest["train_union_duplicate_missing_checks"][method] = check
        manifest["conditions"][method] = []
        for ci, rows in enumerate(clients):
            if method != "centralized":
                assert len(rows) == SIZES[ci]
            random.Random(42 + ci).shuffle(rows)
            counts = Counter(row["dataset"] for row in rows)
            if method == "iid_mixed":
                assert all(counts[d] == quotas[ci][d] for d in DOMAINS)
            if method == "dataset_skewed":
                assert counts == {DOMAINS[ci // 4]: SIZES[ci]}
            path = DATA / "train" / method / f"{ci}.jsonl"
            write_jsonl(path, rows)
            manifest["conditions"][method].append({
                "client": ci, "size": len(rows),
                "dataset_counts": {d: counts[d] for d in DOMAINS},
                "train_id_hash": id_hash(map(key, rows)),
                "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        import yaml
        config_file = ROOT / "configs" / f"heterogeneous_mcq_4datasets_{method}.yaml"
        config_file.write_text(yaml.safe_dump(configuration(method), sort_keys=False), encoding="utf-8")
    assert manifest["train_union_duplicate_missing_checks"]["iid_mixed"]["train_id_hash"] == manifest["train_union_duplicate_missing_checks"]["dataset_skewed"]["train_id_hash"]
    manifest["model"] = {"path": str(MODEL),
                         "config_sha256": sha(MODEL / "config.json"),
                         "weights_sha256": sha(MODEL / "model.safetensors"),
                         "tokenizer_sha256": sha(MODEL / "tokenizer.json")}
    write_json(DATA / "manifest.json", manifest)
    # A tiny one-round training/aggregation smoke input drawn from existing train IDs.
    for ci in range(2):
        write_jsonl(DATA / "train" / "smoke" / f"{ci}.jsonl", skewed[ci][:8])
    print("Partition prepared: 33,981/33,981 IDs, exact client sizes, no overlap or missing IDs.", flush=True)


def verify_partition() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["client_size_vector"] == list(SIZES)
    original_keys = set()
    for domain in DOMAINS:
        source = manifest["train_source"][domain]
        assert sha(ROOT / source["path"]) == source["sha256"]
        original_keys.update(key(row) for row in read_jsonl(ROOT / source["path"]))
    assert len(original_keys) == 33981
    for method in ("iid_mixed", "dataset_skewed"):
        found = []
        for ci, info in enumerate(manifest["conditions"][method]):
            assert info["size"] == SIZES[ci]
            path = ROOT / info["path"]
            assert sha(path) == info["sha256"]
            rows = read_jsonl(path)
            assert len(rows) == info["size"]
            assert Counter(row["dataset"] for row in rows) == {d: n for d, n in info["dataset_counts"].items() if n}
            found.extend(map(key, rows))
        assert len(found) == len(set(found)) == 33981 and set(found) == original_keys
        assert id_hash(found) == manifest["global_train_id_hash"]
    return manifest


def run_training(method: str, smoke: bool = False) -> None:
    from main import FedSim
    FedSim(SimpleNamespace(**configuration(method, smoke)))


def run_base() -> None:
    from utils.mcq_eval import MCQEvaluator
    from utils.model_utils import load_model, load_tokenizer
    from utils.seed_utils import set_global_seed
    args = SimpleNamespace(**configuration("centralized"))
    args.suffix = str(RUNS / "base")
    set_global_seed(42, device=0, deterministic=True)
    model, tokenizer = load_model(args), load_tokenizer(args)
    result = MCQEvaluator(args, tokenizer).evaluate(model, 0)
    write_json(RUNS / "base" / "result.json", result)


def summarize() -> None:
    import torch
    manifest = verify_partition()
    results = {"base": json.loads((RUNS / "base" / "result.json").read_text(encoding="utf-8"))}
    initial = []
    for method in TRAINING:
        run = RUNS / method
        assert (run / "completed.txt").is_file()
        metrics = read_jsonl(run / "evaluation" / "metrics.jsonl")
        assert len(metrics) == 1 and metrics[0]["round"] == 10
        exposures = read_jsonl(run / "exposure.jsonl")
        assert len(exposures) == 10 * (1 if method == "centralized" else 16)
        for rnd in range(1, 11):
            seen = [item for row in exposures if row["round"] == rnd for item in row["ids"]]
            assert len(seen) == len(set(seen)) == 33981
            assert id_hash(seen) == manifest["global_train_id_hash"]
        results[method] = metrics[0]
        initial.append(torch.load(run / "initial_lora.pt", map_location="cpu", weights_only=True))
        assert (run / "adapter").exists()
    assert all(set(initial[0]) == set(state) and all(torch.equal(initial[0][name], state[name]) for name in initial[0]) for state in initial[1:])
    metric_names = [d + "_accuracy" for d in DOMAINS] + ["macro_accuracy", "worst_domain_accuracy"]
    lines = ["# Four-dataset heterogeneous MCQ × Llama-3.2-1B (seed 42)", "",
             "## 1. Data and official splits", "",
             "Only the existing unified official train was used for training, and the existing unified official test was used for final evaluation. Validation was never loaded. No data was re-cleaned, downsampled or assigned to a new train/test split.", "",
             "| Dataset | Train | Official test |", "| --- | ---: | ---: |"]
    for d in DOMAINS:
        lines.append(f"| {d} | {manifest['train_source'][d]['train_count']} | {manifest['official_test'][d]['count']} |")
    lines += ["", f"Total train: {manifest['train_total']:,}. Global train-ID SHA256: `{manifest['global_train_id_hash']}`. Full partition and source-file SHA256 records: `dataset/heterogeneous_mcq_4datasets/manifest.json`.",
              "", "## 2. Sixteen-client partitions", "",
              "IID-Mixed and Dataset-Skewed have identical per-client size vectors and exactly the same global train-ID set. Within each condition every train ID appears once; missing/duplicate/extra counts are all zero. Dataset-Skewed clients 0–3 MedQA, 4–7 LogiQA, 8–11 OpenBookQA and 12–15 SciQ.", "",
              "| Client | Size | IID MedQA | IID LogiQA | IID OpenBookQA | IID SciQ | Skewed dataset |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for ci in range(16):
        row = manifest["conditions"]["iid_mixed"][ci]
        count = row["dataset_counts"]
        lines.append(f"| {ci} | {row['size']} | {count['medqa']} ({100*count['medqa']/row['size']:.2f}%) | "
                     f"{count['logiqa']} ({100*count['logiqa']/row['size']:.2f}%) | "
                     f"{count['openbookqa']} ({100*count['openbookqa']/row['size']:.2f}%) | "
                     f"{count['sciq']} ({100*count['sciq']/row['size']:.2f}%) | {DOMAINS[ci // 4]} |")
    lines += ["", "IID integer allocation floors each ideal client × global-dataset proportion, then assigns residual units under exact client-size and dataset-total constraints using largest remaining fractional priority. Seed 42 shuffles each dataset before assignment.",
              "", "## 3. Model and training protocol", "",
              "Local Llama-3.2-1B BF16 base; identical initial LoRA tensors in all three trained methods. LoRA q_proj/v_proj, r=8, alpha=32, dropout=0.05, bias=none; only LoRA trains. Constant learning rate 1e-4, AdamW, batch 1, gradient accumulation 8, seed 42. Centralized-All performs 10 complete pooled epochs. Each FedAvg condition uses 16/16 clients, 10 rounds and one complete local epoch per round. `drop_last=False`; every one of 33,981 train IDs is visited exactly once per round/epoch. There is no step truncation. Existing LoRA A/B factor aggregation is unchanged.",
              "", "Prompt is the existing `Question: ...` / A–D / `Answer:` format. Training supervises only the space-plus-letter continuation and EOS. Evaluation scores conditional log P of each continuation without free generation. Official test is loaded only for Base or after each trained method's fixed final epoch/round 10; no validation, intermediate test or best-checkpoint selection.",
              "", "## 4. Final official-test results", "",
              "| Method | MedQA | LogiQA | OpenBookQA | SciQ | Macro | Worst |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    labels = (("base", "Base"), ("centralized", "Centralized-All"),
              ("iid_mixed", "FedAvg-IID-Mixed"), ("dataset_skewed", "FedAvg-Dataset-Skewed"))
    for method, label in labels:
        lines.append("| " + label + " | " + " | ".join(f"{100*results[method][name]:.2f}%" for name in metric_names) + " |")
    lines += ["", "Macro is the unweighted average of the four dataset accuracies; Worst is their minimum. These are the prespecified final checkpoints, not selected peaks.",
              "", "## 5. Prespecified comparisons", "", "| Comparison | Difference (percentage points) |",
              "| --- | ---: |"]
    iid, skew, central = (results[name] for name in ("iid_mixed", "dataset_skewed", "centralized"))
    for name in metric_names[:4] + ["macro_accuracy"]:
        label = name.replace("_accuracy", "")
        lines.append(f"| IID − Dataset-Skewed: {label} | {100*(iid[name]-skew[name]):+.4f} |")
    lines.append(f"| Centralized − IID: macro | {100*(central['macro_accuracy']-iid['macro_accuracy']):+.4f} |")
    lines.append(f"| Centralized − Dataset-Skewed: macro | {100*(central['macro_accuracy']-skew['macro_accuracy']):+.4f} |")
    lines += ["", "## 6. Interpretation boundary", "",
              "The gap tests whether client-level dataset specialization is associated with a FedAvg performance penalty under this model, data and single seed. A near-zero gap does not support a claim of knowledge conflict. A positive gap is an association with aggregation degradation, not direct evidence of forgetting or a specific parameter-space mechanism, and does not generalize to all FedLLM systems.",
              "", "Logs: `logs/heterogeneous_mcq_4datasets/`; final adapters, exposure records and official-test predictions: `exp/heterogeneous_mcq_4datasets/<method>/`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(RUNS / "final_results.json", results)
    print(f"Base Macro: {100*results['base']['macro_accuracy']:.2f}%")
    print(f"Centralized Macro: {100*central['macro_accuracy']:.2f}%")
    print(f"IID-Mixed Macro: {100*iid['macro_accuracy']:.2f}%")
    print(f"Dataset-Skewed Macro: {100*skew['macro_accuracy']:.2f}%")
    print(f"IID - Dataset-Skewed Gap: {100*(iid['macro_accuracy']-skew['macro_accuracy']):+.2f} percentage points")
    print("Experiment completed.")


def all_runs() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if not (DATA / "manifest.json").is_file():
        prepare()
    verify_partition()
    for job in ("smoke", "base", *TRAINING):
        marker = RUNS / job / ("result.json" if job == "base" else "completed.txt")
        if marker.is_file():
            continue
        with (LOGS / f"{job}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--job", job],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if job == "smoke":
            assert list((RUNS / "smoke" / "adapter").rglob("lora_weights.pt"))
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("all", "prepare", "smoke", "base", "centralized",
                                         "iid_mixed", "dataset_skewed", "summarize"), default="all")
    job = parser.parse_args().job
    os.chdir(ROOT)
    if job == "all":
        all_runs()
    elif job == "prepare":
        prepare()
    elif job == "smoke":
        run_training("iid_mixed", smoke=True)
    elif job == "base":
        run_base()
    elif job == "summarize":
        summarize()
    else:
        run_training(job)
