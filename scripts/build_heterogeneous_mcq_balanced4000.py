"""Build and verify the frozen seed-42 balanced 4 x 4,000 MCQ pool."""

import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/unified_mcq"
DEST = ROOT / "dataset/heterogeneous_mcq_4datasets_balanced4000"
DOMAINS = ("medqa", "logiqa", "openbookqa", "sciq")
EXPECTED_SOURCE_COUNTS = dict(zip(DOMAINS, (10177, 7179, 4957, 11668)))
SAMPLES_PER_DATASET = 4000
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row):
    return f"{row['dataset']}:{row['id']}"


def id_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def build():
    clients, sources, tests = [], {}, {}
    for domain in DOMAINS:
        source = SOURCE / domain / "train.jsonl"
        rows = read_jsonl(source)
        expected = EXPECTED_SOURCE_COUNTS[domain]
        if len(rows) != expected:
            raise ValueError(f"{domain}: expected {expected} official train rows, found {len(rows)}")
        if any(row.get("dataset") != domain or row.get("split") != "train" for row in rows):
            raise ValueError(f"{domain}: unexpected dataset or split field")
        source_ids = [key(row) for row in rows]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{domain}: duplicate native IDs in official train")
        random.Random(SEED).shuffle(rows)
        selected = rows[:SAMPLES_PER_DATASET]
        clients.append(selected)
        sources[domain] = {
            "path": source.relative_to(ROOT).as_posix(),
            "sha256": sha256(source),
            "original_train_size": expected,
            "selected_count": len(selected),
            "selected_ids": [str(row["id"]) for row in selected],
            "selected_id_sha256": id_hash(map(key, selected)),
        }
        test_source = SOURCE / domain / "test.jsonl"
        test_target = DEST / "test" / f"{domain}.jsonl"
        test_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test_source, test_target)
        if sha256(test_source) != sha256(test_target):
            raise RuntimeError(f"{domain}: official test copy differs")
        tests[domain] = {
            "source": test_source.relative_to(ROOT).as_posix(),
            "path": test_target.relative_to(ROOT).as_posix(),
            "sha256": sha256(test_target),
            "count": len(read_jsonl(test_target)),
        }

    selected_keys = [key(row) for rows in clients for row in rows]
    selected_set = set(selected_keys)
    duplicate_count = len(selected_keys) - len(selected_set)
    if len(selected_keys) != 16000 or duplicate_count:
        raise RuntimeError("Balanced pool is not 16,000 unique IDs")

    manifest = {
        "seed": SEED,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "total_train": len(selected_keys),
        "validation_used": False,
        "train_sources": sources,
        "official_tests": tests,
        "client_dataset_mapping": {str(i): domain for i, domain in enumerate(DOMAINS)},
        "clients": [],
        "global_train_id_sha256": id_hash(selected_keys),
        "duplicate_count": duplicate_count,
        "missing_count": 0,
        "expected_fedavg_weights": {str(i): 0.25 for i in range(4)},
    }
    for client_id, (domain, rows) in enumerate(zip(DOMAINS, clients)):
        path = DEST / "train/dataset_skewed" / f"{client_id}.jsonl"
        write_jsonl(path, rows)
        manifest["clients"].append({
            "client": client_id,
            "dataset": domain,
            "count": len(rows),
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "id_sha256": id_hash(map(key, rows)),
            "expected_fedavg_weight": 0.25,
        })
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return verify()


def verify():
    manifest = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["samples_per_dataset"] == 4000
    assert manifest["total_train"] == 16000
    assert manifest["validation_used"] is False
    assert manifest["duplicate_count"] == manifest["missing_count"] == 0
    assert manifest["expected_fedavg_weights"] == {str(i): 0.25 for i in range(4)}
    found = []
    for client_id, domain in enumerate(DOMAINS):
        source = manifest["train_sources"][domain]
        source_path = ROOT / source["path"]
        assert len(read_jsonl(source_path)) == source["original_train_size"] == EXPECTED_SOURCE_COUNTS[domain]
        assert sha256(source_path) == source["sha256"]
        entry = manifest["clients"][client_id]
        path = ROOT / entry["path"]
        rows = read_jsonl(path)
        assert entry["client"] == client_id and entry["dataset"] == domain
        assert entry["count"] == len(rows) == 4000
        assert all(row["dataset"] == domain and row["split"] == "train" for row in rows)
        assert sha256(path) == entry["sha256"]
        assert id_hash(map(key, rows)) == entry["id_sha256"] == source["selected_id_sha256"]
        assert [str(row["id"]) for row in rows] == source["selected_ids"]
        assert entry["expected_fedavg_weight"] == 0.25
        test = manifest["official_tests"][domain]
        assert sha256(ROOT / test["source"]) == sha256(ROOT / test["path"]) == test["sha256"]
        found.extend(map(key, rows))
    assert len(found) == len(set(found)) == 16000
    assert id_hash(found) == manifest["global_train_id_sha256"]
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else build()
    print("Client sizes:", [entry["count"] for entry in result["clients"]])
    print("FedAvg weights:", list(result["expected_fedavg_weights"].values()))
    print("Balanced pool: PASS")
