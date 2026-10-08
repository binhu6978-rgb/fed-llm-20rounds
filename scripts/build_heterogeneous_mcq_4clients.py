"""Build and verify the seed-42, four-client partition from official train files."""

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/unified_mcq"
DEST = ROOT / "dataset/heterogeneous_mcq_4datasets_4clients"
DOMAINS = ("medqa", "logiqa", "openbookqa", "sciq")
EXPECTED = (10177, 7179, 4957, 11668)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def identity(row):
    return f"{row['dataset']}:{row['id']}"


def id_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def inspect(rows, expected):
    ids = [identity(row) for row in rows]
    found = set(ids)
    return {"rows": len(rows), "unique_ids": len(found),
            "duplicate_ids": len(ids) - len(found),
            "missing_ids": len(expected - found),
            "extra_ids": len(found - expected), "id_sha256": id_hash(ids)}


def build():
    pools, sources, tests = {}, {}, {}
    for domain, count in zip(DOMAINS, EXPECTED):
        source = SOURCE / domain / "train.jsonl"
        rows = read(source)
        if len(rows) != count or any(row["dataset"] != domain or row["split"] != "train" for row in rows):
            raise ValueError(f"Unexpected official train rows: {domain}")
        if len({identity(row) for row in rows}) != count:
            raise ValueError(f"Duplicate source ID: {domain}")
        pools[domain] = rows
        sources[domain] = {"path": str(source.relative_to(ROOT)).replace("\\", "/"),
                           "sha256": sha(source), "count": count}
        test_source = SOURCE / domain / "test.jsonl"
        test_target = DEST / "test" / f"{domain}.jsonl"
        test_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test_source, test_target)
        tests[domain] = {"source": str(test_source.relative_to(ROOT)).replace("\\", "/"),
                         "path": str(test_target.relative_to(ROOT)).replace("\\", "/"),
                         "sha256": sha(test_source), "count": len(read(test_target))}
        assert sha(test_target) == tests[domain]["sha256"]

    global_ids = {identity(row) for domain in DOMAINS for row in pools[domain]}
    assert len(global_ids) == sum(EXPECTED) == 33981
    skewed = [pools[domain][:] for domain in DOMAINS]
    mixed = [[] for _ in range(4)]
    for domain in DOMAINS:
        shuffled = pools[domain][:]
        random.Random(42).shuffle(shuffled)
        q, r = divmod(len(shuffled), 4)
        start = 0
        for client in range(4):
            take = q + (client < r)
            mixed[client].extend(shuffled[start:start + take])
            start += take
        assert start == len(shuffled)

    manifest = {"seed": 42, "client_num": 4, "train_total": len(global_ids),
                "global_train_id_sha256": id_hash(global_ids),
                "train_sources": sources, "official_tests": tests,
                "validation_used": False,
                "dataset_skewed_mapping": {str(i): domain for i, domain in enumerate(DOMAINS)},
                "conditions": {}, "union_checks": {}}
    for condition, clients in (("dataset_skewed", skewed), ("iid_mixed", mixed)):
        manifest["conditions"][condition] = []
        for client, rows in enumerate(clients):
            path = DEST / "train" / condition / f"{client}.jsonl"
            write(path, rows)
            counts = Counter(row["dataset"] for row in rows)
            manifest["conditions"][condition].append({
                "client": client, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha(path), "count": len(rows),
                "dataset_counts": {domain: counts[domain] for domain in DOMAINS},
                "id_sha256": id_hash(identity(row) for row in rows)})
        all_rows = [row for rows in clients for row in rows]
        check = inspect(all_rows, global_ids)
        assert check == {"rows": 33981, "unique_ids": 33981, "duplicate_ids": 0,
                         "missing_ids": 0, "extra_ids": 0,
                         "id_sha256": manifest["global_train_id_sha256"]}
        manifest["union_checks"][condition] = check
    manifest["iid_skewed_same_global_train_ids"] = (
        manifest["union_checks"]["iid_mixed"]["id_sha256"] ==
        manifest["union_checks"]["dataset_skewed"]["id_sha256"])
    assert manifest["iid_skewed_same_global_train_ids"]
    pooled = [row for domain in DOMAINS for row in pools[domain]]
    pooled_path = DEST / "train/centralized/0.jsonl"
    write(pooled_path, pooled)
    manifest["centralized"] = {"path": str(pooled_path.relative_to(ROOT)).replace("\\", "/"),
                               "sha256": sha(pooled_path), "count": len(pooled),
                               "id_sha256": id_hash(map(identity, pooled))}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify()


def verify():
    manifest = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["client_num"] == 4 and manifest["train_total"] == 33981
    global_ids = set()
    for domain in DOMAINS:
        entry = manifest["train_sources"][domain]
        source = ROOT / entry["path"]
        assert sha(source) == entry["sha256"]
        rows = read(source)
        assert len(rows) == entry["count"]
        global_ids.update(map(identity, rows))
        test = manifest["official_tests"][domain]
        assert sha(ROOT / test["source"]) == sha(ROOT / test["path"]) == test["sha256"]
    assert len(global_ids) == 33981 and id_hash(global_ids) == manifest["global_train_id_sha256"]
    for condition in ("dataset_skewed", "iid_mixed"):
        all_rows = []
        entries = manifest["conditions"][condition]
        assert len(entries) == 4
        for client, entry in enumerate(entries):
            path = ROOT / entry["path"]
            rows = read(path)
            assert entry["client"] == client and sha(path) == entry["sha256"]
            assert len(rows) == entry["count"]
            assert id_hash(map(identity, rows)) == entry["id_sha256"]
            assert {d: sum(row["dataset"] == d for row in rows) for d in DOMAINS} == entry["dataset_counts"]
            if condition == "dataset_skewed":
                assert entry["dataset_counts"][DOMAINS[client]] == EXPECTED[client]
            else:
                assert all(abs(entry["dataset_counts"][d] - len(read(ROOT / manifest["train_sources"][d]["path"])) / 4) < 1 for d in DOMAINS)
            all_rows.extend(rows)
        assert inspect(all_rows, global_ids) == manifest["union_checks"][condition]
    assert manifest["iid_skewed_same_global_train_ids"]
    pooled = manifest["centralized"]
    pooled_path = ROOT / pooled["path"]
    assert sha(pooled_path) == pooled["sha256"]
    assert inspect(read(pooled_path), global_ids)["id_sha256"] == pooled["id_sha256"]
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else build()
    for condition in ("dataset_skewed", "iid_mixed"):
        print(condition, [entry["dataset_counts"] for entry in result["conditions"][condition]])
    print("4-client partition: PASS")
