"""Run Base, four ClientLocal specialists, and balanced Dataset-Skewed FedAvg."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import yaml
from scripts.build_heterogeneous_mcq_balanced4000 import (
    DEST, DOMAINS, ROOT as BUILD_ROOT, build, id_hash, key, read_jsonl, verify, write_jsonl,
)

assert ROOT == BUILD_ROOT
SETTINGS = yaml.safe_load((ROOT / "configs/heterogeneous_mcq_balanced4000.yaml").read_text(encoding="utf-8"))
RUNS = ROOT / "exp/heterogeneous_mcq_balanced4000_seed42"
LOGS = ROOT / "logs/heterogeneous_mcq_balanced4000_seed42"
REPORT = ROOT / "reports/heterogeneous_mcq_balanced4000_seed42_results.md"
INITIAL = RUNS / "initial_lora.pt"


def same_adapter(left, right):
    import torch
    first = torch.load(left, map_location="cpu", weights_only=True)
    second = torch.load(right, map_location="cpu", weights_only=True)
    return set(first) == set(second) and all(torch.equal(first[name], second[name]) for name in first)


def config(method, model_path, *, smoke=False, client=None):
    output = RUNS / "smoke" / method if smoke else RUNS / method
    if method == "local":
        output = output / f"client_{client:02d}"
    train_name = "smoke_dataset_skewed" if smoke else "dataset_skewed"
    return SimpleNamespace(
        alg="clientlocal" if method == "local" else "fedit",
        suffix=str(output) if method == "local" else str(output.relative_to(ROOT / "exp")),
        device=0,
        dataset="heterogeneous_mcq_4datasets_balanced4000",
        mcq_train_dir=str(DEST / "train" / train_name),
        mcq_dataset_dir=str(DEST),
        mcq_domains=[DOMAINS[client]] if method == "local" else list(DOMAINS),
        model="llama32_1b_base",
        model_path=model_path,
        task_type="CAUSAL_LM",
        seed=SETTINGS["seed"],
        deterministic=True,
        cn=1 if method == "local" else SETTINGS["client_num"],
        sr=SETTINGS["participation"],
        rnd=1 if smoke else SETTINGS["rounds_or_epochs"],
        test_gap=1,
        mcq=True,
        final_only_test=True,
        skip_mcq_test=smoke,
        global_test=False,
        cross_domain_qa=False,
        round_generation_metrics=False,
        bs=SETTINGS["batch_size"],
        grad_accum=SETTINGS["gradient_accumulation"],
        epoch=1,
        step=0,
        lr=SETTINGS["learning_rate"],
        eval_batch_size=SETTINGS["eval_batch_size"],
        mode="prototype",
        lora_rank=SETTINGS["lora_rank"],
        lora_alpha=SETTINGS["lora_alpha"],
        lora_dropout=SETTINGS["lora_dropout"],
    )


def make_initial(model_path, destination=INITIAL):
    import torch
    from peft import get_peft_model
    from utils.model_utils import load_lora_config, load_model
    from utils.seed_utils import set_global_seed
    args = config("fedavg", model_path)
    set_global_seed(42, device=0, deterministic=True)
    model = get_peft_model(load_model(args), load_lora_config(args))
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items() if "lora_" in name}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return model


def run_base(model_path):
    from utils.mcq_eval import MCQEvaluator
    from utils.model_utils import load_tokenizer
    model = make_initial(model_path)
    args = config("fedavg", model_path)
    args.suffix = str(RUNS / "base")
    result = MCQEvaluator(args, load_tokenizer(args)).evaluate(model, 0)
    (RUNS / "base/result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_local(client_id, model_path, *, smoke=False):
    from alg.clientlocal import Client, Server
    from utils.mcq_eval import MCQEvaluator
    from utils.seed_utils import set_global_seed
    manifest = verify()
    entry = manifest["clients"][client_id]
    args = config("local", model_path, smoke=smoke, client=client_id)
    initial = RUNS / "smoke_initial_lora.pt" if smoke else INITIAL
    if not initial.is_file():
        raise FileNotFoundError("Canonical initial LoRA is missing")
    set_global_seed(42, device=0, deterministic=True)
    client = Client(client_id, args)
    expected_size = 8 if smoke else 4000
    assert len(client.dataset["train"]) == expected_size
    assert len(client.trainer.train_loader) == expected_size
    assert client.trainer.train_loader.drop_last is False
    set_global_seed(42, device=0, deterministic=True)
    server = Server(args, [client], initial)
    assert server.global_evaluator is None
    epochs = 1 if smoke else SETTINGS["rounds_or_epochs"]
    for epoch in range(1, epochs + 1):
        server.train_client_epoch(client, epoch)
        assert client.last_seen_count == expected_size
        assert client.last_optimizer_updates == math.ceil(expected_size / args.grad_accum)
    checkpoint = server.save_client_adapter(client_id)
    assert checkpoint.is_file()
    if smoke:
        return
    evaluator = MCQEvaluator(args, client.tokenizer)
    result = evaluator.evaluate(server.model, epochs)
    payload = {
        "client": client_id,
        "dataset": DOMAINS[client_id],
        "train_count": entry["count"],
        "epoch": epochs,
        "initial_lora_matches_canonical": same_adapter(Path(args.suffix) / "initial_lora.pt", INITIAL),
        "own_accuracy": result[DOMAINS[client_id] + "_accuracy"],
    }
    assert payload["initial_lora_matches_canonical"]
    (Path(args.suffix) / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_fedavg(model_path, *, smoke=False):
    from main import FedSim
    manifest = verify()
    weights = manifest["expected_fedavg_weights"]
    print("Expected FedAvg aggregation weights: " + " ".join(
        f"C{i}={weights[str(i)]:.2f}" for i in range(4)), flush=True)
    args = config("fedavg", model_path, smoke=smoke)
    FedSim(args)
    actual_initial = RUNS / ("smoke/fedavg" if smoke else "fedavg") / "initial_lora.pt"
    canonical = RUNS / "smoke_initial_lora.pt" if smoke else INITIAL
    assert same_adapter(actual_initial, canonical)


def smoke(model_path):
    manifest = verify()
    assert [entry["count"] for entry in manifest["clients"]] == [4000] * 4
    assert sum(entry["count"] for entry in manifest["clients"]) == 16000
    assert list(manifest["expected_fedavg_weights"].values()) == [0.25] * 4
    for client_id in range(4):
        rows = read_jsonl(ROOT / manifest["clients"][client_id]["path"])
        write_jsonl(DEST / "train/smoke_dataset_skewed" / f"{client_id}.jsonl", rows[:8])
    smoke_model = make_initial(model_path, RUNS / "smoke_initial_lora.pt")
    del smoke_model
    run_local(0, model_path, smoke=True)
    run_local(1, model_path, smoke=True)
    run_fedavg(model_path, smoke=True)
    exposure = read_jsonl(RUNS / "smoke/fedavg/exposure.jsonl")
    assert len(exposure) == 4
    assert all(row["samples"] == 8 and row["optimizer_updates"] == 1 for row in exposure)
    (RUNS / "smoke/verified.txt").write_text(
        "Partition, loader, Local, equal-weight FedAvg and aggregation smoke passed.\n", encoding="utf-8")
    print("Smoke test: PASS", flush=True)


def summarize():
    manifest = verify()
    base = json.loads((RUNS / "base/result.json").read_text(encoding="utf-8"))
    local_rows = [json.loads((RUNS / f"local/client_{i:02d}/result.json").read_text(encoding="utf-8"))
                  for i in range(4)]
    local = {domain + "_accuracy": local_rows[i]["own_accuracy"] for i, domain in enumerate(DOMAINS)}
    local["macro_accuracy"] = sum(local[d + "_accuracy"] for d in DOMAINS) / 4
    for client_id in range(4):
        entry = manifest["clients"][client_id]
        records = read_jsonl(RUNS / f"local/client_{client_id:02d}/exposure.jsonl")
        assert len(records) == 10
        assert all(row["round"] == epoch and row["client"] == client_id
                   and row["samples"] == 4000 and row["optimizer_updates"] == 500
                   and len(row["ids"]) == len(set(row["ids"])) == 4000
                   and id_hash(row["ids"]) == entry["id_sha256"]
                   for epoch, row in enumerate(records, 1))
    fed_records = read_jsonl(RUNS / "fedavg/exposure.jsonl")
    assert len(fed_records) == 40
    for round_id in range(1, 11):
        current = [row for row in fed_records if row["round"] == round_id]
        assert len(current) == 4
        assert all(row["samples"] == 4000 and row["optimizer_updates"] == 500 for row in current)
        ids = [sample for row in current for sample in row["ids"]]
        assert len(ids) == len(set(ids)) == 16000
        assert id_hash(ids) == manifest["global_train_id_sha256"]
    metrics = read_jsonl(RUNS / "fedavg/evaluation/metrics.jsonl")
    assert len(metrics) == 1 and metrics[0]["round"] == 10
    fedavg = metrics[0]
    assert same_adapter(RUNS / "fedavg/initial_lora.pt", INITIAL)

    results = {"base": base, "clientlocal": local, "fedavg_dataset_skewed": fedavg}
    names = [domain + "_accuracy" for domain in DOMAINS]
    comparisons = {
        "local_minus_base_pp": {d: 100 * (local[d + "_accuracy"] - base[d + "_accuracy"]) for d in DOMAINS},
        "fedavg_minus_base_pp": {d: 100 * (fedavg[d + "_accuracy"] - base[d + "_accuracy"]) for d in DOMAINS},
        "local_minus_fedavg_pp": {d: 100 * (local[d + "_accuracy"] - fedavg[d + "_accuracy"]) for d in DOMAINS},
    }
    lines = [
        "# Balanced-4000 heterogeneous MCQ experiment (seed 42)", "",
        "Each dataset contributes exactly 4,000 official-train examples. Four Dataset-Skewed clients therefore have identical 4,000-example budgets, 500 optimizer updates per epoch/round, and sample-count FedAvg weights of 0.25 each. Validation is unused; official test is evaluated only for Base and fixed Epoch/Round 10.", "",
        "| Method | MedQA | LogiQA | OpenBookQA | SciQ | Macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result, label in ((base, "Base"), (local, "ClientLocal"), (fedavg, "FedAvg-Dataset-Skewed")):
        lines.append("| " + label + " | " + " | ".join(f"{100*result[name]:.2f}%" for name in names)
                     + f" | {100*result['macro_accuracy']:.2f}% |")
    lines += [
        "", "ClientLocal combines four independent specialist models, each evaluated only on its own official test. FedAvg is one shared global LoRA evaluated on all four official tests.", "",
        "| Comparison | MedQA (pp) | LogiQA (pp) | OpenBookQA (pp) | SciQ (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key_name, label in (("local_minus_base_pp", "Local − Base"),
                            ("fedavg_minus_base_pp", "FedAvg − Base"),
                            ("local_minus_fedavg_pp", "Local − FedAvg (retention gap)")):
        lines.append("| " + label + " | " + " | ".join(
            f"{comparisons[key_name][domain]:+.2f}" for domain in DOMAINS) + " |")
    lines += [
        "", "A positive Local − FedAvg gap means that the specialist achieved higher own-domain accuracy than the shared adapter under this controlled budget. It is evidence of aggregation-level capability degradation in this experiment, but does not by itself identify a parameter-conflict, forgetting, or LoRA-subspace mechanism.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUNS / "final_results.json").write_text(
        json.dumps({"results": results, "comparisons": comparisons}, indent=2) + "\n", encoding="utf-8")

    print("Balanced pool: PASS")
    for label, result in (("Base", base), ("ClientLocal", local),
                          ("FedAvg-Dataset-Skewed", fedavg)):
        print(f"\n{label}:")
        for domain in DOMAINS:
            print(f"{domain.capitalize()}: {100*result[domain + '_accuracy']:.2f}%")
        print(f"Macro: {100*result['macro_accuracy']:.2f}%")
    print("\nLocal - FedAvg:")
    for domain in DOMAINS:
        print(f"{domain.capitalize()}: {comparisons['local_minus_fedavg_pp'][domain]:+.2f} pp")
    print("\nDone.")


def run_all(model_path):
    if not (DEST / "manifest.json").is_file():
        build()
    verify()
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [("base", None), *(('local', i) for i in range(4)), ("fedavg", None)]
    log_names = {
        ("base", None): "base.log",
        ("local", 0): "local_medqa.log",
        ("local", 1): "local_logiqa.log",
        ("local", 2): "local_openbookqa.log",
        ("local", 3): "local_sciq.log",
        ("fedavg", None): "fedavg_dataset_skewed.log",
    }
    for job, client in jobs:
        marker = (RUNS / "base/result.json" if job == "base" else
                  RUNS / f"local/client_{client:02d}/result.json" if job == "local" else
                  RUNS / "fedavg/completed.txt")
        if marker.is_file():
            continue
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", job,
                   "--model-path", model_path]
        if client is not None:
            command += ["--client", str(client)]
        with (LOGS / log_names[(job, client)]).open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("all", "prepare", "verify", "smoke", "base", "local", "fedavg", "report"), default="all")
    parser.add_argument("--client", type=int, choices=range(4))
    parser.add_argument("--model-path", default=SETTINGS["model_path"])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.job == "all":
        run_all(args.model_path)
    elif args.job == "prepare":
        build()
    elif args.job == "verify":
        verify(); print("Balanced pool: PASS")
    elif args.job == "smoke":
        smoke(args.model_path)
    elif args.job == "base":
        verify(); run_base(args.model_path)
    elif args.job == "local":
        if args.client is None:
            raise ValueError("--client 0..3 is required")
        run_local(args.client, args.model_path)
    elif args.job == "fedavg":
        verify(); run_fedavg(args.model_path)
    else:
        summarize()
