"""Fixed-checkpoint four-dataset MCQ runs; invoke --job all only for full training."""

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
from scripts.build_heterogeneous_mcq_4clients import DEST, DOMAINS, ROOT as PARTITION_ROOT, id_hash, read, sha, verify

assert ROOT == PARTITION_ROOT
SETTINGS = yaml.safe_load((ROOT / "configs/heterogeneous_mcq_4clients.yaml").read_text(encoding="utf-8"))
RUNS = ROOT / "exp/heterogeneous_mcq_4datasets_4clients"
LOGS = ROOT / "logs/heterogeneous_mcq_4datasets_4clients"
REPORT = ROOT / "reports/heterogeneous_mcq_4datasets_4clients_seed42_results.md"
INITIAL = RUNS / "initial_lora.pt"
METHODS = ("base", "local", "centralized", "iid_mixed", "dataset_skewed")


def same_adapter(left, right):
    import torch
    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    return set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)


def config(method, model_path, *, smoke=False, client=None):
    train_method = "dataset_skewed" if method == "local" else method
    output = RUNS / "smoke" / method if smoke else RUNS / method
    if method == "local":
        output = output / f"client_{client:02d}"
    return SimpleNamespace(
        alg="clientlocal" if method == "local" else "centralized" if method == "centralized" else "fedit",
        suffix=str(output) if method == "local" else str(output.relative_to(ROOT / "exp")),
        device=0, dataset="heterogeneous_mcq_4datasets_4clients",
        mcq_train_dir=str(DEST / "train" / ("smoke_" + train_method if smoke else train_method)),
        mcq_dataset_dir=str(DEST), mcq_domains=[DOMAINS[client]] if method == "local" else list(DOMAINS),
        model="llama32_1b_base", model_path=model_path, task_type="CAUSAL_LM",
        seed=SETTINGS["seed"], deterministic=True,
        cn=1 if method in ("local", "centralized") else SETTINGS["client_num"],
        sr=SETTINGS["participation"], rnd=1 if smoke else SETTINGS["rounds_or_epochs"],
        test_gap=1, mcq=True, final_only_test=True, skip_mcq_test=smoke,
        global_test=False, cross_domain_qa=False, round_generation_metrics=False,
        bs=SETTINGS["batch_size"], grad_accum=SETTINGS["gradient_accumulation"],
        epoch=1, step=0, lr=SETTINGS["learning_rate"], eval_batch_size=SETTINGS["eval_batch_size"],
        mode="prototype", lora_rank=SETTINGS["lora_rank"],
        lora_alpha=SETTINGS["lora_alpha"], lora_dropout=SETTINGS["lora_dropout"])


def make_initial(model_path, destination=INITIAL):
    import torch
    from peft import get_peft_model
    from utils.model_utils import load_lora_config, load_model
    from utils.seed_utils import set_global_seed
    args = config("centralized", model_path)
    set_global_seed(42, device=0, deterministic=True)
    model = get_peft_model(load_model(args), load_lora_config(args))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return model


def base(model_path):
    from utils.mcq_eval import MCQEvaluator
    from utils.model_utils import load_tokenizer
    model = make_initial(model_path)
    args = config("centralized", model_path)
    args.suffix = str(RUNS / "base")
    result = MCQEvaluator(args, load_tokenizer(args)).evaluate(model, 0)
    (RUNS / "base/result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def local(client_id, model_path, *, smoke=False):
    import torch
    from alg.clientlocal import Client, Server
    from utils.mcq_eval import MCQEvaluator
    from utils.seed_utils import set_global_seed
    manifest = verify()
    entry = manifest["conditions"]["dataset_skewed"][client_id]
    args = config("local", model_path, smoke=smoke, client=client_id)
    initial = RUNS / "smoke_initial_lora.pt" if smoke else INITIAL
    assert initial.is_file(), "Run --job base first (or --job smoke for smoke setup)"
    set_global_seed(42, device=0, deterministic=True)
    client = Client(client_id, args)
    assert len(client.dataset["train"]) == (8 if smoke else entry["count"])
    set_global_seed(42, device=0, deterministic=True)
    server = Server(args, [client], initial)
    assert server.global_evaluator is None, "Official test was loaded during training"
    epochs = 1 if smoke else SETTINGS["rounds_or_epochs"]
    for epoch in range(1, epochs + 1):
        server.train_client_epoch(client, epoch)
        assert client.last_seen_count == len(client.dataset["train"])
        assert client.last_optimizer_updates == math.ceil(client.last_seen_count / args.grad_accum)
    checkpoint = server.save_client_adapter(client_id)
    assert checkpoint.is_file()
    if smoke:
        return
    # Only the fixed epoch-10 adapter is evaluated; other test sets stay unread.
    result = MCQEvaluator(args, client.tokenizer).evaluate(server.model, epochs)
    payload = {"client": client_id, "dataset": DOMAINS[client_id],
               "train_count": entry["count"], "epoch": epochs,
               "initial_lora_sha256": sha(initial),
               "own_accuracy": result[DOMAINS[client_id] + "_accuracy"]}
    (Path(args.suffix) / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fed(method, model_path, *, smoke=False):
    from main import FedSim
    args = config(method, model_path, smoke=smoke)
    if not smoke:
        assert INITIAL.is_file(), "Run --job base first"
    FedSim(args)
    if not smoke:
        import torch
        actual = torch.load(RUNS / method / "initial_lora.pt", map_location="cpu", weights_only=True)
        canonical = torch.load(INITIAL, map_location="cpu", weights_only=True)
        assert set(actual) == set(canonical)
        assert all(torch.equal(actual[key].cpu(), canonical[key]) for key in canonical)


def smoke(model_path):
    from scripts.build_heterogeneous_mcq_4clients import write
    verify()
    # Tiny train-only subsets; the official tests are never touched.
    for method in ("dataset_skewed", "iid_mixed"):
        for client_id in range(4):
            rows = read(DEST / "train" / method / f"{client_id}.jsonl")[:8]
            write(DEST / "train" / f"smoke_{method}" / f"{client_id}.jsonl", rows)
    pooled = [read(DEST / "train" / "dataset_skewed" / f"{i}.jsonl")[0]
              for i in range(4)]
    write(DEST / "train/smoke_centralized/0.jsonl", pooled)
    smoke_model = make_initial(model_path, RUNS / "smoke_initial_lora.pt")
    del smoke_model
    local(0, model_path, smoke=True)
    local(1, model_path, smoke=True)
    for i in (0, 1):
        assert (RUNS / f"smoke/local/client_{i:02d}/client_{i:02d}/lora_weights.pt").is_file()
        assert same_adapter(RUNS / f"smoke/local/client_{i:02d}/initial_lora.pt", RUNS / "smoke_initial_lora.pt")
    fed("centralized", model_path, smoke=True)
    fed("iid_mixed", model_path, smoke=True)
    assert (RUNS / "smoke/iid_mixed/adapter").exists()  # FedAvg reached aggregation.
    (RUNS / "smoke/verified.txt").write_text("Local, centralized and FedAvg smoke passed.\n", encoding="utf-8")
    print("Smoke test: PASS")


def summarize():
    manifest = verify()
    base_result = json.loads((RUNS / "base/result.json").read_text(encoding="utf-8"))
    local_rows = [json.loads((RUNS / f"local/client_{i:02d}/result.json").read_text(encoding="utf-8")) for i in range(4)]
    assert all(row["client"] == i and row["epoch"] == 10 and row["initial_lora_sha256"] == sha(INITIAL)
               and row["train_count"] == manifest["conditions"]["dataset_skewed"][i]["count"]
               for i, row in enumerate(local_rows))
    for i in range(4):
        entry = manifest["conditions"]["dataset_skewed"][i]
        exposure = read(RUNS / f"local/client_{i:02d}/exposure.jsonl")
        assert len(exposure) == 10
        for epoch, record in enumerate(exposure, 1):
            assert record["round"] == epoch and record["client"] == i
            assert record["samples"] == entry["count"]
            assert record["optimizer_updates"] == math.ceil(entry["count"] / SETTINGS["gradient_accumulation"])
            assert id_hash(record["ids"]) == entry["id_sha256"]
    local_result = {d + "_accuracy": local_rows[i]["own_accuracy"] for i, d in enumerate(DOMAINS)}
    local_result["macro_accuracy"] = sum(local_result[d + "_accuracy"] for d in DOMAINS) / 4
    results = {"base": base_result, "local": local_result}
    for method in ("centralized", "iid_mixed", "dataset_skewed"):
        run = RUNS / method
        assert (run / "completed.txt").is_file()
        evaluations = read(run / "evaluation/metrics.jsonl")
        assert len(evaluations) == 1 and evaluations[0]["round"] == 10
        exposures = read(run / "exposure.jsonl")
        assert len(exposures) == 10 * (1 if method == "centralized" else 4)
        for round_id in range(1, 11):
            ids = [sample for row in exposures if row["round"] == round_id for sample in row["ids"]]
            assert len(ids) == len(set(ids)) == 33981
            assert id_hash(ids) == manifest["global_train_id_sha256"]
        assert same_adapter(run / "initial_lora.pt", INITIAL)
        results[method] = evaluations[0]
    labels = (("base", "Base"), ("local", "ClientLocal / Dataset Specialist"),
              ("centralized", "Centralized-All"), ("iid_mixed", "FedAvg-IID-Mixed"),
              ("dataset_skewed", "FedAvg-Dataset-Skewed"))
    lines = ["# Four-dataset, four-client MCQ experiment", "",
             "Fixed seed 42, official train/test only, no validation, and fixed epoch/round 10 selection.", "",
             "| Method | MedQA | LogiQA | OpenBookQA | SciQ | Macro |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, label in labels:
        row = results[method]
        lines.append("| " + label + " | " + " | ".join(f"{100*row[d+'_accuracy']:.2f}%" for d in DOMAINS)
                     + f" | {100*row['macro_accuracy']:.2f}% |")
    lines += ["", "ClientLocal combines four separate specialist models, each evaluated only on its own dataset. All other trained methods use one shared model on all four tests.", "",
              "| Comparison | MedQA (pp) | LogiQA (pp) | OpenBookQA (pp) | SciQ (pp) | Macro (pp) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, a, b in (("IID − Skewed", "iid_mixed", "dataset_skewed"),
                        ("Local − Skewed", "local", "dataset_skewed"),
                        ("Local − Centralized", "local", "centralized"),
                        ("Centralized − IID", "centralized", "iid_mixed")):
        names = [d + "_accuracy" for d in DOMAINS] + ["macro_accuracy"]
        lines.append("| " + label + " | " + " | ".join(
            f"{100*(results[a][name]-results[b][name]):+.2f}" for name in names) + " |")
    lines += ["", "A positive Local − Skewed gap would show that this shared adapter retained less own-dataset accuracy than the specialist under this protocol. It alone does not prove parameter conflict or forgetting."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUNS / "final_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(REPORT.relative_to(ROOT))


def all_runs(model_path):
    verify()
    LOGS.mkdir(parents=True, exist_ok=True)
    for job in METHODS:
        targets = range(4) if job == "local" else (None,)
        for client in targets:
            name = job if client is None else f"local_{client}"
            marker = RUNS / job / ("result.json" if job == "base" else "completed.txt") if client is None else RUNS / f"local/client_{client:02d}/result.json"
            if marker.is_file():
                continue
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", job,
                       "--model-path", model_path]
            if client is not None:
                command += ["--client", str(client)]
            with (LOGS / f"{name}.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("all", "base", "local", "centralized", "iid_mixed",
                                          "dataset_skewed", "smoke", "report"), default="all")
    parser.add_argument("--client", type=int, choices=range(4))
    parser.add_argument("--model-path", default=SETTINGS["model_path"])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.job == "all":
        all_runs(args.model_path)
    elif args.job == "base":
        verify(); base(args.model_path)
    elif args.job == "local":
        if args.client is None:
            raise ValueError("--client 0..3 is required for an individual local run")
        local(args.client, args.model_path)
    elif args.job in ("centralized", "iid_mixed", "dataset_skewed"):
        verify(); fed(args.job, args.model_path)
    elif args.job == "smoke":
        smoke(args.model_path)
    else:
        summarize()
