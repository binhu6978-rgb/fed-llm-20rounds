"""Shared generation evaluation for named-domain extractive-QA experiments.

The module intentionally uses the same Qwen chat prompt and SQuAD-style
normalisation as the existing SQuAD pipeline.  It never concatenates test
domains: each named JSONL is generated and scored independently, followed by
an unweighted macro average.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from transformers import GenerationConfig

from utils.eval_utils import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    normalize_answer,
)
from utils.qa_utils import SQUAD_MAX_NEW_TOKENS, encode_squad_prompt


DOMAINS = ("squad", "newsqa", "triviaqa", "naturalquestions")
RESULT_FIELDS = ("method", "round", "domain", "em", "f1", "containment", "num_samples")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _domains_from_manifest(dataset_dir: Path) -> tuple[str, ...]:
    """Use a dataset-local client map when present; retain legacy fallback."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return DOMAINS
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clients = manifest.get("clients", {})
    try:
        names = tuple(str(clients[str(index)]) for index in range(len(clients)))
    except (KeyError, TypeError):
        return DOMAINS
    return names or DOMAINS


def containment_score(prediction: str, ground_truth: str) -> float:
    """Return whether either non-empty normalized string contains the other."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return float(bool(prediction and ground_truth and (
        prediction in ground_truth or ground_truth in prediction
    )))


def _score_prediction(prediction: str, references: list[str]) -> tuple[float, float, float]:
    return (
        metric_max_over_ground_truths(exact_match_score, prediction, references),
        metric_max_over_ground_truths(f1_score, prediction, references),
        metric_max_over_ground_truths(containment_score, prediction, references),
    )


class CrossDomainQAEvaluator:
    """Generate once per named test domain and append auditable CSV results."""

    def __init__(self, args, reset_results: bool = True):
        self.args = args
        self.dataset_dir = Path("dataset") / args.dataset
        self.domains = _domains_from_manifest(self.dataset_dir)
        self.output_dir = Path(args.suffix) / "evaluation" / "cross_domain"
        self.prediction_dir = self.output_dir / "predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.csv"
        self.macro_path = self.output_dir / "macro_average.csv"
        if reset_results:
            self._initialise_csvs()

    def _initialise_csvs(self) -> None:
        for path in (self.results_path, self.macro_path):
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writeheader()

    def _append(self, path: Path, row: dict) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writerow(row)

    def _generate_domain(self, model, tokenizer, domain: str) -> list[dict]:
        test_path = self.dataset_dir / "test" / f"{domain}.jsonl"
        rows = _read_jsonl(test_path)
        tokenizer.padding_side = "left"
        generation_config = GenerationConfig(
            max_new_tokens=SQUAD_MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        predictions: list[dict] = []
        batch_size = int(getattr(self.args, "eval_batch_size", 8))
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompt_ids = [
                encode_squad_prompt(tokenizer, row["context"], row["question"])
                for row in batch
            ]
            encoding = tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in prompt_ids],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                # Transformers >=4.50 otherwise replaces GenerationConfig
                # values that equal library defaults with model-specific
                # defaults. Qwen's generation_config.json sets do_sample=True,
                # so explicitly disable that fallback to guarantee greedy QA.
                generated = model.generate(
                    **encoding,
                    generation_config=generation_config,
                    use_model_defaults=False,
                )
            prompt_len = encoding["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            for row, prediction in zip(batch, decoded):
                references = row.get("answers") or [row["label"]]
                if isinstance(references, str):
                    references = [references]
                prediction = prediction.strip()
                em, f1, containment = _score_prediction(prediction, list(references))
                predictions.append({
                    "id": row.get("id"),
                    "domain": domain,
                    "context": row["context"],
                    "question": row["question"],
                    "prediction": prediction,
                    "references": list(references),
                    "exact_match": em,
                    "f1": f1,
                    "containment": containment,
                })
        return predictions

    def evaluate_domain(
        self,
        model,
        tokenizer,
        domain: str,
        round_idx: int,
        method: str,
    ) -> dict:
        """Evaluate one named domain and persist its metrics and predictions."""
        if domain not in self.domains:
            raise ValueError(f"Unknown cross-domain QA domain: {domain}")
        model.eval()
        predictions = self._generate_domain(model, tokenizer, domain)
        count = len(predictions)
        if count == 0:
            raise ValueError(f"No evaluation rows found for domain {domain}")
        row = {
            "method": method,
            "round": round_idx,
            "domain": domain,
            "em": 100.0 * sum(p["exact_match"] for p in predictions) / count,
            "f1": 100.0 * sum(p["f1"] for p in predictions) / count,
            "containment": 100.0 * sum(p["containment"] for p in predictions) / count,
            "num_samples": count,
        }
        self._append(self.results_path, row)
        prediction_path = self.prediction_dir / f"round_{round_idx:03d}_{domain}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(
            f"[Round {round_idx}] {domain}: EM={row['em']:.2f} "
            f"F1={row['f1']:.2f} containment={row['containment']:.2f} n={count}"
        )
        return row

    def append_macro(self, per_domain: list[dict], round_idx: int, method: str) -> dict:
        """Append the unweighted macro average of already evaluated domains."""
        if not per_domain:
            raise ValueError("Cannot compute a macro average without domain results")
        macro = {
            "method": method,
            "round": round_idx,
            "domain": "macro_average",
            "em": sum(row["em"] for row in per_domain) / len(per_domain),
            "f1": sum(row["f1"] for row in per_domain) / len(per_domain),
            "containment": sum(row["containment"] for row in per_domain) / len(per_domain),
            "num_samples": sum(row["num_samples"] for row in per_domain),
        }
        self._append(self.results_path, macro)
        self._append(self.macro_path, macro)
        return macro

    def evaluate(self, model, tokenizer, round_idx: int, method: str) -> dict:
        """Evaluate and save a complete per-domain and macro result for a round.

        CSV scores are percentages (0--100), matching the existing SQuAD
        evaluator. Per-example prediction records retain fractional scores.
        """
        per_domain: list[dict] = []
        for domain in self.domains:
            per_domain.append(
                self.evaluate_domain(model, tokenizer, domain, round_idx, method)
            )

        macro = self.append_macro(per_domain, round_idx, method)
        return {"macro_em": macro["em"], "macro_f1": macro["f1"], "macro_containment": macro["containment"]}
