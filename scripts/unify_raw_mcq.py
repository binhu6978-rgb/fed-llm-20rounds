"""Convert the locally downloaded MedQA, LogiQA, OpenBookQA and SciQ splits.

Run with the project's fd environment:
    D:\\software\\minicoonda\\envs\\fd\\python.exe scripts/unify_raw_mcq.py

The script reads raw files only. LogiQA's official Train/Eval/Test text files must
be present in data/raw_mcq/logiqa/source (or directly in its dataset directory).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_mcq"
OUT = ROOT / "data" / "unified_mcq"
REPORT = ROOT / "reports" / "unified_mcq_dataset_summary.md"
DATASETS = ("medqa", "logiqa", "openbookqa", "sciq")
SPLITS = ("train", "validation", "test")
LETTERS = "ABCD"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def source_files(dataset: str, split: str) -> list[Path]:
    root = RAW / dataset
    if dataset == "medqa":
        path = root / "questions" / f"{split}.jsonl"
        return [path] if path.is_file() else []
    if dataset == "logiqa":
        filename = {"train": "Train.txt", "validation": "Eval.txt", "test": "Test.txt"}[split]
        return [path for path in (root / "source" / filename, root / filename) if path.is_file()][:1]
    folder = root / ("main" if dataset == "openbookqa" else "data")
    return sorted(folder.glob(f"{split}-*.parquet"))


def parquet_records(paths: list[Path]):
    for path in paths:
        table = pq.read_table(path)
        for index, row in enumerate(table.to_pylist(), 1):
            yield path, index, row


def medqa_records(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                try:
                    yield path, index, json.loads(line)
                except json.JSONDecodeError:
                    yield path, index, {"_parse_error": "invalid_json", "_raw_line": line.rstrip("\n")}


def logiqa_records(paths: list[Path]):
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        for index, block in enumerate((part for part in re.split(r"\r?\n\s*\r?\n", text) if part.strip()), 1):
            yield path, index, {"lines": block.strip().splitlines()}


def raw_records(dataset: str, paths: list[Path]):
    if dataset == "medqa":
        return medqa_records(paths)
    if dataset == "logiqa":
        return logiqa_records(paths)
    return parquet_records(paths)


def extract(dataset: str, split: str, index: int, row: dict) -> tuple[str, str, list[str], str]:
    """Return (native/stable id, question, A-D choices, answer letter)."""
    generated_id = f"{split}-{index:06d}"
    if dataset == "medqa":
        if row.get("_parse_error"):
            raise ValueError(row["_parse_error"])
        options = row.get("options")
        if not isinstance(options, dict) or set(options) != set(LETTERS):
            raise ValueError("options_not_exactly_A_B_C_D")
        answer = clean(row.get("answer_idx")).upper()
        if answer not in LETTERS:
            raise ValueError("invalid_answer_idx")
        if clean(row.get("answer")) != clean(options[answer]):
            raise ValueError("answer_text_disagrees_with_answer_idx")
        return generated_id, clean(row.get("question")), [clean(options[key]) for key in LETTERS], answer

    if dataset == "logiqa":
        lines = row["lines"]
        if len(lines) != 7:
            raise ValueError("unexpected_logiqa_block_line_count")
        answer = clean(lines[0]).upper()
        if answer not in LETTERS:
            raise ValueError("invalid_answer_label")
        options = {}
        for line in lines[3:]:
            # The official text also uses e.g. "B, option" for some rows.
            match = re.match(r"^\s*([A-Da-d])\s*[.),]\s*(.*?)\s*$", line)
            if not match:
                raise ValueError("unparseable_option_label")
            key = match.group(1).upper()
            if key in options:
                raise ValueError("duplicate_option_label")
            options[key] = match.group(2)
        if set(options) != set(LETTERS):
            raise ValueError("options_not_exactly_A_B_C_D")
        passage, prompt = (clean(lines[1]), clean(lines[2]))
        if not passage or not prompt:
            raise ValueError("missing_passage_or_question")
        question = f"Passage: {passage}\n\nQuestion: {prompt}"
        return generated_id, question, [clean(options[key]) for key in LETTERS], answer

    if dataset == "openbookqa":
        choices = row.get("choices")
        if not isinstance(choices, dict):
            raise ValueError("missing_choices_struct")
        labels, texts = choices.get("label"), choices.get("text")
        if not isinstance(labels, list) or not isinstance(texts, list) or len(labels) != 4 or len(texts) != 4:
            raise ValueError("option_count_not_four")
        labels = [clean(label).upper() for label in labels]
        if set(labels) != set(LETTERS):
            raise ValueError("options_not_exactly_A_B_C_D")
        answer = clean(row.get("answerKey")).upper()
        if answer not in LETTERS:
            raise ValueError("answer_not_in_options")
        by_label = dict(zip(labels, texts))
        return clean(row.get("id")) or generated_id, clean(row.get("question_stem")), [clean(by_label[key]) for key in LETTERS], answer

    # SciQ has no native ID and no native order for the three distractors.
    # A stable SHA256-derived insertion position avoids making every answer D.
    question = clean(row.get("question"))
    correct = clean(row.get("correct_answer"))
    distractors = [clean(row.get(f"distractor{i}")) for i in (1, 2, 3)]
    position = hashlib.sha256(f"sciq:{split}:{index}:{question}".encode("utf-8")).digest()[0] % 4
    options = distractors.copy()
    options.insert(position, correct)
    return generated_id, question, options, LETTERS[position]


def validate(question: str, options: list[str], answer: str) -> None:
    if not question:
        raise ValueError("empty_question")
    if len(options) != 4:
        raise ValueError("option_count_not_four")
    if any(not option for option in options):
        raise ValueError("empty_option")
    if answer not in LETTERS:
        raise ValueError("answer_not_in_options")
    if options.count(options[LETTERS.index(answer)]) != 1:
        raise ValueError("correct_answer_text_occurs_in_multiple_options")


def raw_option_count(dataset: str, row: dict) -> int | None:
    if dataset == "medqa":
        return len(row["options"]) if isinstance(row.get("options"), dict) else None
    if dataset == "logiqa":
        return max(0, len(row["lines"]) - 3)
    if dataset == "openbookqa":
        choices = row.get("choices")
        return len(choices["text"]) if isinstance(choices, dict) and isinstance(choices.get("text"), list) else None
    return sum(row.get(key) is not None for key in ("correct_answer", "distractor1", "distractor2", "distractor3"))


def openbookqa_config_check() -> dict[str, str]:
    """Check whether 'additional' is the same QA set plus metadata."""
    result = {}
    for split in SPLITS:
        main = source_files("openbookqa", split)
        extra = sorted((RAW / "openbookqa" / "additional").glob(f"{split}-*.parquet"))
        if not extra:
            result[split] = "additional absent"
            continue
        columns = ("id", "question_stem", "choices", "answerKey")
        left = [tuple(json.dumps(row.get(k), sort_keys=True, ensure_ascii=False) for k in columns)
                for _, _, row in parquet_records(main)]
        right = [tuple(json.dumps(row.get(k), sort_keys=True, ensure_ascii=False) for k in columns)
                 for _, _, row in parquet_records(extra)]
        result[split] = "same core QA rows" if left == right else "DIFFERENT core QA rows; additional not merged"
    return result


def fmt_count(values: Counter) -> str:
    return ", ".join(f"{key}: {values[key]}" for key in sorted(values, key=str)) or "—"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    excluded_path = OUT / "excluded_rows.jsonl"
    results = defaultdict(Counter)
    answers = defaultdict(Counter)
    raw_options = defaultdict(Counter)
    reasons = defaultdict(Counter)
    lengths = defaultdict(lambda: Counter())
    source_manifest: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    source_schema: dict[str, set[str]] = defaultdict(set)

    with excluded_path.open("w", encoding="utf-8") as excluded_handle:
        for dataset in DATASETS:
            (OUT / dataset).mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                paths = source_files(dataset, split)
                source_manifest[dataset][split] = paths
                if not paths:
                    results[dataset][f"missing_{split}"] = 1
                    continue
                output_path = OUT / dataset / f"{split}.jsonl"
                with output_path.open("w", encoding="utf-8") as output_handle:
                    for path, index, row in raw_records(dataset, paths):
                        results[dataset][f"raw_{split}"] += 1
                        source_schema[dataset].update(row.keys())
                        raw_options[dataset][raw_option_count(dataset, row)] += 1
                        native_id = clean(row.get("id")) or f"{split}-{index:06d}"
                        try:
                            item_id, question, options, answer = extract(dataset, split, index, row)
                            validate(question, options, answer)
                        except (ValueError, TypeError, KeyError, IndexError) as error:
                            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                            excluded = {"dataset": dataset, "split": split, "id": native_id,
                                        "source_file": str(path.relative_to(ROOT)), "source_row_index": index,
                                        "reason": reason, "raw": row}
                            excluded_handle.write(json.dumps(excluded, ensure_ascii=False, default=str) + "\n")
                            results[dataset]["excluded"] += 1
                            reasons[dataset][reason] += 1
                            continue
                        item = {"dataset": dataset, "id": item_id, "split": split,
                                "question": question, "option_a": options[0], "option_b": options[1],
                                "option_c": options[2], "option_d": options[3], "correct_answer": answer}
                        output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        results[dataset][split] += 1
                        answers[dataset][answer] += 1
                        lengths[dataset]["question_chars"] += len(question)
                        lengths[dataset]["question_words"] += len(question.split())
                        lengths[dataset]["option_chars"] += sum(len(value) for value in options)
                        lengths[dataset]["option_words"] += sum(len(value.split()) for value in options)

    config_check = openbookqa_config_check()
    lines = ["# Unified MCQ dataset summary", "",
             "Four local raw MCQ datasets were converted without changing existing raw files or official split membership.",
             "Only valid four-choice rows are in the unified files; rejected rows and original content are in "
             "`data/unified_mcq/excluded_rows.jsonl`.", "",
             "## Counts", "",
             "| Dataset | Train | Validation | Test | Excluded | Raw train | Raw total |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        count = results[dataset]
        def split_value(split: str) -> str:
            return "missing" if count[f"missing_{split}"] else str(count[split])
        raw_total = sum(count[f"raw_{split}"] for split in SPLITS)
        lines.append(f"| {dataset} | {split_value('train')} | {split_value('validation')} | "
                     f"{split_value('test')} | {count['excluded']} | {count['raw_train']} | {raw_total} |")
    lines += ["", "The usable training count is the Train column. No sampling, new split or client partition was made.",
              "", "## Observed raw structure and conversion", "",
              "| Dataset | Observed fields | Native answer | ID handling | Question handling |",
              "| --- | --- | --- | --- | --- |",
              "| medqa | " + ", ".join(sorted(source_schema['medqa'])) +
              " | `answer_idx`, cross-checked against `answer` | source lacks ID: split + row index | native question |",
              "| logiqa | text block: label, context, question, A–D lines | first line a–d | source lacks ID: split + block index | context included as `Passage` before `Question` |",
              "| openbookqa | " + ", ".join(sorted(source_schema['openbookqa'])) +
              " | `answerKey` | native `id` | `question_stem` |",
              "| sciq | " + ", ".join(sorted(source_schema['sciq'])) +
              " | `correct_answer` text | source lacks ID: split + row index | native question; `support` omitted |",
              "", "SciQ has three named distractors and one named correct answer, but no A–D ordering. "
              "The script inserts the correct answer at a SHA256-derived stable position based on dataset, split, "
              "source row index and question. This balances answer letters without changing the split or question content.",
              "", "OpenBookQA uses the `main` config. The `additional` config is not appended because it "
              "contains the same QA examples with extra metadata. Core-row comparison: " +
              "; ".join(f"{split}={config_check[split]}" for split in SPLITS) + ".",
              "", "## Four-choice validity and answer distribution", "",
              "| Dataset | Raw option-count distribution | All retained rows have four nonempty options and unique answer text? | A | B | C | D |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        distribution = fmt_count(raw_options[dataset])
        lines.append(f"| {dataset} | {distribution} | Yes, enforced | " +
                     " | ".join(str(answers[dataset][letter]) for letter in LETTERS) + " |")
    lines += ["", "## Mean lengths over all retained splits", "",
              "| Dataset | Mean question characters | Mean question whitespace words | Mean option characters | Mean option whitespace words |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        n = sum(results[dataset][split] for split in SPLITS)
        l = lengths[dataset]
        values = (l["question_chars"] / n, l["question_words"] / n,
                  l["option_chars"] / (4 * n), l["option_words"] / (4 * n)) if n else (0.,) * 4
        lines.append(f"| {dataset} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines += ["", "## Exclusions", "", "| Dataset | Reason | Count |", "| --- | --- | ---: |"]
    for dataset in DATASETS:
        for reason, count in sorted(reasons[dataset].items()):
            lines.append(f"| {dataset} | {reason} | {count} |")
    if not any(reasons.values()):
        lines.append("| All | None | 0 |")
    lines += ["", "## Source files and SHA256", "", "| Dataset / split | Raw file | SHA256 |",
              "| --- | --- | --- |"]
    for dataset in DATASETS:
        for split in SPLITS:
            for path in source_manifest[dataset][split]:
                lines.append(f"| {dataset} / {split} | `{path.relative_to(ROOT).as_posix()}` | `{sha256(path)}` |")
    lines += ["", "LogiQA's local dataset directory initially had only a loader script; the original "
              "`Train.txt`, `Eval.txt` and `Test.txt` were retrieved from the exact official GitHub URLs "
              "embedded in that local script and stored under `data/raw_mcq/logiqa/source/`. "
              "Existing raw files were not modified.", "", "## Recommended balanced training size", ""]
    usable_train = {dataset: results[dataset]["train"] for dataset in DATASETS
                    if not results[dataset]["missing_train"]}
    if len(usable_train) == len(DATASETS):
        recommendation = min(usable_train.values())
        smallest = ", ".join(dataset for dataset, count in usable_train.items() if count == recommendation)
        lines += [f"**{recommendation:,} train rows per dataset** (limited by {smallest}). "
                  "This permits equal domain sample exposure in later Centralized, IID-Mixed, "
                  "Dataset-Skewed Federated and Local comparisons without fabricating rows. "
              "The other datasets would need future downsampling, but none is done here. "
                  "The differing task types, especially passage-based LogiQA and science-focused SciQ/OpenBookQA, "
                  "remain experimental confounds even with equal counts."]
    else:
        recommendation = None
        lines += ["No balanced size can be recommended until every official train source is present."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = {"medqa": "MedQA", "logiqa": "LogiQA", "openbookqa": "OpenBookQA", "sciq": "SciQ"}
    for dataset in DATASETS:
        count = results[dataset]
        values = ["missing" if count[f"missing_{split}"] else str(count[split]) for split in SPLITS]
        print(f"{names[dataset]} train/val/test: {'/'.join(values)}")
    print(f"Recommended balanced train size per dataset: {recommendation if recommendation is not None else 'undetermined'}")
    print("Done.")


if __name__ == "__main__":
    main()
