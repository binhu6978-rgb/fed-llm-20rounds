import json, os
import torch
from utils.qa_utils import encode_squad_training_example

DOMAINS = ("squad", "newsqa", "triviaqa", "naturalquestions")

def load_probe_rows(dataset_dir, domain):
    path = os.path.join(dataset_dir, "probe", f"{domain}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

@torch.no_grad()
def probe_loss(model, tokenizer, rows, batch_size=8):
    """Mean teacher-forced NLL over a probe set. Returns a python float."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        encoded = [encode_squad_training_example(tokenizer, r) for r in batch_rows]
        input_ids = torch.tensor([e["input_ids"] for e in encoded]).to(model.device)
        attention_mask = torch.tensor([e["attention_mask"] for e in encoded]).to(model.device)
        labels = torch.tensor([e["labels"] for e in encoded]).to(model.device)
        with torch.autocast('cuda'):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total += out.loss.item() * len(batch_rows)
        count += len(batch_rows)
    return total / count

def probe_loss_all_domains(model, tokenizer, dataset_dir, batch_size=8):
    return {d: probe_loss(model, tokenizer, load_probe_rows(dataset_dir, d), batch_size) for d in DOMAINS}