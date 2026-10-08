"""Full continuation likelihood, without free generation or extractive-QA metrics."""
import csv
import json
from pathlib import Path
import torch
from utils.mcq_utils import DOMAINS, LETTERS, read_jsonl, encode_prompt, continuation_ids


class MCQEvaluator:
    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer
        self.domains = tuple(getattr(args, 'mcq_domains', DOMAINS))
        self.rows = {d: read_jsonl(Path(args.mcq_dataset_dir) / 'test' / f'{d}.jsonl') for d in self.domains}
        self.output = Path(args.suffix) / 'evaluation'
        self.output.mkdir(parents=True, exist_ok=True)
        self.encoded = {}
        for domain, rows in self.rows.items():
            self.encoded[domain] = [
                (encode_prompt(tokenizer, row), [continuation_ids(tokenizer, row, l) for l in LETTERS])
                for row in rows
            ]

    @torch.no_grad()
    def evaluate(self, model, round_number):
        model.eval()
        results, predictions = {}, []
        for domain in self.domains:
            rows = self.rows[domain]
            for start in range(0, len(rows), self.args.eval_batch_size):
                batch = rows[start:start + self.args.eval_batch_size]
                encoded = self.encoded[domain][start:start + len(batch)]
                # If all continuations contain one token, one prompt forward scores all four.
                single = all(len(ids) == 1 for _, candidates in encoded for ids in candidates)
                sequences = [p for p, _ in encoded] if single else [p + c for p, cs in encoded for c in cs]
                width = max(map(len, sequences))
                ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id, dtype=torch.long, device=model.device)
                mask = torch.zeros_like(ids)
                for j, seq in enumerate(sequences):
                    ids[j, :len(seq)] = torch.tensor(seq, device=model.device)
                    mask[j, :len(seq)] = 1
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                scores = []
                for j, (prefix, candidates) in enumerate(encoded):
                    values = []
                    for k, target in enumerate(candidates):
                        sequence_index = j if single else 4 * j + k
                        positions = torch.arange(len(prefix) - 1, len(prefix) - 1 + len(target), device=model.device)
                        logp = logits[sequence_index, positions].float().log_softmax(dim=-1)
                        tokens = torch.tensor(target, device=model.device)
                        values.append(logp.gather(1, tokens[:, None]).sum().item())
                    scores.append(values)
                del logits
                for row, values in zip(batch, scores):
                    predicted = LETTERS[max(range(4), key=lambda k: values[k])]
                    predictions.append({
                        'id': row['id'], 'domain': domain, 'gold': row['correct_answer'],
                        'prediction': predicted, **{f'logprob_{l}': v for l, v in zip(LETTERS, values)},
                        'correct': int(predicted == row['correct_answer']),
                    })
            current = [p for p in predictions if p['domain'] == domain]
            results[domain + '_accuracy'] = sum(p['correct'] for p in current) / len(current)
        results['macro_accuracy'] = sum(results[d + '_accuracy'] for d in self.domains) / len(self.domains)
        results['worst_domain_accuracy'] = min(results[d + '_accuracy'] for d in self.domains)
        with (self.output / f'round_{round_number:02d}_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        with (self.output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'round': round_number, **results}) + '\n')
        print(f'Round {round_number} Test: ' + ' '.join(f'{k}={v:.6f}' for k, v in results.items()), flush=True)
        return results
