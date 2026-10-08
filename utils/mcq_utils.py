"""Plain-text, original-order MCQ protocol for the frozen BhashaBench pool."""
import json
from pathlib import Path

DOMAINS = ('ayur', 'legal', 'krishi')
LETTERS = 'ABCD'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt(row):
    return (f"Question: {row['question']}\n"
            + '\n'.join(f"{letter}. {row['option_' + letter.lower()]}" for letter in LETTERS)
            + '\n\nAnswer:')


def encode_prompt(tokenizer, row):
    return [tokenizer.bos_token_id] + tokenizer.encode(prompt(row), add_special_tokens=False)


def continuation_ids(tokenizer, row, letter):
    prefix = tokenizer.encode(prompt(row), add_special_tokens=False)
    combined = tokenizer.encode(prompt(row) + ' ' + letter, add_special_tokens=False)
    # The requested conditional continuation must have an unchanged prompt prefix.
    if combined[:len(prefix)] != prefix:
        raise RuntimeError('MCQ prompt/continuation token boundary changed')
    return combined[len(prefix):]


class MCQTrainingDataset:
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.encoded = []
        for row in rows:
            prefix = encode_prompt(tokenizer, row)
            target = continuation_ids(tokenizer, row, row['correct_answer']) + [tokenizer.eos_token_id]
            self.encoded.append({
                'input_ids': prefix + target,
                'attention_mask': [1] * (len(prefix) + len(target)),
                'labels': [-100] * len(prefix) + target,
                'native_key': row.get('domain', row.get('dataset')) + ':' + str(row['id']),
            })

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]
