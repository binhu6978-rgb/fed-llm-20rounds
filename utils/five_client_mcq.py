"""Reuse original MCQ and RACE passage encoders; no scoring implementation."""
import json
from functools import lru_cache
from pathlib import Path

from utils import mcq_utils as original
from utils.race_mcq import RaceTrainingDataset, encode_race

ROOT = Path(__file__).resolve().parents[1]
OriginalTrainingDataset = original.MCQTrainingDataset
original_prefix = original.encode_prompt
original_continuation = original.continuation_ids


def max_length():
    # Preserve existing RACE's model positional bound, without adding a new cap.
    return json.loads((ROOT / 'dataset/mcq_balanced4000/race/manifest.json').read_text())['max_length']


@lru_cache(maxsize=7000)
def race_encoding(tokenizer, row_json):
    return encode_race(tokenizer, json.loads(row_json), max_length())


def encoded_race(tokenizer, row):
    return race_encoding(tokenizer, json.dumps(row, sort_keys=True, ensure_ascii=False))


def encode_prompt(tokenizer, row):
    return encoded_race(tokenizer, row)[0] if row['dataset'] == 'race' else original_prefix(tokenizer, row)


def continuation_ids(tokenizer, row, letter):
    return (encoded_race(tokenizer, row)[1]['ABCD'.index(letter)] if row['dataset'] == 'race'
            else original_continuation(tokenizer, row, letter))


class TrainingDataset:
    def __init__(self, rows, tokenizer):
        if rows and rows[0]['dataset'] == 'race':
            delegate = RaceTrainingDataset(rows, tokenizer, max_length())
        else:
            delegate = OriginalTrainingDataset(rows, tokenizer)
        self.rows, self.encoded = delegate.rows, delegate.encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]
