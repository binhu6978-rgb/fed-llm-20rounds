"""CosmosQA uses the project's existing passage encoder, without new scoring."""
import json
from functools import lru_cache
from pathlib import Path

from utils.race_mcq import encode_race

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=6000)
def encoded(tokenizer, serialized_row):
    # Same positional bound and exact prompt/continuations as the existing RACE client.
    bound = json.loads((ROOT / 'models/models--llama3.2-1B/config.json').read_text())['max_position_embeddings']
    return encode_race(tokenizer, json.loads(serialized_row), bound)


def encode_prompt(tokenizer, row):
    return encoded(tokenizer, json.dumps(row, ensure_ascii=False, sort_keys=True))[0]


def continuation_ids(tokenizer, row, letter):
    return encoded(tokenizer, json.dumps(row, ensure_ascii=False, sort_keys=True))[1]['ABCD'.index(letter)]
