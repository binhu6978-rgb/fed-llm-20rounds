"""Reuse existing encoders for all five domains, including CosmosQA passages."""
from utils import mcq_utils as original
from utils import five_client_mcq as previous
from utils import cosmosqa_mcq as cosmos

OriginalDataset = original.MCQTrainingDataset
original_prefix = original.encode_prompt
original_continuation = original.continuation_ids


def encode_prompt(tokenizer, row):
    if row['dataset'] == 'cosmosqa': return cosmos.encode_prompt(tokenizer, row)
    if row['dataset'] == 'race': return previous.encode_prompt(tokenizer, row)
    return original_prefix(tokenizer, row)


def continuation_ids(tokenizer, row, letter):
    if row['dataset'] == 'cosmosqa': return cosmos.continuation_ids(tokenizer, row, letter)
    if row['dataset'] == 'race': return previous.continuation_ids(tokenizer, row, letter)
    return original_continuation(tokenizer, row, letter)


class TrainingDataset:
    def __init__(self, rows, tokenizer):
        from unittest.mock import patch
        with patch.object(original, 'encode_prompt', encode_prompt), \
             patch.object(original, 'continuation_ids', continuation_ids):
            delegate = OriginalDataset(rows, tokenizer)
        self.rows, self.encoded = delegate.rows, delegate.encoded

    def __len__(self): return len(self.encoded)
    def __getitem__(self, index): return self.encoded[index]


def clear_cache():
    cosmos.encoded.cache_clear()
    previous.race_encoding.cache_clear()
