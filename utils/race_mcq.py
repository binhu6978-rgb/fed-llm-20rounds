"""RACE-only passage encoding; existing MCQ formats are untouched."""
from pathlib import Path

from utils.mcq_utils import LETTERS, read_jsonl
from utils.mcq_eval import MCQEvaluator


def race_prompt(row):
    return (f"Passage: {row['context']}\n\nQuestion: {row['question']}\n"
            + '\n'.join(f"{letter}. {row['option_' + letter.lower()]}" for letter in LETTERS)
            + '\n\nAnswer:')


def target_ids(tokenizer, text, letter):
    prefix = tokenizer.encode(text, add_special_tokens=False)
    combined = tokenizer.encode(text + ' ' + letter, add_special_tokens=False)
    if combined[:len(prefix)] != prefix:
        raise ValueError('RACE continuation changed the prompt token boundary')
    return combined[len(prefix):]


def encode_race(tokenizer, row, max_length):
    """Reserve BOS, all four continuations and EOS; truncate only passage."""
    text = race_prompt(row)
    prefix = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
    candidates = [target_ids(tokenizer, text, letter) for letter in LETTERS]
    full_length = len(prefix) + max(map(len, candidates)) + 1
    truncated = False
    if full_length > max_length:
        empty = dict(row, context='')
        fixed = race_prompt(empty)
        fixed_prefix = [tokenizer.bos_token_id] + tokenizer.encode(fixed, add_special_tokens=False)
        reserve = max(len(target_ids(tokenizer, fixed, letter)) for letter in LETTERS) + 1
        budget = max_length - len(fixed_prefix) - reserve
        if budget < 1:
            raise ValueError(f"Question/options alone exceed budget: {row['id']}")
        passage = tokenizer.encode(row['context'], add_special_tokens=False)
        # Re-encode whole text to respect boundary merges after passage truncation.
        while budget > 0:
            shortened = dict(row, context=tokenizer.decode(passage[:budget], skip_special_tokens=False))
            text = race_prompt(shortened)
            prefix = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
            candidates = [target_ids(tokenizer, text, letter) for letter in LETTERS]
            overflow = len(prefix) + max(map(len, candidates)) + 1 - max_length
            if overflow <= 0:
                break
            budget -= max(1, overflow)
        else:
            raise ValueError(f"Cannot retain passage within budget: {row['id']}")
        truncated = True
    assert len(prefix) + max(map(len, candidates)) + 1 <= max_length
    return prefix, candidates, {'full_tokens': full_length, 'passage_truncated': truncated,
                                'question_options_truncated': False, 'answer_truncated': False}


class RaceTrainingDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.rows, self.encoded = rows, []
        for row in rows:
            prefix, candidates, _ = encode_race(tokenizer, row, max_length)
            target = candidates[LETTERS.index(row['correct_answer'])] + [tokenizer.eos_token_id]
            self.encoded.append({'input_ids': prefix + target,
                                 'labels': [-100] * len(prefix) + target,
                                 'attention_mask': [1] * (len(prefix) + len(target)),
                                 'native_key': 'race:' + str(row['id'])})

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]


class RaceEvaluator(MCQEvaluator):
    """Reuse unchanged conditional-likelihood scorer, with passage prefixes."""
    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer
        self.domains = ('race',)
        self.rows = {'race': read_jsonl(args.race_test_path)}
        self.output = Path(args.suffix) / 'evaluation'
        self.output.mkdir(parents=True, exist_ok=True)
        self.encoded = {'race': [encode_race(tokenizer, row, args.max_length)[:2]
                                 for row in self.rows['race']]}
