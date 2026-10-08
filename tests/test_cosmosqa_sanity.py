"""Data leakage, native mapping, existing encoding, and Local resume checks."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import cosmosqa_data as data
from scripts import run_cosmosqa_sanity as runner


def raw_row(index, label=0):
    return dict(id=str(index), context=f'context {index}', question=f'question {index}',
                answer0='first', answer1='second', answer2='third', answer3='fourth', label=label)


def write_csv(path, current):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(current[0]))
        writer.writeheader(); writer.writerows(current)


class DataTests(unittest.TestCase):
    def test_actual_files_and_unlabeled_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_csv(folder / 'train.csv', [raw_row(1)])
            write_csv(folder / 'valid.csv', [raw_row(2, 3)])
            test_row = raw_row(3); del test_row['label']
            (folder / 'test.jsonl').write_text(json.dumps(test_row) + '\n', encoding='utf-8')
            splits, details = data.inspect(folder)
            self.assertEqual(set(splits), {'train', 'validation', 'test'})
            self.assertEqual(next(x for x in details if x['split'] == 'test')['valid_accuracy_labels'], 0)
            self.assertEqual(data.label(splits['validation'][0][0]['label']), 'D')
            self.assertIsNone(data.label(-1)); self.assertIsNone(data.label(True))

    def test_mapping_reordered_duplicates_and_conflicting_labels(self):
        first = raw_row(1, 1)
        repeated = dict(first, id='2', answer0='second', answer1='first', label=0)
        bad = dict(first, id='3', label=3)
        clean, rejected, _ = data.clean([(row, 'train.csv', i, i) for i, row in enumerate((first, repeated))], 'train')
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0]['correct_answer'], 'B')
        self.assertEqual(clean[0]['original_id'], '1')
        self.assertEqual(clean[0]['source_row_index'], 0)
        self.assertEqual(rejected[0]['reasons'], ['normalized_duplicate_qa'])
        clean, rejected, _ = data.clean([(row, 'train.csv', i, i) for i, row in enumerate((first, repeated, bad))], 'train')
        self.assertEqual(clean, [])
        self.assertEqual(len(rejected), 3)
        self.assertTrue(all(x['reasons'] == ['inconsistent_duplicate_labels'] for x in rejected))

    def test_quality_only_and_fixed_selection(self):
        invalids = [dict(raw_row(1), context=''), dict(raw_row(2), answer3='first'), raw_row(3, -1)]
        clean, rejected, _ = data.clean([(row, 'train.csv', i, i) for i, row in enumerate(invalids)], 'train')
        self.assertEqual(clean, [])
        self.assertEqual([r['reasons'] for r in rejected], [['empty_context'], ['repeated_choice_text'], ['invalid_label']])
        candidates = [raw_row(i) for i in range(6000)]
        selected = data.select_fixed_train(candidates, count=4000)
        self.assertEqual(selected, data.select_fixed_train(candidates, count=4000))
        self.assertEqual(len({r['id'] for r in selected}), 4000)
        self.assertEqual([int(r['id']) for r in selected], sorted(int(r['id']) for r in selected))

    def test_existing_passage_encoder_and_original_training_dataset(self):
        from utils import cosmosqa_mcq as encoding, mcq_utils
        from utils.race_mcq import encode_race

        class Tokenizer:
            bos_token_id, eos_token_id = 1, 2
            def encode(self, text, add_special_tokens=False):
                return [ord(char) + 3 for char in text]

        tokenizer = Tokenizer()
        clean, _, _ = data.clean([(raw_row(1, 2), 'train.csv', 0, 0)], 'train')
        row = clean[0]
        expected_prefix, expected_choices, _ = encode_race(tokenizer, row, 131072)
        with patch.object(mcq_utils, 'encode_prompt', encoding.encode_prompt), \
             patch.object(mcq_utils, 'continuation_ids', encoding.continuation_ids):
            dataset = mcq_utils.MCQTrainingDataset(clean, tokenizer)
        self.assertEqual(dataset[0]['input_ids'], expected_prefix + expected_choices[2] + [2])
        self.assertEqual(dataset[0]['labels'], [-100] * len(expected_prefix) + expected_choices[2] + [2])
        self.assertEqual(dataset[0]['native_key'], 'cosmosqa:' + row['id'])
        encoding.encoded.cache_clear()

    def test_preparation_uses_valid_once_and_removes_train_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'raw'; source.mkdir()
            write_csv(source / 'train.csv', [raw_row(i) for i in range(4500)])
            # QA overlap with all train rows but distinct native IDs, plus enough nonoverlapping train.
            write_csv(source / 'valid.csv', [dict(raw_row(i), id=f'valid-{i}') for i in range(1600)])
            test_row = raw_row(99999); del test_row['label']
            (source / 'test.jsonl').write_text(json.dumps(test_row) + '\n', encoding='utf-8')
            # Need 4000 eligible rows after preserving selected validation; extend train.
            write_csv(source / 'train.csv', [raw_row(i) for i in range(6000)])
            (root / 'cosmos_qa.py').write_text('fixture', encoding='utf-8')
            config = dict(data.config(), prepared_dir='prepared')
            with patch.object(data, 'ROOT', root), patch.object(data, 'MASTER', root / 'outputs'), \
                 patch.object(data, 'SOURCE_OVERRIDE', source), patch.object(data, 'config', return_value=config), \
                 patch.object(data, 'core_hashes', return_value={}), patch.object(data, 'statistics', return_value={}):
                manifest = data.prepare()
                self.assertEqual(manifest['audit']['test_official_split'], 'validation')
                self.assertEqual(manifest['audit']['train_overlap_removed'], 1500)
                self.assertEqual(len(data.rows(root / 'prepared/train/cosmosqa.jsonl')), 4000)
                self.assertEqual(len(data.rows(root / 'prepared/test/cosmosqa.jsonl')), 1500)
                self.assertEqual(data.prepare(), manifest)
                self.assertTrue(data.verify()['passed'])


class ResumeTests(unittest.TestCase):
    def test_completed_epochs_resume_and_scope_stop(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ex = runner.Experiment.__new__(runner.Experiment)
            ex.outputs = {'clientlocal': root / 'local'}
            ex.initial = {'lora_A': torch.tensor([0.])}
            ex.init_output = lambda method: ex.outputs[method]
            ex.load = lambda state: setattr(ex, 'model', state)
            ex.client = lambda domain, output, server, method: SimpleNamespace(server=server)
            ex.score = lambda *args: {'cosmosqa_accuracy': .5}
            updates = []
            def train(client, domain):
                epoch = client.server.round + 1
                updates.append(epoch)
                ex.model = {'lora_A': ex.model['lora_A'] + 1}
                return ex.model, dict(train_loss=1 / epoch, samples=4000, optimizer_updates=500)
            ex.train = train
            with patch.object(runner.shared, 'adapter', side_effect=lambda model: model), \
                 patch.object(runner.shared, 'status'):
                ex.local()
                self.assertEqual(updates, list(range(1, 11)))
                self.assertEqual(torch.load(root / 'local/client_cosmosqa/final_lora.pt', weights_only=True)['lora_A'].item(), 10)
                updates.clear()
                ex.local()
                self.assertEqual(updates, [])
                self.assertFalse((root / 'fedavg').exists())
                with self.assertRaisesRegex(RuntimeError, 'outside the authorized'):
                    ex.fedavg()


if __name__ == '__main__':
    unittest.main()
