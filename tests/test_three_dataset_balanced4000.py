"""Fixed sample identity and serial order/failure checks, without real training."""
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import three_dataset_balanced4000_data as data
from scripts import run_three_dataset_balanced4000 as runner


class FixedSampleTests(unittest.TestCase):
    def test_sampling_is_fixed_unique_and_preserves_source_order(self):
        candidates = [dict(dataset='logiqa', id=str(i)) for i in range(10000)]
        first = data.select_fixed_train(candidates)
        self.assertEqual(len(first), 4000)
        self.assertEqual(len({r['id'] for r in first}), 4000)
        self.assertEqual(first, data.select_fixed_train(candidates))
        self.assertEqual([int(r['id']) for r in first], sorted(int(r['id']) for r in first))
        self.assertNotEqual(first, data.select_fixed_train(candidates, seed=43))
        self.assertEqual(len(candidates), 10000)

    def test_insufficient_eligible_train_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eligible train'):
            data.select_fixed_train([{}] * 3999)

    def test_prepared_combined_test_is_identical_to_full_run(self):
        manifest = data.read(data.ROOT / data.config()['prepared_dir'] / 'manifest.json')
        for domain in data.DOMAINS:
            train = data.rows(data.ROOT / data.config()['prepared_dir'] / 'train' / f'{domain}.jsonl')
            test = data.rows(data.ROOT / data.config()['prepared_dir'] / 'test' / f'{domain}.jsonl')
            full_test = data.rows(data.ROOT / 'dataset/three_dataset_full/test' / f'{domain}.jsonl')
            self.assertEqual(test, full_test)
            self.assertEqual(len(train), 4000)
            self.assertEqual([data.native_key(r) for r in train], manifest['audit']['sampling'][domain]['selected_ids'])
            self.assertFalse({data.qa_fingerprint(r) for r in train} & {data.qa_fingerprint(r) for r in test})
        for path, digest in manifest['files'].items():
            self.assertEqual(data.sha(data.ROOT / path), digest)


class SequenceTests(unittest.TestCase):
    def exercise(self, failure=False):
        events = []
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)

            class Fixture:
                def __init__(self, smoke=False):
                    self.kind = 'smoke' if smoke else 'formal'
                    self.counts = {d: 4000 for d in data.DOMAINS}
                    self.test_rows = {d: [{}] for d in data.DOMAINS}
                    self.weights = {d: 1 / 3 for d in data.DOMAINS}
                    events.append(self.kind + '_init')

                def base(self):
                    events.append(self.kind + '_base')

                def local(self):
                    events.append(self.kind + '_local')
                    if failure and self.kind == 'formal':
                        raise RuntimeError('local failed')

                def fedavg(self):
                    events.append(self.kind + '_fedavg')

            with patch.object(runner, 'MASTER', root), patch.object(runner, 'Experiment', Fixture), \
                 patch.object(data, 'prepare', side_effect=lambda: events.append('prepare')), \
                 patch.object(data, 'config', return_value={}), patch.object(runner.shared, 'status'), \
                 patch.object(runner, 'protocol_audit', side_effect=lambda e: events.append('smoke_audit')), \
                 patch.object(runner, 'final_report', side_effect=lambda e: events.append('report') or root/'report.md'), \
                 patch('torch.cuda.empty_cache'), redirect_stdout(StringIO()):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'local failed'):
                        runner.run_all()
                    self.assertFalse((root/'completed.txt').exists())
                else:
                    runner.run_all()
                    self.assertTrue((root/'completed.txt').exists())
        expected = ['prepare', 'smoke_init', 'smoke_base', 'smoke_local', 'smoke_fedavg', 'smoke_audit',
                    'formal_init', 'formal_base', 'formal_local']
        if not failure:
            expected += ['formal_fedavg', 'report']
        self.assertEqual(events, expected)

    def test_serial_order_then_report(self):
        self.exercise()

    def test_failure_stops_remaining_training(self):
        self.exercise(failure=True)


if __name__ == '__main__':
    unittest.main()
