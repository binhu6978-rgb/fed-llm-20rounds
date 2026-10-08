"""CPU fixtures for queue order/failure and client/final-evaluation resume.

All simulated weights, predictions and markers live in TemporaryDirectory.
These tests never train a real model or access official test files.
"""
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import peft
from scripts import run_mcq_controls as controls
from scripts import run_mcq_sequence as queue
from utils import model_utils, seed_utils, train_utils


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        self.lora_factor = torch.nn.Parameter(torch.tensor([3.]))


class FixtureDataset:
    drop_last = False

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)


class ResumeTests(unittest.TestCase):
    def exercise(self, method, crash=None):
        with tempfile.TemporaryDirectory(prefix='mcq_resume_check_') as name:
            root = Path(name).resolve()
            outputs = {m: root / 'exp' / m for m in controls.RUNS}
            (root / 'reports').mkdir()
            rows = {d: [dict(dataset=d, id=str(i)) for i in range(4000)] for d in controls.DOMAINS}
            clients = ([sum((rows[d] for d in controls.DOMAINS), [])] if method == 'centralized' else
                       [sum((rows[d][i*1000:(i+1)*1000] for d in controls.DOMAINS), []) for i in range(4)])
            manifest = dict(clients=[dict(client=i, count=len(client), ids=[controls.key(r) for r in client])
                                    for i, client in enumerate(clients)])
            output = outputs[method]
            output.mkdir(parents=True)
            controls.dump(output / 'partition.json', manifest)
            initial = root / 'models/initial_lora/seed42_r8_alpha32_qv.pt'
            initial.parent.mkdir(parents=True)
            torch.save({'lora_factor': torch.tensor([3.])}, initial)
            baseline = dict(canonical_initial_sha256=controls.sha(initial),
                datasets={d: dict(base_accuracy=0.25, local_epoch10_accuracy=0.5, official_test_count=1)
                          for d in controls.DOMAINS})
            controls.dump(root / 'exp/mcq_base_local_seed42/manifest.json', baseline)
            template = vars(controls.original.args_for_run()).copy()
            attempts, evaluations = [], []
            crashes = {crash} if crash else set()

            class FixtureTrainer:
                def __init__(self, args, dataset, client):
                    self.args, self.dataset, self.client = args, dataset, client
                    self.train_loader = dataset['train']

                def train(self, model):
                    c, round_number = self.client, self.client.server.round + 1
                    attempts.append((round_number, c.id))
                    with torch.no_grad():
                        model.lora_factor.add_(c.id + 1)
                    c.last_seen_count = len(self.dataset['train'])
                    c.last_optimizer_updates = c.last_seen_count // 8
                    ids = [controls.key(r) for r in self.dataset['train'].rows]
                    with (Path(self.args.suffix) / 'exposure.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(round=round_number, client=c.id,
                            samples=len(ids), optimizer_updates=len(ids)//8, ids=ids)) + '\n')
                    if (round_number, c.id) in crashes:
                        crashes.remove((round_number, c.id))
                        raise RuntimeError('simulated crash after exposure, before checkpoint')
                    return 0.1

            def evaluate(model, args, tokenizer, output, baseline):
                evaluations.append(model.lora_factor.item())
                folder = output / 'evaluation'
                folder.mkdir(exist_ok=True)
                scores = {d + '_accuracy': 1. for d in controls.DOMAINS}
                scores.update(round=10, macro_accuracy=1., worst_domain_accuracy=1.)
                (folder / 'metrics.jsonl').write_text(json.dumps(scores) + '\n')
                with (folder / 'round_10_predictions.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['id', 'domain', 'gold', 'prediction', 'correct'])
                    writer.writeheader()
                    writer.writerows(dict(id='fixture_' + d, domain=d, gold='A', prediction='A', correct=1)
                                     for d in controls.DOMAINS)
                return scores

            with (patch.object(controls, 'ROOT', root), patch.object(controls, 'RUNS', outputs),
                  patch.object(controls, 'partition', return_value=(clients, manifest)),
                  patch.object(controls, 'datasets_for', return_value=[FixtureDataset(r) for r in clients]),
                  patch.object(controls, 'evaluate', side_effect=evaluate),
                  patch.object(controls.original, 'verify', return_value=baseline),
                  patch.object(controls.original, 'args_for_run', side_effect=lambda: SimpleNamespace(**copy.copy(template))),
                  patch.object(train_utils, 'Trainer', FixtureTrainer),
                  patch.object(model_utils, 'load_model', side_effect=lambda args: TinyModel()),
                  patch.object(model_utils, 'load_tokenizer', return_value=None),
                  patch.object(peft, 'get_peft_model', side_effect=lambda model, config: model),
                  patch.object(seed_utils, 'set_global_seed'), redirect_stdout(io.StringIO())):
                if crash:
                    with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                        controls.run(method)
                    state = torch.load(output / 'resume.pt', weights_only=True)
                    self.assertEqual(state['completed_round'], 1)
                    self.assertEqual(set(state['pending']), {0})
                controls.run(method)
                target = 13. if method == 'centralized' else 28.
                self.assertEqual(evaluations, [target])
                self.assertTrue((output / 'completed.txt').exists())
                self.assertTrue(controls.read(output / 'exposure_audit.json')['passed'])
                self.assertEqual(len(attempts), 10 * len(clients) + int(crash is not None))
                if crash:
                    # The already committed client is never trained twice.
                    self.assertEqual(attempts.count((2, 0)), 1)
                    self.assertEqual(attempts.count(crash), 2)
                before = list(attempts)
                (output / 'completed.txt').unlink()
                (output / 'evaluation/metrics.jsonl').unlink()
                controls.run(method)
                self.assertEqual(attempts, before)
                # Final-only resume must evaluate the trained checkpoint,
                # rather than the canonical initialization rebuilt on startup.
                self.assertEqual(evaluations, [target, target])

    def test_centralized_final_evaluation_resume(self):
        self.exercise('centralized')

    def test_iid_partial_epoch_rollback_and_final_resume(self):
        self.exercise('iid', crash=(2, 1))


class QueueTests(unittest.TestCase):
    def test_unicode_report_output_uses_utf8(self):
        with tempfile.TemporaryDirectory(prefix='mcq_queue_encoding_') as name:
            root = Path(name).resolve()
            queue_root = root / 'queue'
            queue_root.mkdir()
            log = root / 'unicode.log'
            with patch.object(queue, 'QUEUE', queue_root), redirect_stdout(io.StringIO()):
                queue.child('encoding_fixture', [sys.executable, '-c',
                            "print('Local \u2212 FedAvg: \u96c6\u4e2d\u8bad\u7ec3')"], log)
            self.assertIn('Local \u2212 FedAvg: \u96c6\u4e2d\u8bad\u7ec3', log.read_text(encoding='utf-8'))

    def test_order_and_completion(self):
        with tempfile.TemporaryDirectory(prefix='mcq_queue_check_') as name:
            root = Path(name).resolve()
            queue_root, logs = root / 'queue', root / 'logs'
            queue_root.mkdir()
            logs.mkdir()
            stages = []

            def child(stage, command, log):
                stages.append(stage)
                if stage in ('centralized', 'iid'):
                    marker = root / f'exp/mcq_{stage}_seed42/completed.txt'
                    marker.parent.mkdir(parents=True)
                    marker.write_text('fixture')

            with (patch.object(queue, 'ROOT', root), patch.object(queue, 'QUEUE', queue_root),
                  patch.object(queue, 'LOGS', logs), patch.object(queue, 'wait_for_skewed') as wait,
                  patch.object(queue.psutil, 'process_iter', return_value=[]),
                  patch.object(queue, 'child', side_effect=child), redirect_stdout(io.StringIO())):
                queue.sequence(123)
                wait.assert_called_once_with(123)
            self.assertEqual(stages, ['centralized', 'iid', 'final_report'])
            self.assertEqual(queue.read(queue_root / 'status.json')['stage'], 'complete')

    def test_real_subprocess_failure_prevents_next_job(self):
        with tempfile.TemporaryDirectory(prefix='mcq_queue_failure_') as name:
            root = Path(name).resolve()
            queue_root, logs = root / 'queue', root / 'logs'
            queue_root.mkdir()
            logs.mkdir()
            stages, actual_child = [], queue.child

            def failing_child(stage, command, log):
                stages.append(stage)
                return actual_child(stage, [sys.executable, '-c', 'raise SystemExit(17)'], log)

            with (patch.object(queue, 'QUEUE', queue_root), patch.object(queue, 'LOGS', logs),
                  patch.object(queue, 'wait_for_skewed'), patch.object(queue.psutil, 'process_iter', return_value=[]),
                  patch.object(queue, 'child', side_effect=failing_child), redirect_stdout(io.StringIO())):
                with self.assertRaisesRegex(RuntimeError, 'exit code 17'):
                    queue.sequence(None)
            self.assertEqual(stages, ['centralized'])
            self.assertFalse((queue_root / 'completed.txt').exists())


if __name__ == '__main__':
    unittest.main()
