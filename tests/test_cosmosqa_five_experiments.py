"""Mixed-domain source fidelity, real aggregation and both training recovery paths."""
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from scripts import run_cosmosqa_five_experiments as r


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lora_factor = torch.nn.Parameter(torch.zeros(1))


class EncodingTests(unittest.TestCase):
    def test_all_frozen_inputs_and_mixed_encodings(self):
        from utils.model_utils import load_tokenizer
        from utils import five_client_mcq, cosmosqa_mcq
        self.assertTrue(r.data.verify()['passed'])
        tokenizer = load_tokenizer(SimpleNamespace(model_path=str(r.ROOT / r.data.config()['model_path']), mcq=True))
        datasets = {}
        for d in r.DOMAINS:
            rows = r.data.rows(r.ROOT / r.data.config()['prepared_dir'] / 'train' / f'{d}.jsonl')[:3]
            datasets[d] = r.encoding.TrainingDataset(rows, tokenizer)
            for row, encoded in zip(rows, datasets[d].encoded):
                original = cosmosqa_mcq if d == 'cosmosqa' else five_client_mcq
                prefix = original.encode_prompt(tokenizer, row)
                target = original.continuation_ids(tokenizer, row, row['correct_answer']) + [tokenizer.eos_token_id]
                self.assertEqual(encoded['input_ids'], prefix + target)
                self.assertEqual(encoded['labels'], [-100] * len(prefix) + target)
                self.assertEqual(encoded['native_key'], r.data.native_key(row))
        pooled = r.MixedDataset(datasets)
        self.assertEqual(len(pooled), 15)
        self.assertEqual([x['native_key'] for x in pooled.encoded], [r.data.native_key(row) for row in pooled.rows])
        self.assertTrue(all(pooled.encoded[i] is datasets[d].encoded[j]
                            for i, (d, j) in enumerate((d, j) for d in r.DOMAINS for j in range(3))))
        r.encoding.clear_cache()


class RecoveryTests(unittest.TestCase):
    def exercise(self, crash=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); master = root / 'master'
            ex = r.Experiment.__new__(r.Experiment)
            ex.model = TinyModel(); ex.initial = r.shared.adapter(ex.model)
            ex.tokenizer = None
            ex.smoke, ex.rounds, ex.epochs = True, 2, 2
            ex.outputs = {m: master / m for m in r.OUTPUTS}
            ex.protocol, ex.parameter_info = {'fixture': True}, {}
            ex.counts = dict(zip(r.DOMAINS, (9, 8, 17, 9, 8)))
            ex.weights = {d: n / 51 for d, n in ex.counts.items()}
            ex.train_rows = {d: [dict(dataset=d, id=str(i)) for i in range(n)] for d, n in ex.counts.items()}
            ex.supervised_counts = {d: 2 * n for d, n in ex.counts.items()}
            ex.datasets = {d: SimpleNamespace(rows=rows, encoded=[{'native_key': r.data.native_key(row)} for row in rows])
                           for d, rows in ex.train_rows.items()}
            ex.args = SimpleNamespace(seed=42, bs=1, cn=5, suffix='unused')
            failures = {crash} if crash else set(); attempts = []

            def score(self, directory, domains, epoch, checkpoint):
                if ('score', epoch, directory.name) in failures:
                    failures.remove(('score', epoch, directory.name)); raise RuntimeError('simulated evaluation interruption')
                values = {d + '_accuracy': self.model.lora_factor.item() / 100 + i / 1000 for i, d in enumerate(domains)}
                values.update(macro_accuracy=sum(values.values()) / len(domains), worst_domain_accuracy=min(values.values()))
                r.data.dump(directory / 'evaluation_provenance.json', dict(domains=list(domains),
                    lora_tensor_sha256=r.shared.tensor_hash(r.shared.adapter(self.model))))
                r.data.dump(directory / 'result.json', values)
                r.data.write_rows(directory / 'evaluation/metrics.jsonl', [dict(round=epoch, **values)])
                return values

            def base(self):
                self.load(self.initial)
                directory = self.init_output('base'); path = directory / 'initial_lora.pt'
                r.shared.save_state(path, self.initial)
                return self.score(directory, r.DOMAINS, 0, path)

            def client(self, d, directory, server, method):
                return SimpleNamespace(id=r.DOMAINS.index(d), domain=d, directory=directory,
                    server=server, dataset={'train': self.train_rows[d]})

            def train(self, c, d):
                t = c.server.round + 1; attempts.append(('fedavg', t, d))
                with torch.no_grad(): self.model.lora_factor.add_(c.id + 1)
                n = self.counts[d]
                row = dict(round=t, client=c.id, samples=n, optimizer_updates=math.ceil(n / 8),
                    supervised_tokens=2 * n, ids=[r.data.native_key(row) for row in self.train_rows[d]])
                with (c.directory / 'exposure.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                if ('train', t, d) in failures:
                    failures.remove(('train', t, d)); raise RuntimeError('simulated incomplete epoch')
                return r.shared.adapter(self.model), dict(train_loss=.1, samples=n, optimizer_updates=math.ceil(n / 8))

            class Loader:
                drop_last = False
                def __init__(self, n): self.n = n
                def __len__(self): return self.n

            class Trainer:
                def __init__(self, args, dataset, c):
                    self.args, self.dataset, self.client = args, dataset['train'], c
                    self.train_loader = Loader(len(self.dataset))
                def train(self, model):
                    epoch = self.client.server.round + 1; attempts.append(('centralized', epoch, 'all'))
                    with torch.no_grad(): model.lora_factor.add_(1)
                    self.client.last_seen_count = len(self.dataset)
                    self.client.last_optimizer_updates = math.ceil(len(self.dataset) / 8)
                    row = dict(round=epoch, client=0, samples=len(self.dataset),
                        optimizer_updates=math.ceil(len(self.dataset) / 8), supervised_tokens=2 * len(self.dataset),
                        ids=[r.data.native_key(row) for row in self.dataset.rows])
                    with (Path(self.args.suffix) / 'exposure.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                    if ('central', epoch, 'all') in failures:
                        failures.remove(('central', epoch, 'all')); raise RuntimeError('simulated incomplete central epoch')
                    return 1 / epoch

            ex.client, ex.train, ex.score, ex.base = (MethodType(fn, ex) for fn in (client, train, score, base))
            with patch.object(r, 'ROOT', root), patch.object(r, 'MASTER', master), \
                 patch.object(r.shared, 'ROOT', root), patch.object(r.shared, 'MASTER', master), \
                 patch.object(r.shared, 'DOMAINS', r.DOMAINS), patch.object(r.shared, 'NAMES', r.NAMES), \
                 patch('utils.train_utils.Trainer', Trainer), redirect_stdout(StringIO()):
                if crash and crash[0] != 'central':
                    with self.assertRaisesRegex(RuntimeError, 'simulated'): ex.fedavg()
                ex.fedavg()
                expected = 2 * sum(ex.weights[d] * (i + 1) for i, d in enumerate(r.DOMAINS))
                self.assertAlmostEqual(ex.model.lora_factor.item(), expected, places=5)
                final_hash = r.shared.tensor_hash(r.shared.adapter(ex.model))
                ex.fedavg()
                self.assertEqual(r.shared.tensor_hash(r.shared.adapter(ex.model)), final_hash)
                if crash and crash[0] == 'central':
                    with self.assertRaisesRegex(RuntimeError, 'simulated'): ex.centralized()
                ex.centralized()
                self.assertEqual(ex.model.lora_factor.item(), 2.)
                ex.load(ex.initial); ex.centralized()
                self.assertEqual(ex.model.lora_factor.item(), 2.)
                self.assertTrue(ex.budget_audit()['passed']); self.assertTrue(ex.lifecycle_audit()['passed'])
                self.assertEqual(len(r.data.rows(ex.outputs['fedavg'] / 'exposure.jsonl')), 10)
                self.assertEqual(len(r.data.rows(ex.outputs['centralized'] / 'exposure.jsonl')), 2)
                self.assertEqual(len(attempts), 13 if crash and crash[0] in ('train', 'central') else 12)
                self.assertFalse((ex.outputs['fedavg'] / 'round_01/local_accuracy_matrix.json').exists())

    def test_real_five_client_aggregation_and_fresh_central_start(self): self.exercise()
    def test_saved_local_evaluation_resume(self): self.exercise(('score', 1, 'client_hellaswag'))
    def test_after_aggregation_evaluation_resume(self): self.exercise(('score', 1, 'global'))
    def test_uncommitted_client_epoch_rollback(self): self.exercise(('train', 1, 'race'))
    def test_uncommitted_pooled_epoch_rollback(self): self.exercise(('central', 1, 'all'))


class SequenceTests(unittest.TestCase):
    def exercise(self, fail=False):
        from scripts import run_cosmosqa_five_sequence as q
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); master = root / 'suite'; cosmos = root / 'cosmos'
            stages = []
            def wait(pid, script):
                self.assertEqual(pid, 123); self.assertEqual(script, 'run_cosmosqa_sanity.py')
                cosmos.mkdir(parents=True)
                (cosmos / 'completed.txt').write_text('complete')
                q.data.dump(cosmos / 'status.json', {'stage': 'complete'})
                q.data.dump(cosmos / 'protocol_audit.json', {'passed': True})
            def child(stage, args):
                self.assertTrue((cosmos / 'completed.txt').exists())
                stages.append(stage)
                if fail and stage == 'fedavg': raise RuntimeError('simulated failed method')
                if stage == 'smoke': q.data.dump(master / 'protocol_audit.json', {'passed': True})
                if stage in ('fedavg', 'centralized'):
                    (master / stage).mkdir(parents=True)
                    (master / stage / 'completed.txt').write_text('complete')
            with patch.object(q, 'QUEUE', master / 'sequence'), patch.object(q.data, 'MASTER', master), \
                 patch.object(q.cosmos, 'MASTER', cosmos), patch.object(q.data, 'prepare'), \
                 patch.object(q, 'wait_process', side_effect=wait), patch.object(q, 'child', side_effect=child), \
                 patch.object(q, 'patch_cosmos_diagnostic_report'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'simulated'): q.sequence(123)
                    self.assertEqual(stages, ['smoke', 'fedavg'])
                    self.assertFalse((master / 'centralized').exists())
                else:
                    q.sequence(123)
                    self.assertEqual(stages, ['smoke', 'fedavg', 'centralized', 'final_report'])
                    self.assertEqual(q.data.read(master / 'sequence/status.json')['stage'], 'complete')
    def test_local_then_fedavg_then_centralized(self): self.exercise()
    def test_failure_stops_subsequent_method(self): self.exercise(fail=True)


if __name__ == '__main__': unittest.main()
