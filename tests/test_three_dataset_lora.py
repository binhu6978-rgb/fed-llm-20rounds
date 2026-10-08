"""CPU protocol checks; temporary fixtures never access experiment weights."""
import csv
import json
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from alg.ftbase import FTBaseServer
from scripts import run_three_dataset_lora as run
from scripts import three_dataset_data as data
from utils import train_utils


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_factor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(use_cache=True)
        self.device = torch.device('cpu')

    def forward(self, input_ids, **unused):
        return SimpleNamespace(loss=(self.lora_factor * input_ids.float()).mean())


class ProtocolTests(unittest.TestCase):
    def test_full_epochs_tail_normalization_constant_lr_and_no_step_cutoff(self):
        # Nonzero server round and step=1 would expose the generic path's decay/cutoff.
        for count in (8, 9, 17):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as name:
                dataset = [dict(input_ids=[1], labels=[1], attention_mask=[1], native_key=str(i))
                           for i in range(count)]
                args = SimpleNamespace(mcq=True, task_type='CAUSAL_LM', bs=1, epoch=1,
                    grad_accum=8, seed=42, lr=1e-4, step=1, suffix=name)
                client = SimpleNamespace(id=0, server=SimpleNamespace(round=4), dataset={'train': dataset})
                gradients, rates = [], []
                real_optimizer = torch.optim.AdamW

                class RecordingAdamW(real_optimizer):
                    def step(self, *a, **kw):
                        gradients.append(self.param_groups[0]['params'][0].grad.item())
                        rates.append(self.param_groups[0]['lr'])
                        return super().step(*a, **kw)

                with patch.object(train_utils, 'AdamW', RecordingAdamW), \
                     patch.object(train_utils, 'autocast', side_effect=lambda *a, **kw: nullcontext()), \
                     redirect_stdout(StringIO()):
                    trainer = train_utils.Trainer(args, client.dataset, client)
                    self.assertFalse(trainer.train_loader.drop_last)
                    trainer.train(TinyModel())
                self.assertEqual(client.last_optimizer_updates, math.ceil(count / 8))
                self.assertEqual(client.last_seen_count, count)
                self.assertEqual(rates, [1e-4] * math.ceil(count / 8))
                self.assertEqual(gradients, [1.] * math.ceil(count / 8))
                record = data.rows(Path(name) / 'exposure.jsonl')[0]
                self.assertEqual(set(record['ids']), {str(i) for i in range(count)})

    def test_sample_weighted_existing_aggregation(self):
        counts = (7179, 4936, 11668)
        values = (1., 5., 10.)
        clients = [SimpleNamespace(id=i, dataset={'train': range(n)},
                   lora={'lora_factor': torch.tensor([v])})
                   for i, (n, v) in enumerate(zip(counts, values))]
        server = SimpleNamespace(sampled_clients=clients, model=TinyModel())
        with redirect_stdout(StringIO()):
            FTBaseServer.aggregate(server)
        expected = sum(n * v for n, v in zip(counts, values)) / sum(counts)
        self.assertAlmostEqual(server.model.lora_factor.item(), expected, places=6)
        self.assertNotAlmostEqual(server.model.lora_factor.item(), sum(values) / 3, places=3)

    def test_qa_identity_is_invariant_to_option_order(self):
        row = dict(question='  Which   answer? ', option_a='one', option_b='two',
                   option_c='three', option_d='four', correct_answer='B')
        moved = dict(row, option_a='two', option_b='one', correct_answer='A')
        self.assertEqual(data.qa_fingerprint(row), data.qa_fingerprint(moved))
        self.assertNotEqual(data.qa_fingerprint(row, ordered=True), data.qa_fingerprint(moved, ordered=True))

    def test_diagnostic_math_undefined_and_unclipped_retention(self):
        row = run.diagnostic_row(1, 'sciq', .4, .5, .7)
        self.assertAlmostEqual(row['local_gain'], .1)
        self.assertAlmostEqual(row['aggregation_gap'], -.2)
        self.assertAlmostEqual(row['global_gain'], .3)
        self.assertAlmostEqual(row['retention_ratio'], 3.)
        for local in (.3, .4):
            self.assertIsNone(run.diagnostic_row(1, 'sciq', .4, local, .5)['retention_ratio'])


class ResumeTests(unittest.TestCase):
    def exercise(self, crash):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            master = root / 'master'
            output = {m: root / m for m in run.OUTPUTS}
            experiment = run.Experiment.__new__(run.Experiment)
            experiment.smoke, experiment.rounds, experiment.epochs = True, 2, 1
            experiment.model = TinyModel()
            experiment.initial = run.adapter(experiment.model)
            experiment.outputs = output
            experiment.counts = dict(zip(run.DOMAINS, (9, 8, 17)))
            experiment.weights = {d: n / 34 for d, n in experiment.counts.items()}
            experiment.train_rows = {d: [dict(dataset=d, id=str(i)) for i in range(n)]
                                     for d, n in experiment.counts.items()}
            experiment.protocol, experiment.parameter_info = {'fixture': True}, {}
            attempts, failures = [], {crash} if crash else set()

            def client_factory(self, domain, directory, server, method):
                return SimpleNamespace(id=run.DOMAINS.index(domain), server=server,
                    domain=domain, directory=directory, dataset={'train': self.train_rows[domain]})

            def train(self, client, domain):
                t = client.server.round + 1
                attempts.append((t, domain))
                with torch.no_grad():
                    self.model.lora_factor.add_(client.id + 1)
                n = self.counts[domain]
                record = dict(round=t, client=client.id, samples=n,
                    optimizer_updates=math.ceil(n / 8), ids=[data.native_key(r) for r in self.train_rows[domain]])
                with (client.directory / 'exposure.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                return run.adapter(self.model), dict(train_loss=.1, samples=n, optimizer_updates=math.ceil(n / 8))

            def score(self, directory, domains, t, checkpoint):
                if directory.parent.name.startswith('round_') and (t, directory.name) in failures:
                    failures.remove((t, directory.name))
                    raise RuntimeError('simulated evaluation interruption after saved checkpoint')
                value = self.model.lora_factor.item() / 100
                result = {d + '_accuracy': value for d in domains}
                result.update(macro_accuracy=value, worst_domain_accuracy=value)
                data.dump(directory / 'evaluation_provenance.json', dict(domains=list(domains),
                    lora_tensor_sha256=run.tensor_hash(run.adapter(self.model))))
                data.dump(directory / 'result.json', result)
                return result

            experiment.client = MethodType(client_factory, experiment)
            experiment.train = MethodType(train, experiment)
            experiment.score = MethodType(score, experiment)
            with patch.object(run, 'ROOT', root), patch.object(run, 'MASTER', master), redirect_stdout(StringIO()):
                experiment.base()
                experiment.local()
                if crash:
                    with self.assertRaisesRegex(RuntimeError, 'simulated'):
                        experiment.fedavg()
                records = experiment.fedavg()
                final_hash = run.tensor_hash(run.adapter(experiment.model))
                experiment.fedavg()  # Finished resume must restore final global, despite base evaluation.
                self.assertEqual(final_hash, run.tensor_hash(run.adapter(experiment.model)))
                experiment.lifecycle_audit()
                experiment.budget_audit()
                expected = sum(experiment.weights[d] * (i + 1) for i, d in enumerate(run.DOMAINS))
                self.assertAlmostEqual(experiment.model.lora_factor.item(), 2 * expected, places=6)
                self.assertEqual(len(attempts), 9)  # 3 independent epochs + 6 federated epochs.
                self.assertEqual(records[0]['common_start_tensor_sha256'], run.tensor_hash(experiment.initial))
                self.assertEqual(records[1]['common_start_tensor_sha256'], records[0]['global_after_tensor_sha256'])
                self.assertEqual(len(data.rows(output['fedavg'] / 'exposure.jsonl')), 6)
                self.assertEqual(len(data.read(output['fedavg'] / 'round_diagnostics.json')['rows']), 6)
                with (output['fedavg'] / 'round_diagnostics.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 6)

    def test_normal_lifecycle(self):
        self.exercise(None)

    def test_resume_saved_local_before_evaluation(self):
        self.exercise((1, 'client_openbookqa'))

    def test_resume_after_aggregation_before_evaluation(self):
        self.exercise((1, 'global'))


if __name__ == '__main__':
    unittest.main()
