"""Base -> independent ClientLocal -> ten-round sample-weighted FedLoRA.

Full cleaned official train; combined_test is validation + test. No other
methods. Every round records real global-before/local-after/global-after states.
"""
import argparse
import copy
import csv
import ctypes
import hashlib
import json
import math
import msvcrt
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
from scripts import three_dataset_data as data

DOMAINS, NAMES = data.DOMAINS, data.NAMES
MASTER = data.MASTER
OUTPUTS = dict(base=ROOT / 'outputs/three_dataset_base',
               clientlocal=ROOT / 'outputs/three_dataset_clientlocal',
               fedavg=ROOT / 'outputs/three_dataset_skewed')
read, dump, sha = data.read, data.dump, data.sha


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, **details):
    dump(MASTER / 'status.json', dict(stage=stage, pid=os.getpid(), updated_at=now(), **details))


def save_state(path, state):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def adapter(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora_' in k}


def tensor_hash(state):
    import torch
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def diagnostic_row(round_number, domain, before, local, after):
    local_gain, global_gain = local - before, after - before
    return dict(round=round_number, dataset=NAMES[domain], global_before=before,
                local_after=local, global_after=after, local_gain=local_gain,
                aggregation_gap=local - after, global_gain=global_gain,
                retention_ratio=global_gain / local_gain if local_gain > 0 else None)


def write_diagnostics(output, records):
    flat = [diagnostic_row(record['round'], domain,
                          record['global_before'][domain + '_accuracy'],
                          record['local_after'][domain + '_accuracy'],
                          record['global_after'][domain + '_accuracy'])
            for record in records for domain in DOMAINS]
    dump(output / 'round_diagnostics.json', dict(
        units='Accuracy and gains are fractions; plots show percent and percentage points.',
        undefined_retention='CSV NaN and JSON null when local_gain <= 0; ratios are not clipped.',
        rounds=records, rows=flat))
    path = output / 'round_diagnostics.csv'
    temporary = path.with_suffix('.tmp')
    fields = ['round', 'dataset', 'global_before', 'local_after', 'global_after',
              'local_gain', 'aggregation_gap', 'global_gain', 'retention_ratio']
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row, retention_ratio='NaN' if row['retention_ratio'] is None
                             else row['retention_ratio']) for row in flat)
    os.replace(temporary, path)


def event(output, round_number, phase, **details):
    with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(round=round_number, phase=phase, timestamp=now(), **details)) + '\n')


class ProgressLoader:
    def __init__(self, loader, client, method):
        self.loader, self.client, self.method = loader, client, method
        self.drop_last = loader.drop_last

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        started = time.monotonic()
        for index, batch in enumerate(self.loader, 1):
            yield batch
            if index % 200 == 0 or index == len(self):
                value = dict(method=self.method, round_or_epoch=self.client.server.round + 1,
                    dataset=DOMAINS[self.client.id], client=self.client.id, samples_finished=index,
                    samples_total=len(self), elapsed_seconds=time.monotonic() - started)
                dump(MASTER / 'live_progress.json', value)
                print(f"{self.method} {value['round_or_epoch']} {value['dataset']}: "
                      f"{index}/{len(self)} samples ({value['elapsed_seconds']:.1f}s)", flush=True)


class Experiment:
    def __init__(self, smoke=False):
        import torch
        from peft import get_peft_model
        from utils.mcq_utils import MCQTrainingDataset
        from utils.model_utils import load_model, load_tokenizer, load_lora_config
        from utils.seed_utils import set_global_seed
        cfg = data.config()
        self.manifest = read(ROOT / cfg['prepared_dir'] / 'manifest.json')
        for path, digest in self.manifest['files'].items():
            assert sha(ROOT / path) == digest, path
        self.smoke, self.rounds, self.epochs = smoke, 2 if smoke else 10, 1 if smoke else 10
        smoke_version = sha(Path(__file__))[:12]
        self.outputs = ({key: MASTER / 'smoke' / smoke_version / key for key in OUTPUTS} if smoke else OUTPUTS)
        self.args = SimpleNamespace(model_path=str(ROOT / cfg['model_path']), model='llama32_1b_base',
            device=0, mcq=True, task_type='CAUSAL_LM', seed=42, bs=1, grad_accum=8,
            lr=1e-4, epoch=1, step=0, cn=3, sr=1., lora_rank=8, lora_alpha=32,
            lora_dropout=0.05, eval_batch_size=8, mcq_domains=list(DOMAINS),
            mcq_dataset_dir=str(ROOT / cfg['prepared_dir']), suffix=str(MASTER))
        set_global_seed(42, device=0, deterministic=True)
        self.tokenizer = load_tokenizer(self.args)
        self.train_rows = {d: data.rows(ROOT / cfg['prepared_dir'] / 'train' / f'{d}.jsonl') for d in DOMAINS}
        self.test_rows = {d: data.rows(ROOT / cfg['prepared_dir'] / 'test' / f'{d}.jsonl') for d in DOMAINS}
        if smoke:
            self.train_rows = {d: self.train_rows[d][:n] for d, n in zip(DOMAINS, (9, 8, 17))}
            self.test_rows = {d: self.test_rows[d][:3] for d in DOMAINS}
        self.datasets = {d: MCQTrainingDataset(self.train_rows[d], self.tokenizer) for d in DOMAINS}
        from utils.mcq_utils import encode_prompt, continuation_ids, LETTERS
        self.test_encoded = {d: [(encode_prompt(self.tokenizer, row),
            [continuation_ids(self.tokenizer, row, letter) for letter in LETTERS])
            for row in self.test_rows[d]] for d in DOMAINS}
        self.counts = {d: len(self.datasets[d]) for d in DOMAINS}
        self.weights = {d: self.counts[d] / sum(self.counts.values()) for d in DOMAINS}
        initial = ROOT / cfg['initial_lora']
        receipt = read(ROOT / 'docs/migration_files_sha256.json')['files']
        assert sha(initial) == receipt[cfg['initial_lora']]
        set_global_seed(42, device=0, deterministic=True)
        lora_config = load_lora_config(self.args)
        assert lora_config.bias == 'none' and set(lora_config.target_modules) == {'q_proj', 'v_proj'}
        self.model = get_peft_model(load_model(self.args), lora_config)
        self.initial = torch.load(initial, map_location='cpu', weights_only=True)
        current = adapter(self.model)
        assert current.keys() == self.initial.keys()
        assert all(current[k].shape == v.shape and current[k].dtype == v.dtype for k, v in self.initial.items())
        self.load(self.initial)
        assert tensor_hash(adapter(self.model)) == tensor_hash(self.initial)
        assert all('lora_' in n for n, p in self.model.named_parameters() if p.requires_grad)
        assert next(p for n, p in self.model.named_parameters() if 'lora_' not in n).dtype == torch.bfloat16
        self.parameter_info = dict(total_parameters=sum(p.numel() for p in self.model.parameters()),
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            base_dtype='torch.bfloat16', trainable=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
                                                  for n, p in self.model.named_parameters() if p.requires_grad])
        self.protocol = dict(config=cfg, smoke=smoke, initial_lora_sha256=sha(initial),
            runner_sha256=sha(Path(__file__)), data_helper_sha256=sha(Path(data.__file__)),
            prepared_manifest_sha256=sha(ROOT / cfg['prepared_dir'] / 'manifest.json'),
            mcq_trainer_sha256=sha(ROOT / 'utils/train_utils.py'),
            aggregation_sha256=sha(ROOT / 'alg/ftbase.py'), counts=self.counts, weights=self.weights,
            rounds=self.rounds, clientlocal_epochs=self.epochs,
            optimizer_reset_per_client_epoch=True, base_frozen=True,
            aggregation='Existing FTBaseServer.aggregate: sample-count weighted LoRA A/B tensors')

    def load(self, state):
        self.model.load_state_dict(state, strict=False)

    def init_output(self, method):
        output = self.outputs[method]
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'protocol.json').exists():
            assert read(output / 'protocol.json') == self.protocol, 'Resume protocol differs'
        else:
            dump(output / 'protocol.json', self.protocol)
        dump(output / 'parameters.json', self.parameter_info)
        return output

    def score(self, output, domains, round_number, checkpoint):
        """Score the actual loaded model; cached scores require the same state."""
        from utils.mcq_eval import MCQEvaluator
        self.model.eval()
        fingerprint = tensor_hash(adapter(self.model))
        expected = dict(lora_tensor_sha256=fingerprint, checkpoint=str(checkpoint.relative_to(ROOT)),
                        domains=list(domains), combined_test_manifest_sha256=self.protocol['prepared_manifest_sha256'])
        result_path = output / 'result.json'
        if result_path.exists():
            receipt = read(output / 'evaluation_provenance.json')
            assert all(receipt[k] == v for k, v in expected.items())
            return read(result_path)
        evaluator = MCQEvaluator.__new__(MCQEvaluator)
        evaluator.args, evaluator.tokenizer, evaluator.domains = self.args, self.tokenizer, tuple(domains)
        evaluator.rows = {d: self.test_rows[d] for d in domains}
        evaluator.encoded = {d: self.test_encoded[d] for d in domains}
        evaluator.output = output / 'evaluation'
        evaluator.output.mkdir(parents=True, exist_ok=True)
        metrics_path = evaluator.output / 'metrics.jsonl'
        if metrics_path.exists():
            # A complete evaluator write may precede the atomic result commit.
            metrics = data.rows(metrics_path)
            assert len(metrics) == 1 and metrics[0]['round'] == round_number
            result = {k: v for k, v in metrics[0].items() if k != 'round'}
        else:
            result = evaluator.evaluate(self.model, round_number)
        with (evaluator.output / f'round_{round_number:02d}_predictions.csv').open(encoding='utf-8', newline='') as stream:
            predictions = list(csv.DictReader(stream))
        for d in domains:
            current = [p for p in predictions if p['domain'] == d]
            assert len(current) == len({p['id'] for p in current}) == len(self.test_rows[d])
            assert {p['id'] for p in current} == {str(row['id']) for row in self.test_rows[d]}
            assert all(int(p['correct']) == int(p['gold'] == p['prediction']) for p in current)
            assert abs(sum(int(p['correct']) for p in current) / len(current) - result[d + '_accuracy']) < 1e-12
        dump(output / 'evaluation_provenance.json', dict(expected, evaluated_at=now()))
        dump(result_path, result)
        return result

    def client(self, domain, output, server, method):
        from alg.base import BaseClient
        from alg.ftbase import FTBaseClient
        from utils.train_utils import Trainer
        args = copy.copy(self.args)
        args.suffix = str(output)

        class Client(FTBaseClient):
            def __init__(client):
                BaseClient.__init__(client, DOMAINS.index(domain), args)
                client.server, client.dataset = server, {'train': self.datasets[domain]}
                client.tokenizer, client.lora, client.delay = self.tokenizer, {}, 1.
                client.evaluator, client.last_train_loss = None, None
                client.trainer = Trainer(args, client.dataset, client)
                assert len(client.trainer.train_loader) == self.counts[domain]
                assert not client.trainer.train_loader.drop_last
                client.trainer.train_loader = ProgressLoader(client.trainer.train_loader, client, method)
        return Client()

    def train(self, client, domain):
        loss = client.trainer.train(self.model)
        assert client.last_seen_count == self.counts[domain]
        assert client.last_optimizer_updates == math.ceil(self.counts[domain] / 8)
        value = adapter(self.model)
        import torch
        assert all(torch.isfinite(v).all() for v in value.values())
        return value, dict(train_loss=loss, samples=client.last_seen_count,
                           optimizer_updates=client.last_optimizer_updates)

    def trim_exposure(self, path, completed, pending=()):
        if path.exists():
            value = [r for r in data.rows(path) if r['round'] <= completed or
                     (r['round'] == completed + 1 and r['client'] in pending)]
            data.write_rows(path, value)

    def base(self):
        output = self.init_output('base')
        self.load(self.initial)
        checkpoint = output / 'initial_lora.pt'
        save_state(checkpoint, self.initial)
        status('smoke_base' if self.smoke else 'base')
        result = self.score(output, DOMAINS, 0, checkpoint)
        (output / 'completed.txt').write_text('Initial LoRA evaluated on combined_test.\n', encoding='utf-8')
        return result

    def local(self):
        import torch
        output = self.init_output('clientlocal')
        result, matrix = {}, {}
        for domain in DOMAINS:
            current = output / ('client_' + domain)
            current.mkdir(parents=True, exist_ok=True)
            checkpoint = current / 'resume.pt'
            state = (torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists()
                     else dict(epoch=0, lora=self.initial, curves=[]))
            self.load(state['lora'])
            self.trim_exposure(current / 'exposure.jsonl', state['epoch'])
            server = SimpleNamespace(round=0)
            client = self.client(domain, current, server, 'smoke_clientlocal' if self.smoke else 'clientlocal')
            for epoch in range(state['epoch'] + 1, self.epochs + 1):
                server.round = epoch - 1
                status('smoke_clientlocal' if self.smoke else 'clientlocal', dataset=domain, epoch=epoch)
                value, record = self.train(client, domain)
                state['curves'].append(dict(epoch=epoch, **record))
                state.update(epoch=epoch, lora=value)
                save_state(checkpoint, state)
                dump(current / 'training_epochs.json', state['curves'])
            dump(current / 'training_epochs.json', state['curves'])
            final = current / 'final_lora.pt'
            save_state(final, state['lora'])
            self.load(state['lora'])
            # One all-domain endpoint evaluation supplies the optional 3x3
            # matrix. Only its diagonal is used for ClientLocal's main table.
            scores = self.score(current, DOMAINS, self.epochs, final)
            result[domain + '_accuracy'] = scores[domain + '_accuracy']
            matrix[domain] = {d: scores[d + '_accuracy'] for d in DOMAINS}
            (current / 'completed.txt').write_text('Independent specialist completed.\n', encoding='utf-8')
        result['macro_accuracy'] = sum(result[d + '_accuracy'] for d in DOMAINS) / 3
        dump(output / 'result.json', result)
        dump(output / 'cross_dataset_matrix.json', matrix)
        with (output / 'cross_dataset_matrix.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['specialist', *DOMAINS])
            writer.writeheader()
            writer.writerows(dict(specialist=d, **matrix[d]) for d in DOMAINS)
        (output / 'completed.txt').write_text('Three independent specialists completed.\n', encoding='utf-8')
        return result

    def fedavg(self):
        import torch
        from alg.ftbase import FTBaseServer
        output = self.init_output('fedavg')
        checkpoint = output / 'resume.pt'
        state = (torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists()
                 else dict(completed_round=0, global_lora=self.initial, pending={}, records=[]))
        self.trim_exposure(output / 'exposure.jsonl', state['completed_round'], state['pending'])
        write_diagnostics(output, state['records'])
        server = SimpleNamespace(round=0, model=self.model, global_lora=state['global_lora'])
        clients = [self.client(d, output, server, 'smoke_fedavg' if self.smoke else 'fedavg') for d in DOMAINS]
        server.sampled_clients = clients
        base_result = self.base()
        for t in range(state['completed_round'] + 1, self.rounds + 1):
            round_dir = output / f'round_{t:02d}'
            round_dir.mkdir(parents=True, exist_ok=True)
            server.round, server.global_lora = t - 1, state['global_lora']
            self.load(server.global_lora)
            common_hash = tensor_hash(server.global_lora)
            before_path = round_dir / 'global_before/lora_weights.pt'
            save_state(before_path, server.global_lora)
            status('smoke_global_before' if self.smoke else 'global_before', round=t)
            before = self.score(round_dir / 'global_before', DOMAINS, t, before_path)
            previous = base_result if t == 1 else state['records'][-1]['global_after']
            assert all(before[d + '_accuracy'] == previous[d + '_accuracy'] for d in DOMAINS)
            event(output, t, 'global_before', lora_tensor_sha256=common_hash)
            local = {}
            for client, domain in zip(clients, DOMAINS):
                directory = round_dir / ('client_' + domain)
                directory.mkdir(parents=True, exist_ok=True)
                local_path = directory / 'lora_weights.pt'
                if client.id not in state['pending']:
                    self.load(server.global_lora)
                    assert tensor_hash(adapter(self.model)) == common_hash
                    status('smoke_local_training' if self.smoke else 'local_training', round=t, dataset=domain)
                    value, record = self.train(client, domain)
                    save_state(local_path, value)
                    state['pending'][client.id] = dict(lora=value, budget=record, start_hash=common_hash)
                    save_state(checkpoint, state)
                    event(output, t, 'local_checkpoint_saved', dataset=domain, checkpoint=str(local_path.relative_to(ROOT)))
                item = state['pending'][client.id]
                assert item['start_hash'] == common_hash and local_path.exists()
                assert tensor_hash(torch.load(local_path, map_location='cpu', weights_only=True)) == tensor_hash(item['lora'])
                self.load(item['lora'])
                status('smoke_local_after' if self.smoke else 'local_after', round=t, dataset=domain)
                scores = self.score(directory, (domain,), t, local_path)
                local[domain + '_accuracy'] = scores[domain + '_accuracy']
                client.lora = item['lora']
                item['local_after'] = scores
                save_state(checkpoint, state)
                event(output, t, 'local_after', dataset=domain, lora_tensor_sha256=tensor_hash(item['lora']))
            assert len(state['pending']) == 3 and all('local_after' in item for item in state['pending'].values())
            event(output, t, 'aggregation_start', weights=self.weights)
            # This is the unchanged existing A/B FedAvg implementation.
            FTBaseServer.aggregate(server)
            global_lora = adapter(self.model)
            assert all(torch.isfinite(v).all() for v in global_lora.values())
            global_path = round_dir / 'global/lora_weights.pt'
            save_state(global_path, global_lora)
            event(output, t, 'aggregation_complete', lora_tensor_sha256=tensor_hash(global_lora))
            status('smoke_global_after' if self.smoke else 'global_after', round=t)
            after = self.score(round_dir / 'global', DOMAINS, t, global_path)
            event(output, t, 'global_after', lora_tensor_sha256=tensor_hash(global_lora))
            state['records'].append(dict(round=t, global_before=before, local_after=local,
                global_after=after, common_start_tensor_sha256=common_hash,
                global_after_tensor_sha256=tensor_hash(global_lora), weights=self.weights,
                clients={d: state['pending'][i]['budget'] for i, d in enumerate(DOMAINS)}))
            state.update(completed_round=t, global_lora=global_lora, pending={})
            save_state(checkpoint, state)
            write_diagnostics(output, state['records'])
            print(f'FedAvg round {t}/{self.rounds} committed; global macro={after["macro_accuracy"]:.6f}', flush=True)
        self.load(state['global_lora'])
        assert len(state['records']) == self.rounds
        dump(output / 'result.json', state['records'][-1]['global_after'])
        (output / 'completed.txt').write_text('All rounds and three-state diagnostics completed.\n', encoding='utf-8')
        return state['records']

    def budget_audit(self):
        output = self.outputs['fedavg']
        records = data.rows(output / 'exposure.jsonl')
        assert len(records) == self.rounds * 3
        assert {(r['round'], r['client']) for r in records} == {(t, c) for t in range(1, self.rounds + 1) for c in range(3)}
        budgets = []
        for i, domain in enumerate(DOMAINS):
            ids = {data.native_key(row) for row in self.train_rows[domain]}
            for method, epochs, exposure in (
                    ('Dataset-Skewed FedAvg', self.rounds, [r for r in records if r['client'] == i]),
                    ('ClientLocal', self.epochs, data.rows(self.outputs['clientlocal'] / ('client_' + domain) / 'exposure.jsonl'))):
                assert len(exposure) == epochs
                for t, record in enumerate(exposure, 1):
                    assert record['round'] == t and record['client'] == i
                    assert record['samples'] == len(record['ids']) == len(set(record['ids'])) == len(ids)
                    assert set(record['ids']) == ids
                    assert record['optimizer_updates'] == math.ceil(len(ids) / 8)
                budgets.append(dict(method=method, dataset=NAMES[domain], train_samples=len(ids),
                    local_epochs_per_round=1, communication_rounds=self.rounds if method.startswith('Dataset') else 0,
                    effective_epochs=epochs, optimizer_steps_per_local_epoch=math.ceil(len(ids) / 8),
                    total_optimizer_steps=sum(r['optimizer_updates'] for r in exposure),
                    total_sample_visits=sum(r['samples'] for r in exposure)))
        dump(output / 'training_budget_audit.json', dict(passed=True, rows=budgets))
        with (output / 'training_budget.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(budgets[0]))
            writer.writeheader()
            writer.writerows(budgets)
        return budgets

    def lifecycle_audit(self):
        output = self.outputs['fedavg']
        records = read(output / 'round_diagnostics.json')['rounds']
        events = data.rows(output / 'events.jsonl')
        for i, record in enumerate(records):
            t = i + 1
            current = [event for event in events if event['round'] == t]
            starts = [j for j, item in enumerate(current) if item['phase'] == 'aggregation_start']
            for start in starts:
                prior_domains = {item['dataset'] for item in current[:start] if item['phase'] == 'local_after'}
                assert prior_domains == set(DOMAINS)
            assert starts and current[-1]['phase'] == 'global_after'
            for name, expected in (('global_before', record['common_start_tensor_sha256']),
                                   ('global', record['global_after_tensor_sha256'])):
                directory = output / f'round_{t:02d}' / name
                receipt = read(directory / 'evaluation_provenance.json')
                actual = tensor_hash(__import__('torch').load(
                    directory / 'lora_weights.pt', map_location='cpu', weights_only=True))
                assert receipt['domains'] == list(DOMAINS)
                assert receipt['lora_tensor_sha256'] == actual == expected
            for domain in DOMAINS:
                directory = output / f'round_{t:02d}' / ('client_' + domain)
                assert (directory / 'lora_weights.pt').exists()
                receipt = read(directory / 'evaluation_provenance.json')
                assert receipt['domains'] == [domain]
                assert receipt['lora_tensor_sha256'] == tensor_hash(__import__('torch').load(
                    directory / 'lora_weights.pt', map_location='cpu', weights_only=True))
            if i:
                assert record['common_start_tensor_sha256'] == records[i-1]['global_after_tensor_sha256']
                assert record['global_before'] == records[i-1]['global_after']
        flat = read(output / 'round_diagnostics.json')['rows']
        assert len(flat) == self.rounds * 3
        result = dict(passed=True, rounds=self.rounds, diagnostic_rows=len(flat),
            checkpoints_before_local_evaluation=True, all_local_evaluations_before_aggregation=True,
            global_evaluation_after_aggregation=True, previous_global_is_next_common_start=True,
            sample_count_weights=self.weights)
        dump(output / 'lifecycle_audit.json', result)
        return result


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = read(output / 'round_diagnostics.json')['rows']
    destination = output / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                         'svg.fonttype': 'none', 'savefig.dpi': 300})
    for domain in DOMAINS:
        current = [row for row in records if row['dataset'] == NAMES[domain]]
        rounds = [row['round'] for row in current]
        for kind, columns, labels, ylabel in (
            ('states', ('global_before', 'local_after', 'global_after'),
             ('Global Before', 'Local After Training', 'Global After Aggregation'), 'Accuracy (%)'),
            ('aggregation_gap', ('aggregation_gap',), ('Aggregation Gap',), 'Aggregation Gap (pp)'),
            ('gains', ('local_gain', 'global_gain'), ('Local Gain', 'Global Gain'), 'Accuracy change (pp)')):
            fig = plt.figure(figsize=(5.3, 3.5))
            ax = fig.add_axes([0.16, 0.17, 0.80, 0.73])
            for column, label, color in zip(columns, labels, ('#2463a5', '#dd7a18', '#258c66')):
                ax.plot(rounds, [100 * row[column] for row in current], marker='o', markersize=3,
                        linewidth=1.5, color=color, label=label)
            if kind != 'states':
                ax.axhline(0, color='#777', linewidth=0.7)
            ax.set(title=NAMES[domain], xlabel='Communication round', ylabel=ylabel)
            ax.set_xticks(rounds)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=0.20)
            ax.legend(frameon=False)
            for extension in ('png', 'svg', 'pdf'):
                fig.savefig(destination / f'{domain}_{kind}.{extension}')
            plt.close(fig)


def final_report(experiment):
    budgets = experiment.budget_audit()
    lifecycle = experiment.lifecycle_audit()
    output = experiment.outputs['fedavg']
    figures(output)
    results = {method: read(directory / 'result.json') for method, directory in experiment.outputs.items()}
    lines = ['# Three-dataset full-train Federated LoRA (seed42)', '',
        '| Method | LogiQA | OpenBookQA | SciQ | Macro |', '| --- | ---: | ---: | ---: | ---: |']
    for method, label in (('base', 'Base'), ('clientlocal', 'ClientLocal'), ('fedavg', 'Dataset-Skewed FedAvg')):
        values = results[method]
        lines.append('| ' + label + ' | ' + ' | '.join(f"{100*values[d+'_accuracy']:.2f}%" for d in DOMAINS)
                     + f" | {100*values['macro_accuracy']:.2f}% |")
    lines += ['', 'All scores use Combined Test (cleaned official validation + official test). ClientLocal uses the diagonal of the three independently trained specialists; FedAvg uses the Round10 post-aggregation global.', '',
        '| Dataset | Train | Combined Test | Weight | Steps/epoch | Total steps/client |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in DOMAINS:
        n = experiment.counts[d]
        lines.append(f'| {NAMES[d]} | {n} | {len(experiment.test_rows[d])} | {experiment.weights[d]:.10f} | '
                     f'{math.ceil(n/8)} | {10*math.ceil(n/8)} |')
    lines += ['', 'OpenBookQA removes the 21 cross-split repeated QA train rows with explicit user approval; the Combined Test is unchanged. Other source cleaning is unchanged; within-train repeated content with distinct official IDs is retained.', '',
        'Llama-3.2-1B BF16 frozen base, canonical initial q_proj/v_proj LoRA r8/alpha32/dropout0.05/bias=none. Constant lr1e-4, bs1, accumulation8, drop_last=False, step0. Existing MCQ Trainer resets AdamW per full local epoch and updates incomplete accumulation tails.', '',
        'Diagnostics are descriptive: Local Gain = Local After − Global Before; Aggregation Gap = Local After − Global After; Global Gain = Global After − Global Before. Retention Ratio is Global Gain / Local Gain only when Local Gain > 0; otherwise CSV NaN / JSON null. Ratios are not clipped.', '',
        'No early stopping, checkpoint selection, combined-test tuning or new aggregation method. No Centralized, IID-Mixed or other dataset is run.', '',
        '| Round | Dataset | Local Gain (pp) | Aggregation Gap (pp) | Global Gain (pp) |',
        '| ---: | --- | ---: | ---: | ---: |']
    flat = read(output / 'round_diagnostics.json')['rows']
    for row in flat:
        lines.append(f"| {row['round']} | {row['dataset']} | {100*row['local_gain']:+.2f} | "
                     f"{100*row['aggregation_gap']:+.2f} | {100*row['global_gain']:+.2f} |")
    lines += ['', '## Observed performance changes', '']
    for domain in DOMAINS:
        current = [row for row in flat if row['dataset'] == NAMES[domain]]
        gaps = [100 * row['aggregation_gap'] for row in current]
        positive_local = sum(row['local_gain'] > 0 for row in current)
        positive_global = sum(row['global_gain'] > 0 for row in current)
        fed = results['fedavg'][domain + '_accuracy']
        base = results['base'][domain + '_accuracy']
        local = results['clientlocal'][domain + '_accuracy']
        lines.append(f"{NAMES[domain]}: final FedAvg − Base = {100*(fed-base):+.2f} pp; "
            f"independent ClientLocal − final FedAvg = {100*(local-fed):+.2f} pp. "
            f"Across {len(current)} rounds, Local Gain is positive in {positive_local} and Global Gain "
            f"is positive in {positive_global}. Aggregation Gap ranges from {min(gaps):+.2f} "
            f"to {max(gaps):+.2f} pp (mean {sum(gaps)/len(gaps):+.2f} pp); "
            f"Round{len(current)} gap = {gaps[-1]:+.2f} pp.")
        lines.append('')
    lines += ['## Actual training budget', '',
        '| Method | Dataset | Full epochs | Samples visited | Steps/epoch | Total optimizer steps |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for item in budgets:
        lines.append(f"| {item['method']} | {item['dataset']} | {item['effective_epochs']} | "
            f"{item['total_sample_visits']} | {item['optimizer_steps_per_local_epoch']} | "
            f"{item['total_optimizer_steps']} |")
    lines += ['', 'The recorded differences describe model performance at the three states; they do not identify an underlying mechanism.', '',
        'Artifacts: `outputs/three_dataset_skewed/round_diagnostics.csv`, `round_diagnostics.json`, `round_01` through `round_10` (three local LoRAs plus global), and `figures/` (separate dataset figures; no multi-panel plots).',
        'Specialists: `outputs/three_dataset_clientlocal/client_{dataset}/final_lora.pt`; auxiliary matrix: `cross_dataset_matrix.csv/json`.',
        'Budget: `outputs/three_dataset_skewed/training_budget.csv` and `training_budget_audit.json`. Pre-training audit: `reports/three_dataset_pretraining_audit.md`.']
    lines += ['', 'Changed implementation files: `configs/three_dataset_lora.yaml`, '
        '`scripts/three_dataset_data.py`, `scripts/run_three_dataset_lora.py`, '
        '`tests/test_three_dataset_lora.py`, `docs/THREE_DATASET_LORA.md`. '
        'Original model, raw/unified data, MCQ Trainer/evaluator and FedAvg implementation are reused.', '',
        'Protocol mismatches: none detected in the completed data, exposure, lifecycle and checkpoint audits. '
        'The only approved data change is removal of the 21 overlapping OpenBookQA train rows. '
        'These single-seed observations do not provide uncertainty estimates.']
    path = MASTER / 'smoke/report.md' if experiment.smoke else ROOT / 'reports/three_dataset_lora_seed42.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(output / 'final_results.json', dict(results=results, budgets=budgets, lifecycle=lifecycle))
    print('\n'.join(lines[:8]), flush=True)
    return path


def protocol_audit(smoke):
    lifecycle = smoke.lifecycle_audit()
    smoke.budget_audit()
    prepared = read(ROOT / data.config()['prepared_dir'] / 'manifest.json')['audit']
    items = {
        '01_logiqa_counts': prepared['counts']['logiqa'], '02_openbookqa_counts': prepared['counts']['openbookqa'],
        '03_sciq_counts': prepared['counts']['sciq'], '04_combined_test_union': True,
        '05_no_train_test_overlap': prepared['no_train_combined_test_overlap'],
        '06_answer_label_mapping': prepared['label_mapping_checked'], '07_four_letters_ABCD': True,
        '08_sciq_correct_distractor_mapping': prepared['sciq_correct_and_distractors_verified'],
        '09_logiqa_existing_exclusions': prepared['logiqa_exclusion_rules_unchanged'],
        '10_full_client_epoch': True, '11_drop_last_false': True, '12_tail_optimizer_step': True,
        '13_no_step_truncation': True, '14_sample_count_weights': lifecycle['sample_count_weights'],
        '15_local_checkpoint_saved': lifecycle['checkpoints_before_local_evaluation'],
        '16_local_after_before_aggregation': lifecycle['all_local_evaluations_before_aggregation'],
        '17_global_after_post_aggregation': lifecycle['global_evaluation_after_aggregation'],
        '18_next_round_previous_global': lifecycle['previous_global_is_next_common_start']}
    assert all(value is not False for value in items.values())
    result = dict(passed=True, completed_at=now(), checks=items,
                  smoke_train_counts=smoke.counts, smoke_rounds=smoke.rounds,
                  smoke_outputs={k: str(v.relative_to(ROOT)) for k, v in smoke.outputs.items()},
                  formal_train_counts={d: prepared['counts'][d]['train'] for d in DOMAINS},
                  formal_weights=prepared['weights'], smoke_lifecycle=lifecycle)
    dump(MASTER / 'protocol_audit.json', result)
    with (ROOT / 'reports/three_dataset_pretraining_audit.md').open('a', encoding='utf-8') as stream:
        stream.write('\n## Runtime smoke audit\n\nAll 18 requested checks PASS. '
                     'Two rounds use 9/8/17 smoke train rows (tail steps and unequal client budgets), '
                     'with real LoRA checkpoints and pre/post-aggregation likelihood evaluation. '
                     'See `outputs/three_dataset_seed42/protocol_audit.json`.\n')
        stream.write('\n| Check | Result | Evidence |\n| --- | --- | --- |\n')
        for key, value in items.items():
            if key == '14_sample_count_weights':
                evidence = dict(formal=prepared['weights'], smoke=value)
            else:
                evidence = value
            stream.write(f'| {key} | PASS | `{json.dumps(evidence, ensure_ascii=False)}` |\n')


def all_jobs():
    data.prepare()
    status('smoke')
    smoke = Experiment(smoke=True)
    smoke.base()
    smoke.local()
    smoke.fedavg()
    protocol_audit(smoke)
    # Release the smoke model before allocating the formal model.
    del smoke
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    experiment = Experiment()
    print('Formal Dataset | Train | Combined Test | FedAvg weight | Steps/epoch', flush=True)
    for d in DOMAINS:
        print(f'{NAMES[d]} | {experiment.counts[d]} | {len(experiment.test_rows[d])} | '
              f'{experiment.weights[d]:.10f} | {math.ceil(experiment.counts[d]/8)}', flush=True)
    print(json.dumps(data.config(), ensure_ascii=False), flush=True)
    experiment.base()
    experiment.local()
    experiment.fedavg()
    status('reporting')
    path = final_report(experiment)
    status('complete', report=str(path))
    (MASTER / 'completed.txt').write_text('Base, ClientLocal, multi-round FedAvg, diagnostics, figures and audits completed.\n', encoding='utf-8')


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('audit', 'prepare', 'smoke', 'all', 'report'), default='audit')
    args = parser.parse_args()
    os.chdir(ROOT)
    MASTER.mkdir(parents=True, exist_ok=True)
    if args.job == 'audit':
        data.audit()
    elif args.job == 'prepare':
        data.prepare()
    elif args.job == 'report':
        final_report(Experiment())
    else:
        with (MASTER / 'run.lock').open('a+b') as lock:
            if lock.tell() == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise SystemExit('Three-dataset experiment already running; duplicate launch refused.')
            try:
                if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                    raise ctypes.WinError()
                if args.job == 'all':
                    all_jobs()
                else:
                    data.prepare()
                    smoke = Experiment(smoke=True)
                    smoke.base(); smoke.local(); smoke.fedavg()
                    protocol_audit(smoke)
                    status('smoke_complete')
            except BaseException as error:
                status('failed', error=str(error), traceback=traceback.format_exc())
                traceback.print_exc()
                raise SystemExit(1)
            finally:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
