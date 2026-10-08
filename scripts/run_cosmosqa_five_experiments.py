"""Reuse all Base/Local endpoints, then fresh five-client FedAvg and Centralized-All."""
import argparse
import copy
import ctypes
import hashlib
import json
import math
import msvcrt
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cosmosqa_five_data as data
from scripts import run_three_dataset_lora as shared
from utils import cosmosqa_five_mcq as encoding

DOMAINS, NAMES, MASTER = data.DOMAINS, data.NAMES, data.MASTER
OUTPUTS = {m: MASTER / m for m in ('base', 'clientlocal', 'fedavg', 'centralized')}
REPORT = ROOT / 'reports/cosmosqa_five_comparison_seed42.md'


def configure():
    shared.data, shared.MASTER, shared.OUTPUTS = data, MASTER, OUTPUTS
    shared.DOMAINS, shared.NAMES = DOMAINS, NAMES


class MixedDataset:
    def __init__(self, datasets):
        self.rows = [row for d in DOMAINS for row in datasets[d].rows]
        self.encoded = [row for d in DOMAINS for row in datasets[d].encoded]
    def __len__(self): return len(self.encoded)
    def __getitem__(self, index): return self.encoded[index]


class CentralProgress:
    def __init__(self, loader, client):
        self.loader, self.client, self.drop_last = loader, client, loader.drop_last
    def __len__(self): return len(self.loader)
    def __iter__(self):
        started = time.monotonic()
        for count, batch in enumerate(self.loader, 1):
            yield batch
            if count % 200 == 0 or count == len(self):
                value = dict(method='centralized', round_or_epoch=self.client.server.round + 1,
                    dataset='all_five', samples_finished=count, samples_total=len(self),
                    optimizer_updates_finished=count // 8, elapsed_seconds=time.monotonic() - started)
                data.dump(MASTER / 'live_progress.json', value)
                print('CENTRAL', json.dumps(value), flush=True)


class Experiment(shared.Experiment):
    def __init__(self, smoke=False):
        configure()
        from utils import mcq_utils
        with patch.object(mcq_utils, 'MCQTrainingDataset', encoding.TrainingDataset), \
             patch.object(mcq_utils, 'encode_prompt', encoding.encode_prompt), \
             patch.object(mcq_utils, 'continuation_ids', encoding.continuation_ids):
            super().__init__(smoke=False)
        self.smoke, self.rounds, self.epochs = smoke, 2 if smoke else 10, 2 if smoke else 10
        self.args.cn = 5
        if smoke:
            self.train_rows = {d: self.train_rows[d][:n] for d, n in zip(DOMAINS, (9, 8, 17, 9, 8))}
            self.test_rows = {d: self.test_rows[d][:3] for d in DOMAINS}
            self.datasets = {d: encoding.TrainingDataset(self.train_rows[d], self.tokenizer) for d in DOMAINS}
            self.test_encoded = {d: [(encoding.encode_prompt(self.tokenizer, row),
                [encoding.continuation_ids(self.tokenizer, row, l) for l in 'ABCD']) for row in self.test_rows[d]] for d in DOMAINS}
        self.counts = {d: len(self.datasets[d]) for d in DOMAINS}
        self.weights = {d: self.counts[d] / sum(self.counts.values()) for d in DOMAINS}
        self.supervised_counts = {d: sum(sum(label != -100 for label in item['labels'])
            for item in self.datasets[d].encoded) for d in DOMAINS}
        statistics = data.read(ROOT / data.config()['prepared_dir'] / 'sequence_statistics.json')
        for d in DOMAINS:
            if not smoke:
                assert self.counts[d] == 4000 and self.weights[d] == .2
                assert sum(len(x['input_ids']) for x in self.datasets[d].encoded) == statistics[d]['train']['token_total']
                assert self.supervised_counts[d] == statistics[d]['train']['supervised_token_total']
        self.protocol.update(smoke=smoke, rounds=self.rounds, clientlocal_epochs=10,
            centralized_epochs=self.epochs, client_count=5, counts=self.counts, weights=self.weights,
            suite_runner_sha256=data.sha(Path(__file__)), encoder_sha256=data.sha(Path(encoding.__file__)),
            choice_evaluator_sha256=data.sha(ROOT / 'utils/mcq_eval.py'),
            passage_encoder_sha256=data.sha(ROOT / 'utils/race_mcq.py'),
            cosmos_encoder_sha256=data.sha(ROOT / 'utils/cosmosqa_mcq.py'),
            previous_encoder_sha256=data.sha(ROOT / 'utils/five_client_mcq.py'),
            seed_utils_sha256=data.sha(ROOT / 'utils/seed_utils.py'),
            model_utils_sha256=data.sha(ROOT / 'utils/model_utils.py'),
            canonical_initial_tensor_sha256=shared.tensor_hash(self.initial),
            reuse_all_base_local=True, own_domain_local_evaluation_only=True,
            cross_domain_matrix=False, checkpoint_selection='fixed Round10 / Epoch10',
            fresh_methods=['fedavg', 'centralized'], centralized_samples=sum(self.counts.values()),
            centralized_optimizer_reset='Original Trainer AdamW once per pooled full epoch',
            client_seed_rule='client_round_seed(42, client_id, round_or_epoch-1)')
        if smoke:
            version = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()[:12]
            self.outputs = {m: MASTER / 'smoke' / version / m for m in OUTPUTS}
        else:
            self.outputs = dict(OUTPUTS)
        encoding.clear_cache()

    def base(self):
        if self.smoke: return super().base()
        import torch
        output = self.init_output('base')
        result, references = {}, {}
        initial_hash = shared.tensor_hash(self.initial)
        for d in DOMAINS:
            source, _ = data.baseline_sources(d)
            scores = data.read(source / 'result.json')
            receipt = data.read(source / 'evaluation_provenance.json')
            assert d in receipt['domains'] and receipt['lora_tensor_sha256'] == initial_hash
            source_checkpoint = ROOT / receipt['checkpoint']
            assert shared.tensor_hash(torch.load(source_checkpoint, map_location='cpu', weights_only=True)) == initial_hash
            result[d + '_accuracy'] = scores[d + '_accuracy']
            references[d] = dict(result=str((source / 'result.json').relative_to(ROOT)),
                result_sha256=data.sha(source / 'result.json'),
                receipt=str((source / 'evaluation_provenance.json').relative_to(ROOT)),
                receipt_sha256=data.sha(source / 'evaluation_provenance.json'),
                original_evaluation_receipt=receipt, accuracy=result[d + '_accuracy'])
        result.update(macro_accuracy=sum(result.values()) / 5, worst_domain_accuracy=min(result.values()))
        data.dump(output / 'reused_results.json', references)
        data.dump(output / 'result.json', result)
        (output / 'completed.txt').write_text('All five canonical Base results reused; no baseline reevaluation.\n', encoding='utf-8')
        self.load(self.initial)
        return result

    def local(self):
        if self.smoke: raise RuntimeError('No Independent Local smoke/retraining in this suite')
        import torch
        output = self.init_output('clientlocal')
        result, references = {}, {}
        for d in DOMAINS:
            _, source = data.baseline_sources(d)
            checkpoint = source / 'final_lora.pt'
            receipt = data.read(source / 'evaluation_provenance.json')
            assert d in receipt['domains']
            assert shared.tensor_hash(torch.load(checkpoint, map_location='cpu', weights_only=True)) == receipt['lora_tensor_sha256']
            curves = data.read(source / 'training_epochs.json')
            assert [r['epoch'] for r in curves] == list(range(1, 11))
            assert all(r['samples'] == 4000 and r['optimizer_updates'] == 500 for r in curves)
            assert data.read(source.parent / 'protocol.json')['initial_lora_sha256'] == self.protocol['initial_lora_sha256']
            exposure = data.rows(source / 'exposure.jsonl')
            expected_ids = {data.native_key(row) for row in self.train_rows[d]}
            assert len(exposure) == 10
            for epoch, row in enumerate(exposure, 1):
                assert row['round'] == epoch and row['client'] == DOMAINS.index(d)
                assert row['samples'] == len(row['ids']) == len(set(row['ids'])) == 4000
                assert set(row['ids']) == expected_ids and row['optimizer_updates'] == 500
                assert row['supervised_tokens'] == self.supervised_counts[d]
            result[d + '_accuracy'] = data.read(source / 'result.json')[d + '_accuracy']
            references[d] = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=data.sha(checkpoint),
                receipt=str((source / 'evaluation_provenance.json').relative_to(ROOT)),
                receipt_sha256=data.sha(source / 'evaluation_provenance.json'),
                result=str((source / 'result.json').relative_to(ROOT)), result_sha256=data.sha(source / 'result.json'),
                accuracy=result[d + '_accuracy'], formal_epoch=10, reused=True)
        result['macro_accuracy'] = sum(result.values()) / 5
        data.dump(output / 'reused_specialists.json', references)
        data.dump(output / 'result.json', result)
        (output / 'completed.txt').write_text('All five epoch10 specialists reused; no Independent Local training.\n', encoding='utf-8')
        return result

    def fedavg(self):
        # Same shared three-state loop; generalized only from 3 to 5 clients.
        import torch
        from alg.ftbase import FTBaseServer
        output = self.init_output('fedavg'); checkpoint = output / 'resume.pt'
        state = (torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists()
                 else dict(completed_round=0, global_lora=self.initial, pending={}, records=[]))
        self.trim_exposure(output / 'exposure.jsonl', state['completed_round'], state['pending'])
        shared.write_diagnostics(output, state['records'])
        server = SimpleNamespace(round=0, model=self.model, global_lora=state['global_lora'])
        clients = [self.client(d, output, server, 'smoke_fedavg' if self.smoke else 'fedavg') for d in DOMAINS]
        server.sampled_clients = clients
        base_result = self.base()
        for t in range(state['completed_round'] + 1, self.rounds + 1):
            directory = output / f'round_{t:02d}'; directory.mkdir(parents=True, exist_ok=True)
            server.round, server.global_lora = t - 1, state['global_lora']
            self.load(server.global_lora); common_hash = shared.tensor_hash(server.global_lora)
            if t == 1: assert common_hash == shared.tensor_hash(self.initial)
            before_path = directory / 'global_before/lora_weights.pt'; shared.save_state(before_path, server.global_lora)
            shared.status('smoke_global_before' if self.smoke else 'global_before', round=t)
            before = self.score(directory / 'global_before', DOMAINS, t, before_path)
            previous = base_result if t == 1 else state['records'][-1]['global_after']
            assert all(before[d + '_accuracy'] == previous[d + '_accuracy'] for d in DOMAINS)
            shared.event(output, t, 'global_before', lora_tensor_sha256=common_hash)
            local = {}
            for client, d in zip(clients, DOMAINS):
                current = directory / ('client_' + d); current.mkdir(parents=True, exist_ok=True)
                local_path = current / 'lora_weights.pt'
                if client.id not in state['pending']:
                    self.load(server.global_lora)
                    assert shared.tensor_hash(shared.adapter(self.model)) == common_hash
                    shared.status('smoke_local_training' if self.smoke else 'local_training', round=t, dataset=d)
                    value, record = self.train(client, d); shared.save_state(local_path, value)
                    state['pending'][client.id] = dict(lora=value, budget=record, start_hash=common_hash)
                    shared.save_state(checkpoint, state)
                    shared.event(output, t, 'local_checkpoint_saved', dataset=d, checkpoint=str(local_path.relative_to(ROOT)))
                item = state['pending'][client.id]
                assert item['start_hash'] == common_hash
                assert shared.tensor_hash(torch.load(local_path, map_location='cpu', weights_only=True)) == shared.tensor_hash(item['lora'])
                self.load(item['lora'])
                shared.status('smoke_local_after' if self.smoke else 'local_after', round=t, dataset=d)
                scores = self.score(current, (d,), t, local_path)
                local[d + '_accuracy'] = scores[d + '_accuracy']
                client.lora, item['local_after'] = item['lora'], scores
                shared.save_state(checkpoint, state)
                shared.event(output, t, 'local_after', dataset=d, lora_tensor_sha256=shared.tensor_hash(item['lora']))
            assert len(state['pending']) == 5 and all('local_after' in item for item in state['pending'].values())
            shared.event(output, t, 'aggregation_start', weights=self.weights)
            FTBaseServer.aggregate(server)
            global_lora = shared.adapter(self.model)
            assert all(torch.isfinite(v).all() for v in global_lora.values())
            global_path = directory / 'global/lora_weights.pt'; shared.save_state(global_path, global_lora)
            global_hash = shared.tensor_hash(global_lora)
            shared.event(output, t, 'aggregation_complete', lora_tensor_sha256=global_hash)
            shared.status('smoke_global_after' if self.smoke else 'global_after', round=t)
            after = self.score(directory / 'global', DOMAINS, t, global_path)
            shared.event(output, t, 'global_after', lora_tensor_sha256=global_hash)
            state['records'].append(dict(round=t, global_before=before, local_after=local, global_after=after,
                common_start_tensor_sha256=common_hash, global_after_tensor_sha256=global_hash,
                weights=self.weights, clients={d: state['pending'][i]['budget'] for i, d in enumerate(DOMAINS)}))
            state.update(completed_round=t, global_lora=global_lora, pending={})
            shared.save_state(checkpoint, state); shared.write_diagnostics(output, state['records'])
            print(f'FedAvg {t}/{self.rounds}: macro={after["macro_accuracy"]:.6f}', flush=True)
        self.load(state['global_lora'])
        data.dump(output / 'result.json', state['records'][-1]['global_after'])
        (output / 'completed.txt').write_text('Five-client FedAvg complete; own-domain diagnostics only.\n', encoding='utf-8')
        return state['records']

    def centralized(self):
        import torch
        from alg.base import BaseClient
        from alg.ftbase import FTBaseClient
        from utils.train_utils import Trainer
        output = self.init_output('centralized'); checkpoint = output / 'resume.pt'
        dataset = MixedDataset(self.datasets)
        assert len(dataset) == sum(self.counts.values())
        args = copy.copy(self.args); args.suffix, args.cn = str(output), 1
        class CentralClient(FTBaseClient):
            def __init__(self):
                BaseClient.__init__(self, 0, args)
        client = CentralClient()
        client.server = SimpleNamespace(round=0)
        client.dataset, client.tokenizer = {'train': dataset}, self.tokenizer
        client.trainer = Trainer(args, client.dataset, client)
        assert not client.trainer.train_loader.drop_last
        client.trainer.train_loader = CentralProgress(client.trainer.train_loader, client)
        state = (torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists()
                 else dict(epoch=0, lora=self.initial, curves=[]))
        self.load(state['lora']); self.trim_exposure(output / 'exposure.jsonl', state['epoch'])
        for epoch in range(state['epoch'] + 1, self.epochs + 1):
            client.server.round = epoch - 1
            before = shared.tensor_hash(shared.adapter(self.model))
            if epoch == 1: assert before == shared.tensor_hash(self.initial)
            shared.status('smoke_centralized' if self.smoke else 'centralized', epoch=epoch)
            loss = client.trainer.train(self.model)
            assert client.last_seen_count == len(dataset)
            assert client.last_optimizer_updates == math.ceil(len(dataset) / 8)
            value = shared.adapter(self.model)
            assert all(torch.isfinite(v).all() for v in value.values())
            state['curves'].append(dict(epoch=epoch, train_loss=loss, samples=len(dataset),
                optimizer_updates=client.last_optimizer_updates, before_lora_tensor_sha256=before,
                after_lora_tensor_sha256=shared.tensor_hash(value)))
            state.update(epoch=epoch, lora=value)
            shared.save_state(checkpoint, state)
            shared.save_state(output / f'epoch_{epoch:02d}_lora.pt', value)
            data.dump(output / 'training_epochs.json', state['curves'])
            print(f'Centralized {epoch}/{self.epochs}: loss={loss:.6f}', flush=True)
        self.load(state['lora'])
        final = output / 'final_lora.pt'; shared.save_state(final, state['lora'])
        shared.status('smoke_centralized_evaluation' if self.smoke else 'centralized_evaluation', epoch=self.epochs)
        scores = self.score(output, DOMAINS, self.epochs, final)
        (output / 'completed.txt').write_text('Centralized fixed final epoch evaluated.\n', encoding='utf-8')
        return scores

    def budget_audit(self):
        output = self.outputs['fedavg']; records = data.rows(output / 'exposure.jsonl')
        assert len(records) == self.rounds * 5
        assert {(r['round'], r['client']) for r in records} == {(t, c) for t in range(1, self.rounds + 1) for c in range(5)}
        budgets = []
        for i, d in enumerate(DOMAINS):
            ids = {data.native_key(row) for row in self.train_rows[d]}
            current = [r for r in records if r['client'] == i]
            for t, row in enumerate(current, 1):
                assert row['round'] == t and row['samples'] == len(row['ids']) == len(set(row['ids'])) == len(ids)
                assert set(row['ids']) == ids and row['optimizer_updates'] == math.ceil(len(ids) / 8)
                assert row['supervised_tokens'] == self.supervised_counts[d]
            budgets.append(dict(method='FedAvg', dataset=d, epochs=self.rounds, samples_per_epoch=len(ids),
                sample_visits=sum(r['samples'] for r in current), optimizer_updates=sum(r['optimizer_updates'] for r in current)))
        central = data.rows(self.outputs['centralized'] / 'exposure.jsonl')
        union = {data.native_key(row) for current in self.train_rows.values() for row in current}
        assert len(central) == self.epochs
        for t, row in enumerate(central, 1):
            assert row['round'] == t and row['client'] == 0
            assert row['samples'] == len(row['ids']) == len(set(row['ids'])) == len(union)
            assert set(row['ids']) == union and row['optimizer_updates'] == math.ceil(len(union) / 8)
            assert row['supervised_tokens'] == sum(self.supervised_counts.values())
        budgets.append(dict(method='Centralized', dataset='all_five', epochs=self.epochs, samples_per_epoch=len(union),
            sample_visits=sum(r['samples'] for r in central), optimizer_updates=sum(r['optimizer_updates'] for r in central)))
        result = dict(passed=True, rows=budgets, same_frozen_training_union=True)
        if not self.smoke:
            for folder, filename in ((self.outputs['base'], 'reused_results.json'),
                                     (self.outputs['clientlocal'], 'reused_specialists.json')):
                references = data.read(folder / filename)
                for entry in references.values():
                    for key in ('result', 'receipt', 'checkpoint'):
                        if key in entry: assert data.sha(ROOT / entry[key]) == entry[key + '_sha256']
            result['reused_baselines_and_checkpoints_unchanged'] = True
        data.dump(MASTER / ('smoke_budget_audit.json' if self.smoke else 'training_budget_audit.json'), result)
        return result

    def lifecycle_audit(self):
        import torch
        output = self.outputs['fedavg']
        records = data.read(output / 'round_diagnostics.json')['rounds']
        events = data.rows(output / 'events.jsonl')
        assert len(records) == self.rounds
        for i, record in enumerate(records):
            t = i + 1; current = [e for e in events if e['round'] == t]
            starts = [j for j, e in enumerate(current) if e['phase'] == 'aggregation_start']
            assert starts and current[-1]['phase'] == 'global_after'
            for j in starts: assert {e['dataset'] for e in current[:j] if e['phase'] == 'local_after'} == set(DOMAINS)
            for name, expected, domains in [('global_before', record['common_start_tensor_sha256'], list(DOMAINS)),
                    ('global', record['global_after_tensor_sha256'], list(DOMAINS))] + [
                    ('client_' + d, None, [d]) for d in DOMAINS]:
                directory = output / f'round_{t:02d}' / name
                actual = shared.tensor_hash(torch.load(directory / 'lora_weights.pt', map_location='cpu', weights_only=True))
                receipt = data.read(directory / 'evaluation_provenance.json')
                assert receipt['domains'] == domains and receipt['lora_tensor_sha256'] == actual
                if expected: assert actual == expected
            expected_start = shared.tensor_hash(self.initial) if i == 0 else records[i - 1]['global_after_tensor_sha256']
            assert record['common_start_tensor_sha256'] == expected_start
        assert len(data.read(output / 'round_diagnostics.json')['rows']) == self.rounds * 5
        central = self.outputs['centralized']; curves = data.read(central / 'training_epochs.json')
        assert len(curves) == self.epochs
        for i, curve in enumerate(curves):
            assert curve['epoch'] == i + 1
            expected = shared.tensor_hash(self.initial) if i == 0 else curves[i - 1]['after_lora_tensor_sha256']
            assert curve['before_lora_tensor_sha256'] == expected
        final_hash = shared.tensor_hash(torch.load(central / 'final_lora.pt', map_location='cpu', weights_only=True))
        assert final_hash == curves[-1]['after_lora_tensor_sha256']
        assert data.read(central / 'evaluation_provenance.json')['lora_tensor_sha256'] == final_hash
        assert [r['round'] for r in data.rows(central / 'evaluation/metrics.jsonl')] == [self.epochs]
        result = dict(passed=True, fedavg_rounds=self.rounds, own_domain_rows=self.rounds * 5,
            all_local_evaluations_before_aggregation=True, each_client_same_round_start=True,
            previous_global_is_next_round_start=True, both_methods_start_from_canonical=True,
            centralized_final_epoch=self.epochs, no_checkpoint_selection=True, no_cross_domain_matrix=True)
        data.dump(MASTER / ('smoke_lifecycle_audit.json' if self.smoke else 'lifecycle_audit.json'), result)
        return result


def final_report(ex):
    inputs, budgets, lifecycle = data.verify(), ex.budget_audit(), ex.lifecycle_audit()
    results = {m: data.read(ex.outputs[m] / 'result.json') for m in OUTPUTS}
    lines = ['# CosmosQA five-domain comparison (fixed4000, seed42)', '',
        'CosmosQA replaces LogiQA. All five Base/Independent Local epoch10 endpoints are reused; '
        'the other four prepared inputs are byte-identical. FedAvg and Centralized each start fresh from '
        'the canonical initial LoRA and see the exact same frozen 20000 training rows for ten full passes.', '',
        '| Method | ' + ' | '.join(NAMES[d] for d in DOMAINS) + ' | Macro |',
        '| --- | ' + ' | '.join(['---:'] * 6) + ' |']
    for m, label in [('base', 'Base'), ('clientlocal', 'Independent Local epoch10'),
                     ('fedavg', 'FedAvg Round10'), ('centralized', 'Centralized Epoch10')]:
        value = results[m]
        lines.append('| ' + label + ' | ' + ' | '.join(f"{value[d + '_accuracy'] * 100:.2f}%" for d in DOMAINS)
                     + f" | {value['macro_accuracy'] * 100:.2f}% |")
    lines += ['', '| Dataset | FedAvg − Base (pp) | Local − FedAvg (pp) | Centralized − Base (pp) |',
        '| --- | ---: | ---: | ---: |']
    for d in DOMAINS:
        key = d + '_accuracy'
        lines.append(f"| {NAMES[d]} | {100 * (results['fedavg'][key] - results['base'][key]):+.2f} | "
            f"{100 * (results['clientlocal'][key] - results['fedavg'][key]):+.2f} | "
            f"{100 * (results['centralized'][key] - results['base'][key]):+.2f} |")
    lines += ['', '## Every round own-domain Local → Global', '',
        '| Round | ' + ' | '.join(NAMES[d] for d in DOMAINS) + ' |',
        '| ---: | ' + ' | '.join(['---'] * 5) + ' |']
    rounds = data.read(ex.outputs['fedavg'] / 'round_diagnostics.json')['rounds']
    for record in rounds:
        lines.append(f"| {record['round']} | " + ' | '.join(
            f"{100 * record['local_after'][d + '_accuracy']:.2f}% → {100 * record['global_after'][d + '_accuracy']:.2f}%"
            for d in DOMAINS) + ' |')
    lines += ['', '## Training protocol and budget', '',
        'Unchanged frozen BF16 Llama-3.2-1B, q_proj/v_proj LoRA r8/alpha32/dropout0.05/bias=none; '
        'constant lr1e-4, bs1, accumulation8, step0, drop_last=False; original Trainer and choice-likelihood '
        'evaluator. CosmosQA and RACE use the existing passage template. Other prompts are unchanged.',
        'FedAvg: five full-participation clients, 4000 visits/500 updates per client per round, '
        'sample-count weights 0.2, unchanged direct A/B aggregation. Centralized: concatenated frozen rows, '
        'shuffled by the original seeded DataLoader, 20000 visits/2500 updates per full epoch. '
        'Each method totals 200000 visits/25000 updates. Original AdamW resets per full client/pooled epoch: '
        '50 resets in FedAvg, 10 in Centralized; this existing scheduling difference is reported, not tuned.',
        'No IID experiment, cross-domain 5×5 matrix, preservation/tolerance/Fisher analysis, tuning, '
        'resampling or best-checkpoint selection. Formal endpoints are fixed Round10/Epoch10.', '',
        f"Input integrity: {inputs['passed']}; budget audit: {budgets['passed']}; lifecycle audit: {lifecycle['passed']}.",
        f"Tests: {json.dumps(inputs['test_counts'])}; Macro is the unweighted mean of the five domain accuracies.",
        '', '## Artifacts', '',
        f'- Fixed data/manifest/token statistics: `{data.config()["prepared_dir"]}/`.',
        f'- Baseline receipts and original Local checkpoint references: `{MASTER.relative_to(ROOT)}/base/`, `clientlocal/`.',
        f'- FedAvg final global/round checkpoints/own-domain predictions/diagnostics: `{MASTER.relative_to(ROOT)}/fedavg/`.',
        f'- Centralized final checkpoint, ten loss records, predictions and receipt: `{MASTER.relative_to(ROOT)}/centralized/`.',
        f'- Audits and final JSON: `{MASTER.relative_to(ROOT)}/`.',
        'Only descriptive fixed-seed comparisons are made; no causal or mechanism attribution.']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    data.dump(MASTER / 'final_results.json', dict(results=results, input_integrity=inputs,
        budgets=budgets, lifecycle=lifecycle, report=str(REPORT)))
    shared.figures(ex.outputs['fedavg'])
    return REPORT


def run_smoke():
    ex = Experiment(smoke=True)
    ex.base(); ex.fedavg(); ex.centralized()
    budgets, lifecycle = ex.budget_audit(), ex.lifecycle_audit()
    data.dump(MASTER / 'protocol_audit.json', dict(passed=True, smoke_only=True, formal_training_not_started=True,
        smoke_outputs={m: str(p.relative_to(ROOT)) for m, p in ex.outputs.items()},
        smoke_counts=ex.counts, budget=budgets, lifecycle=lifecycle,
        unchanged_trainer=True, unchanged_evaluator=True, no_independent_local_training=True))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('prepare', 'smoke', 'fedavg', 'centralized', 'report'), required=True)
    args = parser.parse_args(); configure(); MASTER.mkdir(parents=True, exist_ok=True)
    with (MASTER / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: raise SystemExit('Five-domain job already running; duplicate launch refused')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            data.prepare()
            if args.job == 'prepare': shared.status('data_ready'); return
            if args.job == 'smoke':
                run_smoke(); shared.status('smoke_complete'); return
            ex = Experiment(); ex.base(); ex.local()
            if args.job == 'fedavg':
                ex.fedavg(); shared.status('fedavg_complete')
            elif args.job == 'centralized':
                assert (ex.outputs['fedavg'] / 'completed.txt').exists()
                ex.centralized(); shared.status('centralized_complete')
            else:
                path = final_report(ex); shared.status('complete', report=str(path))
                (MASTER / 'completed.txt').write_text('CosmosQA Local10 reused -> fresh FedAvg10 -> Centralized10 -> report.\n', encoding='utf-8')
        except BaseException as error:
            shared.status('failed', error=str(error), traceback=traceback.format_exc())
            traceback.print_exc(); raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__': main()
