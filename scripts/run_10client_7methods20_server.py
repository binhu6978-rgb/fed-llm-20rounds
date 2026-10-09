"""Seven unchanged algorithms: fresh 10-client rounds1-20 on the saved split.

Use --alg fedavg (alg.fedit), ffalora, fedsvd, fedexlora, fedrotlora,
flexlora or fedmomentum. Repeat the command to resume this method's own run.
No old round10 checkpoints or split preparation are used.
"""
import argparse
import copy
import csv
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Portable I/O, locks, RNG and progress only; no continuation/preflight/hash calls.
from scripts.run_four_baselines20_server import (
    adapter, cpu, dump, file_lock, load, now, ProgressLoader, read,
    restore_rng, rng_state, rows, save, status,
)

from scripts import fedlora_10client20_data as data

DOMAINS, CLIENTS, SOURCE, COUNTS, MASTER = data.DOMAINS, data.CLIENTS, data.SOURCE, data.COUNTS, data.MASTER
METHODS = ('fedavg', 'ffalora', 'fedsvd', 'fedexlora', 'fedrotlora', 'flexlora', 'fedmomentum')
NAMES = dict(zip(METHODS, ('FedAvg', 'FFA-LoRA', 'FedSVD', 'FedEx-LoRA', 'FedRot-LoRA', 'FlexLoRA', 'FedMomentum')))
ALG_MODULES = {m: ('fedit' if m == 'fedavg' else m) for m in METHODS}
MUTATORS = ('fedexlora', 'fedmomentum')
B_ONLY = ('ffalora', 'fedsvd')
preflight = data.preflight
WARNINGS = {
    'fedexlora': ['Original residual merge omits alpha/r=4; implementation unchanged.'],
    'fedmomentum': ['Original residual merge omits alpha/r=4; implementation unchanged.'],
    'flexlora': ['Original repository SVD factorization retained with s=2.'],
}


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


class RecordingTrainer:
    """Keep the repository trainer and optimizer unchanged; record its loss."""
    def __init__(self, delegate, client):
        self.delegate, self.client = delegate, client

    def train(self, model):
        factor = 'lora_B' if self.client.args.alg in B_ONLY else 'lora_'
        assert {n for n, p in model.named_parameters() if p.requires_grad} == {
            n for n, p in model.named_parameters() if factor in n
        }, 'Unexpected trainable parameters; backbone must stay frozen'
        result = self.delegate.train(model)
        self.client.last_train_loss = result
        return result

    def __getattr__(self, key):
        return getattr(self.delegate, key)


class Experiment:
    def __init__(self, method, cfg):
        import torch
        from peft import get_peft_model
        from utils import cosmosqa_five_mcq as encoding
        from utils.model_utils import load_model, load_tokenizer, load_lora_config
        from utils.seed_utils import set_global_seed
        assert method in METHODS
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Requires a CUDA GPU with BF16 support'
        self.method, self.cfg, self.target = method, cfg, 20
        self.output = MASTER / method
        self.output.mkdir(parents=True, exist_ok=True)
        self.module = importlib.import_module('alg.' + ALG_MODULES[method])
        self.args = SimpleNamespace(
            model_path=str(ROOT / cfg['model_path']), model='llama32_1b_base', device=0,
            mcq=True, task_type='CAUSAL_LM', seed=42, bs=1, grad_accum=8, lr=1e-4,
            epoch=1, step=0, cn=10, sr=1., rnd=20, lora_rank=8, lora_alpha=32,
            lora_dropout=.05, eval_batch_size=8, mcq_domains=list(DOMAINS),
            mcq_dataset_dir=str(ROOT / cfg['prepared_dir']), suffix=str(self.output),
            alg=method, lam=.5, s=2., residual_threshold=.9999, deterministic=True,
        )
        set_global_seed(42, device=0, deterministic=True)
        self.tokenizer = load_tokenizer(self.args)
        prepared = ROOT / cfg['prepared_dir']
        manifest = read(prepared / 'manifest.json')
        source_rows = {d: rows(prepared / 'train' / f'{d}.jsonl') for d in DOMAINS}
        source_datasets = {d: encoding.TrainingDataset(source_rows[d], self.tokenizer) for d in DOMAINS}
        self.datasets = {c: data.IndexedDataset(source_datasets[SOURCE[c]],
            manifest['clients'][c]['source_train_indices']) for c in CLIENTS}
        self.train_rows = {c: self.datasets[c].rows for c in CLIENTS}
        self.test_rows = {d: rows(prepared / 'test' / f'{d}.jsonl') for d in DOMAINS}
        self.test_encoded = {d: [(encoding.encode_prompt(self.tokenizer, r),
            [encoding.continuation_ids(self.tokenizer, r, letter) for letter in 'ABCD'])
            for r in self.test_rows[d]] for d in DOMAINS}
        self.counts = {c: len(self.datasets[c]) for c in CLIENTS}
        self.supervised = {c: sum(sum(v != -100 for v in x['labels']) for x in self.datasets[c].encoded)
                           for c in CLIENTS}
        statistics = read(ROOT / cfg['source_prepared_dir'] / 'sequence_statistics.json')
        for c in CLIENTS:
            assert self.counts[c] == 2000
            assert self.train_rows[c] == rows(prepared / 'clients' / f'{c}.jsonl')
        for d in DOMAINS:
            assert len(self.test_rows[d]) == COUNTS[d]
            assert sum(len(x['input_ids']) for x in source_datasets[d].encoded) == statistics[d]['train']['token_total']
            assert sum(self.supervised[d + '_' + side] for side in ('a', 'b')) == statistics[d]['train']['supervised_token_total']
        encoding.clear_cache()
        set_global_seed(42, device=0, deterministic=True)
        lora_config = load_lora_config(self.args)
        assert lora_config.bias == 'none' and set(lora_config.target_modules) == {'q_proj', 'v_proj'}
        self.model = get_peft_model(load_model(self.args), lora_config)
        self.initial = load(ROOT / cfg['initial_lora'])
        self.shape_check(self.initial)
        self.model.load_state_dict(self.initial, strict=False)
        # FP32 LoRA is the existing initialization, including for torch.svd.
        assert all(v.dtype == torch.float32 for v in self.initial.values())
        assert all(p.dtype == torch.bfloat16 and not p.requires_grad
                   for n, p in self.model.named_parameters() if 'lora_' not in n)
        self.base_keys = tuple(k.replace('.lora_A.default.weight', '.base_layer.weight')
                               for k in self.initial if 'lora_A' in k)
        self.canonical_base = cpu({k: self.model.state_dict()[k] for k in self.base_keys})
        self.protocol = dict(
            format_version=1, method=method, source_round=0, reused_rounds=0, target_round=20,
            config=cfg, model_path=cfg['model_path'], initial_lora=cfg['initial_lora'],
            domains=list(DOMAINS), client_order=list(CLIENTS), client_sources=SOURCE,
            train_counts=self.counts, test_counts=COUNTS, client_count=10,
            weights={c: .1 for c in CLIENTS}, source_weights={d: .2 for d in DOMAINS}, seed=42,
            base_dtype='torch.bfloat16', lora_dtype='torch.float32', base_frozen_during_local=True,
            aggregation_mutates_backbone=method in MUTATORS,
            lora=dict(target_modules=['q_proj', 'v_proj'], rank=8, alpha=32, dropout=.05, bias='none'),
            training=dict(lr=1e-4, batch_size=1, gradient_accumulation=8, local_epochs=1,
                rounds=20, full_participation=True, optimizer='Original Trainer AdamW; recreated per client per round'),
            client_seed_rule='client_round_seed(42, client_id, round-1)',
            total_optimizer_updates=50000, total_sample_visits=400000,
            local_evaluation='Raw upload-before-aggregation model; own-domain full test; equal mean of ten clients',
            global_evaluation='Same aggregated shared model on all five full tests; equal Global Macro',
            checkpoint_selection='Highest unrounded shared Global Macro in rounds1-20; earliest tie',
            client_run='alg.' + ALG_MODULES[method] + '.Client.run',
            server_aggregate=('alg.ftbase.FTBaseServer.aggregate' if method in ('fedavg', 'ffalora', 'fedrotlora')
                              else 'alg.' + method + '.Server.aggregate'),
            algorithm_parameters=dict(lam=.5) if method == 'fedrotlora' else dict(s=2.) if method == 'flexlora'
                else dict(residual_threshold=.9999) if method == 'fedmomentum' else {},
            warnings=WARNINGS.get(method, []),
            trainable_factors=['B'] if method in B_ONLY else ['A', 'B'],
            algorithm_source_modified=False, file_hash_validation=False,
            upload_contents=('B only; fixed initial A saved in every complete state' if method == 'ffalora'
                             else 'A/B; FedSVD redundant frozen upload tensors omitted'),
        )
        protocol_path = self.output / 'protocol.json'
        if protocol_path.exists():
            assert read(protocol_path) == self.protocol, 'Incompatible existing method protocol'
        else:
            # A failed input/GPU check can leave only a failure status and lock.
            if any(p.name not in ('run.lock', 'status.json') for p in self.output.iterdir()):
                raise RuntimeError(f'Output has artifacts without a protocol; inspect it before proceeding: {self.output}')
            dump(protocol_path, self.protocol)
        packages = {n: importlib.metadata.version(n) for n in
                    ('torch', 'transformers', 'peft', 'numpy', 'datasets', 'safetensors', 'PyYAML', 'accelerate')}
        environment = dict(python=sys.version, platform=sys.platform, packages=packages,
            torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cross_machine_bitwise_identity_guaranteed=False)
        dump(self.output / 'environment_receipt.json', environment)
        print('METHOD', NAMES[method], 'fresh rounds1-20; 10 clients x 2000; train=20000 test=7498', flush=True)
        print('ENVIRONMENT', environment, flush=True)

    def shape_check(self, lora):
        import torch
        template = {k: v for k, v in self.model.state_dict().items() if 'lora_' in k}
        assert set(lora) == set(template) and len(lora) == 64
        assert all(v.shape == template[k].shape and v.dtype == template[k].dtype
                   and torch.isfinite(v).all() for k, v in lora.items())
        assert all((v.shape[0] if 'lora_A' in k else v.shape[1]) == 8 for k, v in lora.items())

    def capture(self):
        # Persist every backbone projection modified by FedEx/FedMomentum.
        return dict(format_version=1, method=self.method, lora=adapter(self.model),
                    base_weights=cpu({k: self.model.state_dict()[k] for k in self.base_keys})
                        if self.method in MUTATORS else {},
                    canonical_model=self.cfg['model_path'], canonical_initial_lora=self.cfg['initial_lora'])

    def restore(self, state):
        import torch
        assert state['method'] == self.method
        assert state['canonical_model'] == self.cfg['model_path']
        assert state['canonical_initial_lora'] == self.cfg['initial_lora']
        self.shape_check(state['lora'])
        base = state['base_weights'] if self.method in MUTATORS else self.canonical_base
        assert bool(state['base_weights']) == (self.method in MUTATORS)
        assert set(base) == set(self.base_keys)
        assert all(v.shape == self.canonical_base[k].shape and v.dtype == self.canonical_base[k].dtype
                   and torch.isfinite(v).all() for k, v in base.items())
        self.model.load_state_dict(base, strict=False)
        self.model.load_state_dict(state['lora'], strict=False)
        factor = 'lora_B' if self.method in B_ONLY else 'lora_'
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = factor in name

    def check_local_base(self, start):
        import torch
        expected = start['base_weights'] if self.method in MUTATORS else self.canonical_base
        current = self.model.state_dict()
        assert all(torch.equal(current[k].detach().cpu(), v) for k, v in expected.items()), 'Local frozen backbone changed'

    def bind(self):
        from alg.base import BaseClient, BaseServer
        from utils.train_utils import Trainer
        server = self.module.Server.__new__(self.module.Server)
        BaseServer.__init__(server, self.args, [])
        server.model, server.round, server.sample_rate, server.wall_clock_time = self.model, 0, 1., 0.
        server.global_lora = {}
        if self.method == 'flexlora':
            server.r, server.s = self.args.lora_rank, self.args.s
        if self.method == 'fedmomentum':
            server.tau = self.args.residual_threshold
        ex = self
        clients = []
        for name in CLIENTS:
            class Client(ex.module.Client):
                def __init__(client, d):
                    args = copy.copy(ex.args)
                    BaseClient.__init__(client, CLIENTS.index(d), args)
                    client.server, client.dataset = server, {'train': ex.datasets[d]}
                    client.tokenizer, client.lora, client.delay = ex.tokenizer, {}, 1.
                    client.evaluator, client.last_train_loss = None, None
                    delegate = Trainer(args, client.dataset, client)
                    assert not delegate.train_loader.drop_last and len(delegate.train_loader) == ex.counts[d]
                    delegate.train_loader = ProgressLoader(delegate.train_loader, ex, d)
                    client.trainer = RecordingTrainer(delegate, client)
            clients.append(Client(name))
        server.clients = clients
        return server, clients

    def validate_predictions(self, evaluation, domains, t, result):
        path = evaluation / f'round_{t:02d}_predictions.csv'
        with path.open(encoding='utf-8', newline='') as f:
            predictions = list(csv.DictReader(f))
        assert len(predictions) == sum(len(self.test_rows[d]) for d in domains)
        assert set(r['domain'] for r in predictions) == set(domains)
        for d in domains:
            current = [r for r in predictions if r['domain'] == d]
            gold = {str(r['id']): r['correct_answer'] for r in self.test_rows[d]}
            assert len(current) == len({r['id'] for r in current}) == len(gold)
            assert {r['id'] for r in current} == set(gold)
            for r in current:
                assert r['gold'] == gold[r['id']] and r['prediction'] in 'ABCD'
                assert int(r['correct']) == int(r['gold'] == r['prediction'])
                assert all(math.isfinite(float(r['logprob_' + letter])) for letter in 'ABCD')
            assert abs(sum(int(r['correct']) for r in current) / len(current) - result[d + '_accuracy']) < 1e-12
        assert abs(sum(result[d + '_accuracy'] for d in domains) / len(domains) - result['macro_accuracy']) < 1e-12
        assert result['worst_domain_accuracy'] == min(result[d + '_accuracy'] for d in domains)

    def score(self, directory, domains, t, checkpoint):
        from utils.mcq_eval import MCQEvaluator
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(method=self.method, round=t, domains=list(domains), checkpoint=relative(checkpoint),
            canonical_model=self.cfg['model_path'], aggregation_mutates_backbone=self.method in MUTATORS,
            local_backbone_source='round_start_checkpoint', local_model_before_upload_alignment=True,
            model_role='global_after_aggregation' if len(domains) == 5 else 'raw_local_before_upload',
            round_start_checkpoint=relative(self.output / 'initial_complete_state.pt') if t == 1
                else relative(self.output / f'round_{t-1:02d}/global/complete_state.pt'))
        provenance = directory / 'evaluation_provenance.json'
        if provenance.exists():
            assert read(provenance) == identity, 'Cached evaluation belongs to another round/model role'
        else:
            dump(provenance, identity)
        evaluation = directory / 'evaluation'
        if evaluation.exists():
            metric = rows(evaluation / 'metrics.jsonl')
            assert len(metric) == 1 and metric[0]['round'] == t
            result = {k: v for k, v in metric[0].items() if k != 'round'}
            self.validate_predictions(evaluation, domains, t, result)
        else:
            # Only an entire validated evaluation directory becomes a cache hit.
            staging = directory / 'evaluation_pending'
            staging.mkdir(exist_ok=True)
            for name in ('metrics.jsonl', f'round_{t:02d}_predictions.csv'):
                (staging / name).unlink(missing_ok=True)
            evaluator = MCQEvaluator.__new__(MCQEvaluator)
            evaluator.args, evaluator.tokenizer, evaluator.domains = self.args, self.tokenizer, tuple(domains)
            evaluator.rows = {d: self.test_rows[d] for d in domains}
            evaluator.encoded = {d: self.test_encoded[d] for d in domains}
            evaluator.output = staging
            result = evaluator.evaluate(self.model, t)
            self.validate_predictions(staging, domains, t, result)
            os.replace(staging, evaluation)
        dump(directory / 'result.json', result)
        return result

    def report(self, records):
        assert records and [r['round'] for r in records] == list(range(1, len(records) + 1))
        best = max(records, key=lambda r: (r['metrics']['macro_accuracy'], -r['round']))
        combined = [dict(round=r['round'], local_mean=r['local_mean'],
            global_macro=r['metrics']['macro_accuracy'],
            local_minus_global_pp=100 * (r['local_mean'] - r['metrics']['macro_accuracy'])) for r in records]
        dump(self.output / 'round_trajectory.json', records)
        dump(self.output / 'local_global_rounds.json', combined)
        for name, values in (('local_global_rounds.csv', combined), ('round_trajectory.csv', [
                dict(round=r['round'], local_mean=r['local_mean'], **r['metrics']) for r in records])):
            path = self.output / name
            temporary = path.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader(); writer.writerows(values)
            os.replace(temporary, path)
        last = records[-1]
        dump(self.output / 'result.json', dict(method=self.method, completed_round=last['round'],
            target_round=self.target, reused_rounds=0, best_round=best['round'],
            peak_shared_macro=best['metrics']['macro_accuracy'], best_shared_macro=best['metrics']['macro_accuracy'],
            best_metrics=best['metrics'], best_task_accuracies={d: best['metrics'][d + '_accuracy'] for d in DOMAINS},
            best_checkpoint=best['checkpoint'], final_metrics=last['metrics'],
            final_checkpoint=last['checkpoint'], final_local_mean=last['local_mean'],
            best_predictions=best['checkpoint'].rsplit('/', 1)[0] + f"/evaluation/round_{best['round']:02d}_predictions.csv",
            final_predictions=last['checkpoint'].rsplit('/', 1)[0] + f"/evaluation/round_{last['round']:02d}_predictions.csv",
            round20_macro=last['metrics']['macro_accuracy'] if last['round'] == 20 else None,
            checkpoint_selection='Unrounded global Macro; earliest tied round'))
        lines = ['# ' + NAMES[self.method] + ' 10-client rounds1-20', '',
            'Fresh from canonical seed42 LoRA; ten original split clients. No historical rounds imported.', '',
            '| Round | Local own-domain mean (%) | Global Macro (%) | Local - Global (pp) |',
            '|---:|---:|---:|---:|']
        for r in combined:
            lines.append('| %d | %.2f | %.2f | %+.2f |' % (r['round'], r['local_mean'] * 100,
                r['global_macro'] * 100, r['local_minus_global_pp']))
        lines += ['', f"Best shared round: {best['round']}; Macro {best['metrics']['macro_accuracy'] * 100:.2f}%."]
        (self.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def trim_exposure(self, state):
        exposure = self.output / 'exposure.jsonl'
        if exposure.exists():
            kept = [r for r in rows(exposure) if r['round'] <= state['completed_round'] or
                (r['round'] == state['completed_round'] + 1 and r['client'] in state['pending'])]
            temporary = exposure.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(r) + '\n' for r in kept), encoding='utf-8')
            os.replace(temporary, exposure)

    def run(self):
        import torch
        resume = self.output / 'resume.pt'
        if resume.exists():
            state = load(resume)
        else:
            self.model.load_state_dict(self.initial, strict=False)
            initial = self.capture()
            save(self.output / 'initial_complete_state.pt', initial)
            state = dict(completed_round=0, global_state=initial, pending={}, pending_global=None,
                         records=[], aggregation_rng=None)
            save(resume, state)
        assert 0 <= state['completed_round'] <= self.target
        assert [r['round'] for r in state['records']] == list(range(1, state['completed_round'] + 1))
        assert set(state['pending']).issubset(range(len(CLIENTS)))
        self.trim_exposure(state)
        self.restore(state['global_state'])
        server, clients = self.bind()
        if state['records']:
            self.report(state['records'])
        for t in range(state['completed_round'] + 1, self.target + 1):
            self.current_round = t
            server.round = t - 1
            server.sample()
            assert [c.id for c in server.sampled_clients] == list(range(len(CLIENTS)))
            start = state['global_state']
            self.restore(start)
            server.global_lora = {k: v.to(self.model.device).clone() for k, v in start['lora'].items()
                                 if self.method != 'ffalora' or 'lora_B' in k}
            directory = self.output / f'round_{t:02d}'
            if state['pending_global'] is None:
                for client, c in zip(clients, CLIENTS):
                    d = SOURCE[c]
                    current = directory / f'client_{c}'
                    raw_path, upload_path = current / 'raw_lora.pt', current / 'upload_lora.pt'
                    if client.id not in state['pending']:
                        self.restore(start)
                        status(self.output / 'status.json', 'local_training', method=self.method, round=t, client=c, domain=d)
                        client.run(self.model)  # Unchanged Client.run, including FedRot upload alignment.
                        assert client.last_seen_count == self.counts[c]
                        assert client.last_optimizer_updates == (self.counts[c] + 7) // 8
                        raw = adapter(self.model)
                        self.shape_check(raw)
                        if self.method in B_ONLY:
                            assert all(torch.equal(raw[k], v) for k, v in start['lora'].items() if 'lora_A' in k), 'Local A changed'
                        self.check_local_base(start)
                        # FedSVD clones the full state, but its aggregate reads only B.
                        # Drop redundant frozen tensors immediately; keep A/B as the upload.
                        upload = cpu({k: v for k, v in client.lora.items() if 'lora_' in k})
                        client.lora = {}
                        expected = {k for k in self.initial if self.method != 'ffalora' or 'lora_B' in k}
                        assert set(upload) == expected
                        assert all(v.shape == self.initial[k].shape and v.dtype == self.initial[k].dtype
                                   and torch.isfinite(v).all() for k, v in upload.items())
                        if self.method != 'fedrotlora':
                            assert all(torch.equal(v, raw[k]) for k, v in upload.items())
                        save(raw_path, raw); save(upload_path, upload)
                        state['pending'][client.id] = dict(round=t, client=c, source=d, lora=upload,
                            budget=dict(train_loss=client.last_train_loss, samples=client.last_seen_count,
                                        optimizer_updates=client.last_optimizer_updates))
                        state['aggregation_rng'] = rng_state()
                        save(resume, state)
                    item = state['pending'][client.id]
                    assert item['round'] == t
                    if 'local_after' not in item:
                        local_state = dict(start, lora=load(raw_path))
                        self.restore(local_state)
                        status(self.output / 'status.json', 'local_evaluation', method=self.method, round=t, client=c, domain=d)
                        item['local_after'] = self.score(current, [d], t, raw_path)
                        save(resume, state)
                    client.lora = {k: v.to(self.model.device).clone() for k, v in item['lora'].items()}
                assert len(state['pending']) == len(CLIENTS)
                self.restore(start)
                restore_rng(state['aggregation_rng'])
                status(self.output / 'status.json', 'aggregation', method=self.method, round=t)
                server.aggregate()  # Original math; each client contributes 0.1, each source 0.2.
                value = self.capture()
                self.shape_check(value['lora'])
                assert all(torch.isfinite(v).all() for v in value['base_weights'].values())
                if self.method not in MUTATORS:
                    self.check_local_base(start)
                if self.method == 'ffalora':
                    assert all(torch.equal(value['lora'][k], v) for k, v in self.initial.items() if 'lora_A' in k)
                state['pending_global'] = value
                save(directory / 'global/complete_state.pt', value)
                save(resume, state)
            value = state['pending_global']
            self.restore(value)
            checkpoint = directory / 'global/complete_state.pt'
            status(self.output / 'status.json', 'global_evaluation', method=self.method, round=t)
            scores = self.score(directory / 'global', DOMAINS, t, checkpoint)
            local_accuracies = {c: state['pending'][i]['local_after'][SOURCE[c] + '_accuracy']
                                for i, c in enumerate(CLIENTS)}
            local_mean = sum(local_accuracies.values()) / len(CLIENTS)
            state['records'].append(dict(round=t, metrics=scores, local_mean=local_mean,
                local_accuracies=local_accuracies, checkpoint=relative(checkpoint),
                weights={c: .1 for c in CLIENTS}, source_weights={d: .2 for d in DOMAINS},
                start_checkpoint=relative(self.output / 'initial_complete_state.pt') if t == 1 else
                    relative(self.output / f'round_{t-1:02d}/global/complete_state.pt'),
                clients={c: {k: v for k, v in state['pending'][i].items() if k != 'lora'}
                         for i, c in enumerate(CLIENTS)}))
            state.update(completed_round=t, global_state=value, pending={}, pending_global=None, aggregation_rng=None)
            save(resume, state)
            for client in clients:
                client.lora = {}
            self.report(state['records'])
            print('ROUND', self.method, t, 'LOCAL MEAN', local_mean, 'GLOBAL MACRO', scores['macro_accuracy'], flush=True)
        self.audit(state)
        (self.output / 'completed.txt').write_text('All fresh rounds1-20 and full local/global evaluations completed.\n', encoding='utf-8')
        status(self.output / 'status.json', 'complete', method=self.method, round=self.target)

    def audit(self, state):
        """Budget, outputs and prediction-count checks; no file hash audit."""
        import torch
        records = state['records']
        assert state['completed_round'] == self.target and not state['pending'] and state['pending_global'] is None
        assert [r['round'] for r in records] == list(range(1, self.target + 1))
        visits = rows(self.output / 'exposure.jsonl')
        assert len(visits) == len(CLIENTS) * self.target
        assert {(r['round'], r['client']) for r in visits} == {
            (t, i) for t in range(1, self.target + 1) for i in range(len(CLIENTS))}
        for r in visits:
            c = CLIENTS[r['client']]
            assert r['samples'] == len(r['ids']) == len(set(r['ids'])) == self.counts[c]
            assert set(r['ids']) == {data.native_key(x) for x in self.train_rows[c]}
            assert r['optimizer_updates'] == (self.counts[c] + 7) // 8
            assert r['supervised_tokens'] == self.supervised[c]
        initial = load(self.output / 'initial_complete_state.pt')
        assert all(torch.equal(initial['lora'][k], v) for k, v in self.initial.items())
        for r in records:
            global_dir = self.output / f"round_{r['round']:02d}/global"
            saved = load(global_dir / 'complete_state.pt')
            self.shape_check(saved['lora'])
            assert bool(saved['base_weights']) == (self.method in MUTATORS)
            assert read(global_dir / 'result.json') == r['metrics']
            self.validate_predictions(global_dir / 'evaluation', DOMAINS, r['round'], r['metrics'])
            for c in CLIENTS:
                d = SOURCE[c]
                current = global_dir.parent / f'client_{c}'
                assert (current / 'raw_lora.pt').is_file() and (current / 'upload_lora.pt').is_file()
                assert read(current / 'result.json') == r['clients'][c]['local_after']
                self.validate_predictions(current / 'evaluation', [d], r['round'], r['clients'][c]['local_after'])
            assert set(r['clients']) == set(CLIENTS) and r['weights'] == {c: .1 for c in CLIENTS}
            assert abs(sum(r['clients'][c]['local_after'][SOURCE[c] + '_accuracy'] for c in CLIENTS) / len(CLIENTS) - r['local_mean']) < 1e-12
        updates, samples = sum(r['optimizer_updates'] for r in visits), sum(r['samples'] for r in visits)
        assert updates == sum((n + 7) // 8 for n in self.counts.values()) * self.target
        assert samples == sum(self.counts.values()) * self.target
        if self.target == 20:
            assert updates == 50000 and samples == 400000
        dump(self.output / 'protocol_audit.json', dict(passed=True, method=self.method, rounds=self.target,
            reused_rounds=0, client_epochs=len(visits), optimizer_updates=updates, sample_visits=samples,
            raw_local_test_each_round=True, full_test_sets=True,
            trainable_factors=['B'] if self.method in B_ONLY else ['A', 'B'],
            base_frozen_during_local=True, updated_backbone_persisted=self.method in MUTATORS,
            one_shared_global_model=True, algorithm_source_modified=False,
            file_hash_validation=False, test_counts=COUNTS))


def configure_environment(gpu=None):
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def is_complete(method):
    """A completed method is reused without model allocation/training/evaluation."""
    folder = MASTER / method
    if not (folder / 'completed.txt').is_file():
        return False
    audit, result = read(folder / 'protocol_audit.json'), read(folder / 'result.json')
    protocol, records = read(folder / 'protocol.json'), read(folder / 'round_trajectory.json')
    assert audit['passed'] and audit['method'] == method and audit['rounds'] == 20
    assert audit['optimizer_updates'] == 50000 and audit['sample_visits'] == 400000
    assert audit['reused_rounds'] == protocol['reused_rounds'] == 0
    assert protocol['method'] == method and protocol['client_order'] == list(CLIENTS)
    assert protocol['config'] == data.config() and protocol['target_round'] == 20
    assert result['method'] == method and result['completed_round'] == result['target_round'] == 20
    assert [r['round'] for r in records] == list(range(1, 21))
    best = max(records, key=lambda r: (r['metrics']['macro_accuracy'], -r['round']))
    assert result['best_round'] == best['round'] and result['best_metrics'] == best['metrics']
    assert result['best_checkpoint'] == best['checkpoint']
    assert result['round20_macro'] == records[-1]['metrics']['macro_accuracy']
    for r in records:
        assert set(r['clients']) == set(CLIENTS)
        global_dir = folder / f"round_{r['round']:02d}/global"
        assert (global_dir / 'complete_state.pt').is_file()
        assert (global_dir / 'evaluation' / f"round_{r['round']:02d}_predictions.csv").is_file()
        for c in CLIENTS:
            current = global_dir.parent / f'client_{c}'
            assert (current / 'raw_lora.pt').is_file() and (current / 'upload_lora.pt').is_file()
            assert (current / 'evaluation' / f"round_{r['round']:02d}_predictions.csv").is_file()
    return True


def run_method(method):
    folder = MASTER / method
    with file_lock(folder / 'run.lock'):
        try:
            cfg = preflight()
            if is_complete(method):
                status(folder / 'status.json', 'complete', method=method, round=20, reused_completed=True)
                print(NAMES[method], 'already completed20; skipped', flush=True)
                return
            ex = Experiment(method, cfg)
            ex.run()
        except BaseException as error:
            status(folder / 'status.json', 'failed', method=method,
                   error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alg', choices=METHODS, required=True, help='Method (fedavg uses alg.fedit)')
    parser.add_argument('--check', action='store_true', help='Read fixed input counts/IDs; no model loading or training')
    parser.add_argument('--gpu', help='Optional CUDA_VISIBLE_DEVICES; default preserves the scheduler environment')
    args = parser.parse_args()
    configure_environment(args.gpu)
    if args.check:
        preflight()
        print('INPUT CHECK PASSED: saved 10-client split; fresh rounds1-20; 20000 train / 7498 test', flush=True)
        return
    run_method(args.alg)


if __name__ == '__main__':
    main()
