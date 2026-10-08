"""Centralized-All and balanced FedAvg-IID on the existing fixed train pool.

No new training-example selection. Reuses the existing MCQ Trainer and
FTBaseServer.aggregate; the latter is the identity for the centralized client.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mcq_dataset_skewed as original

DOMAINS = original.DOMAINS
CONFIG = ROOT / 'configs/mcq_controls_seed42.yaml'
RUNS = {method: ROOT / f'exp/mcq_{method}_seed42' for method in ('centralized', 'iid')}
read, sha, dump, save_state = original.read, original.sha, original.dump, original.save_state


def settings():
    import yaml
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(model_path='models/models--llama3.2-1B',
        initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt',
        dataset_dir='dataset/mcq_balanced4000', domains=list(DOMAINS), seed=42,
        rounds_or_epochs=10, batch_size=1, gradient_accumulation=8, local_epochs=1,
        step=0, learning_rate=1e-4, lora_rank=8, lora_alpha=32, lora_dropout=0.05,
        eval_batch_size=8, centralized_samples=16000, iid_clients=4,
        iid_samples_per_domain_per_client=1000, iid_aggregation_weights=[0.25] * 4,
        final_test_epoch_or_round=10)
    if value != expected:
        raise ValueError('Controls differ from the fixed comparison protocol')
    return value


def key(row):
    return row['dataset'] + ':' + str(row['id'])


def partition(method):
    """Shuffle membership only, never resample the frozen 16000 examples."""
    settings()
    rows = original.train_rows()
    if method == 'centralized':
        clients = [[row for domain in DOMAINS for row in rows[domain]]]
    else:
        rng = random.Random(42)
        clients = [[] for _ in range(4)]
        for domain in DOMAINS:
            current = list(rows[domain])
            rng.shuffle(current)
            for i in range(4):
                clients[i].extend(current[i * 1000:(i + 1) * 1000])
    source_ids = {key(row) for domain in DOMAINS for row in rows[domain]}
    all_ids = [key(row) for client in clients for row in client]
    assert len(all_ids) == len(set(all_ids)) == len(source_ids) == 16000
    assert set(all_ids) == source_ids
    for client in clients:
        assert len(client) == (16000 if method == 'centralized' else 4000)
        assert {d: sum(row['dataset'] == d for row in client) for d in DOMAINS} == {
            d: 4000 if method == 'centralized' else 1000 for d in DOMAINS}
    manifest = dict(method=method, partition_seed=42,
        partition_rule='One Random(42), shuffle each domain in RACE/LogiQA/OpenBookQA/SciQ order; four disjoint consecutive 1000-row slices.'
                       if method == 'iid' else 'Concatenate all fixed train rows in domain order.',
        new_training_example_selection=False, test_used=False,
        train_sha256={d: sha(ROOT / settings()['dataset_dir'] / d / 'train.jsonl') for d in DOMAINS},
        clients=[dict(client=i, count=len(client),
            domain_counts={d: sum(row['dataset'] == d for row in client) for d in DOMAINS},
            ids=[key(row) for row in client]) for i, client in enumerate(clients)])
    path = RUNS[method] / 'partition.json'
    if path.exists():
        assert read(path) == manifest, 'Fixed partition changed'
    else:
        dump(path, manifest)
    return clients, manifest


class MixedDataset:
    def __init__(self, rows, encoded):
        self.rows = rows
        self.encoded = [encoded[key(row)] for row in rows]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]


def datasets_for(method, tokenizer, args):
    sources = original.make_datasets(tokenizer, args)
    encoded = {key(row): item for dataset in sources for row, item in zip(dataset.rows, dataset.encoded)}
    clients, _ = partition(method)
    return [MixedDataset(rows, encoded) for rows in clients]


def check():
    from utils.model_utils import load_tokenizer
    # The original all-row check covers the encoders and real aggregation/reset.
    original.check()
    args = original.args_for_run()
    tokenizer = load_tokenizer(args)
    source = original.make_datasets(tokenizer, args)
    encoded = {key(row): item for dataset in source for row, item in zip(dataset.rows, dataset.encoded)}
    results = {}
    for method in RUNS:
        clients, manifest = partition(method)
        for rows in clients:
            dataset = MixedDataset(rows, encoded)
            assert all(dataset[i] is encoded[key(row)] for i, row in enumerate(rows))
            assert all(dataset[i]['native_key'] == key(row) for i, row in enumerate(rows))
        result = dict(passed=True, method=method, training_performed=False, test_read=False,
                      exact_original_encodings=True, total_train=16000,
                      clients=[{k: v for k, v in entry.items() if k != 'ids'} for entry in manifest['clients']],
                      runner_sha256=sha(Path(__file__)))
        dump(RUNS[method] / 'implementation_checks.json', result)
        results[method] = result
        print(f'{method}: frozen union, disjoint membership, per-domain counts and exact encodings PASS', flush=True)
    return results


class ProgressLoader:
    def __init__(self, loader, client, output):
        self.loader, self.client, self.output = loader, client, output
        self.drop_last = loader.drop_last

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        started = time.monotonic()
        for i, batch in enumerate(self.loader, 1):
            yield batch
            if i % 200 == 0:
                status = dict(round_or_epoch=self.client.server.round + 1,
                    client=self.client.id, samples_finished=i, samples_total=len(self),
                    optimizer_updates_finished=i // 8, elapsed_seconds=time.monotonic() - started)
                dump(self.output / 'live_progress.json', status)
                print(f"Epoch/round {status['round_or_epoch']} client {self.client.id}: "
                      f"{i}/{len(self)} samples, {status['elapsed_seconds']:.1f}s", flush=True)


def evaluate(model, args, tokenizer, output, baseline):
    from utils.mcq_eval import MCQEvaluator
    from utils.mcq_utils import read_jsonl, encode_prompt, continuation_ids, LETTERS
    from utils.race_mcq import encode_race

    class FinalEvaluator(MCQEvaluator):
        def __init__(self):
            self.args, self.tokenizer, self.domains = args, tokenizer, DOMAINS
            self.output = output / 'evaluation'
            self.output.mkdir(parents=True, exist_ok=True)
            self.rows, self.encoded = {}, {}
            for domain in DOMAINS:
                folder = ROOT / settings()['dataset_dir'] / domain
                assert sha(folder / 'test.jsonl') == read(folder / 'manifest.json')['output_sha256']['test.jsonl']
                rows = read_jsonl(folder / 'test.jsonl')
                assert len(rows) == baseline['datasets'][domain]['official_test_count']
                self.rows[domain] = rows
                self.encoded[domain] = [original_encode(tokenizer, row) for row in rows]

    def original_encode(tokenizer, row):
        if row['dataset'] == 'race':
            return encode_race(tokenizer, row, args.max_length)[:2]
        return encode_prompt(tokenizer, row), [continuation_ids(tokenizer, row, letter) for letter in LETTERS]

    return FinalEvaluator().evaluate(model, 10)


def audit(method):
    from utils.mcq_utils import read_jsonl
    output = RUNS[method]
    _, manifest = partition(method)
    expected = {entry['client']: set(entry['ids']) for entry in manifest['clients']}
    records = read_jsonl(output / 'exposure.jsonl')
    assert len(records) == 10 * len(expected)
    assert {(r['round'], r['client']) for r in records} == {(r, c) for r in range(1, 11) for c in expected}
    for r in records:
        ids = expected[r['client']]
        assert r['samples'] == len(r['ids']) == len(set(r['ids'])) == len(ids)
        assert set(r['ids']) == ids and r['optimizer_updates'] == len(ids) // 8
    curves = read(output / 'training_rounds.json')
    assert len(curves) == 10
    weights = [1 / len(expected)] * len(expected)
    assert all(c['round_or_epoch'] == i and c['samples_visited'] == 16000 and c['optimizer_updates'] == 2000
               and c['weights'] == weights
               for i, c in enumerate(curves, 1))
    dump(output / 'exposure_audit.json', dict(passed=True, sample_visits=160000,
         optimizer_updates=20000, client_epochs=len(records), fixed_partition=True))


def report(method):
    from utils.mcq_utils import read_jsonl
    audit(method)
    output = RUNS[method]
    metrics = read_jsonl(output / 'evaluation/metrics.jsonl')
    assert len(metrics) == 1 and metrics[0]['round'] == 10
    metrics = metrics[0]
    baseline = read(ROOT / 'exp/mcq_base_local_seed42/manifest.json')['datasets']
    with (output / 'evaluation/round_10_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    assert {row['domain'] for row in predictions} == set(DOMAINS)
    for domain in DOMAINS:
        current = [p for p in predictions if p['domain'] == domain]
        assert len(current) == len({p['id'] for p in current}) == baseline[domain]['official_test_count']
        assert all(int(p['correct']) == int(p['gold'] == p['prediction']) for p in current)
        assert abs(sum(int(p['correct']) for p in current) / len(current) - metrics[domain + '_accuracy']) < 1e-12
    assert abs(sum(metrics[d + '_accuracy'] for d in DOMAINS) / 4 - metrics['macro_accuracy']) < 1e-12
    comparisons = {}
    label = 'Centralized-All Epoch10' if method == 'centralized' else 'FedAvg-IID Round10'
    lines = [f'# {label}, seed42', '',
        '| Dataset | Base | Local own-domain | Shared model | Shared − Base | Local − Shared |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in (*DOMAINS, 'macro'):
        base = sum(baseline[x]['base_accuracy'] for x in DOMAINS) / 4 if d == 'macro' else baseline[d]['base_accuracy']
        local = sum(baseline[x]['local_epoch10_accuracy'] for x in DOMAINS) / 4 if d == 'macro' else baseline[d]['local_epoch10_accuracy']
        value = metrics[d + '_accuracy']
        comparisons[d] = dict(base_accuracy=base, local_accuracy=local, shared_accuracy=value,
                              shared_minus_base_pp=100 * (value - base), local_minus_shared_pp=100 * (local - value))
        lines.append(f'| {d} | {100*base:.2f}% | {100*local:.2f}% | {100*value:.2f}% | '
                     f'{100*(value-base):+.2f} pp | {100*(local-value):+.2f} pp |')
    lines += ['', 'Existing Base/Local results are reused. Same canonical initial LoRA and frozen 16000 train examples; original RACE passage and other prompts; BF16 frozen base, q_proj/v_proj r8/alpha32/dropout0.05/bias=none, bs1/accum8/constant lr1e-4/step0/drop_last=False. Only answer + EOS is supervised; existing AdamW is reset each local epoch.', '',
        ('One pooled client: 16000 examples and 2000 optimizer updates per epoch, 10 epochs.' if method == 'centralized' else
         'Four clients each contain 1000 fixed examples from each of four domains. Full participation, 4000 visits and 500 updates per client per round, ten rounds, direct A/B FedAvg weights 0.25.'), '',
        'Total: 160000 sample visits, 20000 optimizer updates. Official test is read only after the fixed tenth epoch/round and scored once by the unchanged conditional-likelihood evaluator.']
    path = ROOT / f'reports/mcq_{method}_seed42.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(output / 'result.json', metrics)
    dump(output / 'comparisons.json', comparisons)
    print('\n'.join(lines[:10]), flush=True)


def run(method):
    import torch
    from peft import get_peft_model
    from alg.base import BaseClient, BaseServer
    from alg.ftbase import FTBaseClient, FTBaseServer
    from utils.train_utils import Trainer
    from utils.model_utils import load_model, load_tokenizer, load_lora_config
    from utils.seed_utils import set_global_seed
    from utils.mcq_utils import read_jsonl
    baseline = original.verify(include_test=False)
    output = RUNS[method]
    if (output / 'completed.txt').exists():
        report(method)
        return
    clients_rows, manifest = partition(method)
    args = original.args_for_run()
    args.suffix, args.cn = str(output), len(clients_rows)
    initial_path = ROOT / settings()['initial_lora']
    assert sha(initial_path) == baseline['canonical_initial_sha256']
    protocol = dict(method=method, config=settings(), client_count=args.cn,
        canonical_initial_sha256=sha(initial_path), runner_sha256=sha(Path(__file__)),
        shared_helpers_sha256=sha(Path(original.__file__)),
        partition_sha256=sha(output / 'partition.json'), aggregation_weights=[1 / args.cn] * args.cn,
        base_frozen=True, optimizer_reset_per_client_epoch=True, test_rounds=[10])
    if (output / 'protocol.json').exists():
        assert read(output / 'protocol.json') == protocol, 'Resume protocol changed'
    else:
        dump(output / 'protocol.json', protocol)
    set_global_seed(42, device=0, deterministic=True)
    tokenizer = load_tokenizer(args)
    datasets = datasets_for(method, tokenizer, args)

    class Client(FTBaseClient):
        def __init__(self, i, dataset):
            BaseClient.__init__(self, i, args)
            self.tokenizer, self.dataset = tokenizer, {'train': dataset}
            self.lora, self.last_train_loss, self.evaluator, self.delay = {}, None, None, 1.
            self.trainer = Trainer(args, self.dataset, self)
            assert len(self.trainer.train_loader) == len(dataset) and not self.trainer.train_loader.drop_last
            self.trainer.train_loader = ProgressLoader(self.trainer.train_loader, self, output)

    class Server(FTBaseServer):
        def __init__(self, clients):
            BaseServer.__init__(self, args, clients)
            set_global_seed(42, device=0, deterministic=True)
            config = load_lora_config(args)
            assert config.bias == 'none' and set(config.target_modules) == {'q_proj', 'v_proj'}
            self.model = get_peft_model(load_model(args), config)
            self.global_lora = torch.load(initial_path, map_location='cpu', weights_only=True)
            current = original.lora_state(self.model)
            assert current.keys() == self.global_lora.keys()
            assert all(current[k].shape == v.shape and current[k].dtype == v.dtype for k, v in self.global_lora.items())
            self.model.load_state_dict(self.global_lora, strict=False)
            received = original.lora_state(self.model)
            assert all(torch.equal(received[k], v) for k, v in self.global_lora.items())
            assert all('lora_' in n for n, p in self.model.named_parameters() if p.requires_grad)
            assert next(p for n, p in self.model.named_parameters() if 'lora_' not in n).dtype == torch.bfloat16
            dump(output / 'parameters.json', dict(
                total_parameters=sum(p.numel() for p in self.model.parameters()),
                trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                trainable=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
                           for n, p in self.model.named_parameters() if p.requires_grad]))
            save_state(output / 'initial_lora.pt', self.global_lora)
            self.round, self.wall_clock_time, self.sample_rate, self.global_evaluator = 0, 0., 1., None

    clients = [Client(i, ds) for i, ds in enumerate(datasets)]
    server = Server(clients)
    checkpoint = output / 'resume.pt'
    state = (torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists() else
             dict(completed_round=0, global_lora=server.global_lora, pending={}, curves=[]))
    server.global_lora = state['global_lora']
    # Also restore the evaluated model when ten epochs are already committed
    # and the process is resuming only the final evaluation.
    server.model.load_state_dict(server.global_lora, strict=False)
    pending, curves = state['pending'], state['curves']
    dump(output / 'training_rounds.json', curves)
    exposure = output / 'exposure.jsonl'
    if exposure.exists():
        records = read_jsonl(exposure)
        records = [r for r in records if r['round'] <= state['completed_round'] or
                   (r['round'] == state['completed_round'] + 1 and r['client'] in pending)]
        temporary = exposure.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        os.replace(temporary, exposure)
    for round_number in range(state['completed_round'] + 1, 11):
        server.round, server.sampled_clients = round_number - 1, clients
        for client in clients:
            if client.id in pending:
                client.lora = pending[client.id]['lora']
                continue
            server.model.load_state_dict(server.global_lora, strict=False)
            received = original.lora_state(server.model)
            assert all(torch.equal(received[k], v) for k, v in server.global_lora.items())
            print(f'Start {method} epoch/round {round_number} client {client.id}', flush=True)
            started = time.monotonic()
            client.run(server.model)
            assert client.last_seen_count == len(client.dataset['train'])
            assert client.last_optimizer_updates == len(client.dataset['train']) // 8
            client.lora = original.lora_state(server.model)
            pending[client.id] = dict(lora=client.lora, loss=client.last_train_loss, seconds=time.monotonic() - started)
            save_state(checkpoint, dict(completed_round=round_number - 1,
                       global_lora=server.global_lora, pending=pending, curves=curves))
        weights = [len(c.dataset['train']) / 16000 for c in clients]
        assert weights == [1 / args.cn] * args.cn
        server.aggregate()
        server.global_lora = original.lora_state(server.model)
        assert all(torch.isfinite(v).all() for v in server.global_lora.values())
        curves.append(dict(round_or_epoch=round_number, samples_visited=16000,
            optimizer_updates=2000, weights=weights,
            clients=[dict(client=i, loss=pending[i]['loss'], seconds=pending[i]['seconds']) for i in range(args.cn)]))
        save_state(output / f'round_{round_number:02d}_lora.pt', server.global_lora)
        pending = {}
        save_state(checkpoint, dict(completed_round=round_number,
                   global_lora=server.global_lora, pending=pending, curves=curves))
        dump(output / 'training_rounds.json', curves)
        print(f'{method} epoch/round {round_number} committed: 16000 visits, 2000 updates.', flush=True)
    audit(method)
    # No test bytes have been read by this process up to this point.
    if not (output / 'evaluation/metrics.jsonl').exists():
        evaluate(server.model, args, tokenizer, output, baseline)
    report(method)
    (output / 'completed.txt').write_text('Ten complete epochs/rounds and one final four-domain test completed.\n', encoding='utf-8')


def suite_report():
    original.summarize()
    report('centralized')
    report('iid')
    baseline = read(ROOT / 'exp/mcq_base_local_seed42/manifest.json')['datasets']
    results = {
        'Base': {d + '_accuracy': baseline[d]['base_accuracy'] for d in DOMAINS},
        'Local own-domain': {d + '_accuracy': baseline[d]['local_epoch10_accuracy'] for d in DOMAINS},
        'Dataset-Skewed FedAvg': read(original.RUN / 'result.json'),
        'Centralized-All': read(RUNS['centralized'] / 'result.json'),
        'FedAvg-IID': read(RUNS['iid'] / 'result.json')}
    lines = ['# Four-domain MCQ comparison, seed42', '',
        '| Method | RACE | LogiQA | OpenBookQA | SciQ | Macro |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, values in results.items():
        values['macro_accuracy'] = sum(values[d + '_accuracy'] for d in DOMAINS) / 4
        lines.append('| ' + label + ' | ' + ' | '.join(f"{100*values[d + '_accuracy']:.2f}%" for d in DOMAINS)
                     + f" | {100*values['macro_accuracy']:.2f}% |")
    comparisons = {}
    lines += ['', '| Difference (pp) | RACE | LogiQA | OpenBookQA | SciQ | Macro |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, left, right in [
        ('Skewed − Base', 'Dataset-Skewed FedAvg', 'Base'),
        ('Local − Skewed', 'Local own-domain', 'Dataset-Skewed FedAvg'),
        ('Centralized − Skewed', 'Centralized-All', 'Dataset-Skewed FedAvg'),
        ('IID − Skewed', 'FedAvg-IID', 'Dataset-Skewed FedAvg'),
        ('Centralized − IID', 'Centralized-All', 'FedAvg-IID'),
        ('Local − IID', 'Local own-domain', 'FedAvg-IID')]:
        values = {d: 100 * (results[left][d + '_accuracy'] - results[right][d + '_accuracy'])
                  for d in (*DOMAINS, 'macro')}
        comparisons[label] = values
        lines.append('| ' + label + ' | ' + ' | '.join(f'{values[d]:+.2f}' for d in (*DOMAINS, 'macro')) + ' |')
    lines += ['', 'Base and Local are reused. Each trained method starts from the same canonical LoRA and uses the exact same fixed 16000 training examples for ten complete passes: 160000 sample visits and 20000 optimizer updates in total.', '',
        'Dataset-Skewed: one source per client. IID: each client has exactly 1000 examples from every source (4000 total). Centralized: one pooled 16000-example client. AdamW resets at each local epoch; thus Centralized has 10 resets and each federated run has 40 client-epoch resets. This optimizer scheduling difference is part of the fixed protocol.', '',
        'All prompts and final conditional-likelihood scoring are unchanged, including the RACE passage. A single seed and endpoint comparison does not by itself establish a causal mechanism for the gaps.']
    path = ROOT / 'reports/mcq_comparison_seed42.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(ROOT / 'exp/mcq_controls_seed42/final_results.json', dict(results=results, differences_pp=comparisons))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    # Windows may otherwise encode redirected report output as GBK, which
    # cannot represent the mathematical minus sign in the comparison tables.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('check', 'run', 'report', 'suite-report'), default='check')
    parser.add_argument('--method', choices=tuple(RUNS))
    args = parser.parse_args()
    os.chdir(ROOT)
    settings()
    if args.job == 'check':
        check()
    elif args.job == 'suite-report':
        suite_report()
    else:
        if args.method is None:
            parser.error('--method is required for run/report')
        {'run': run, 'report': report}[args.job](args.method)
