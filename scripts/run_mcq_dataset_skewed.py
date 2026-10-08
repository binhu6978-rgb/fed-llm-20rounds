"""Frozen RACE/LogiQA/OpenBookQA/SciQ FedAvg, with final-only testing.

Reuses Trainer, FTBaseClient.run and FTBaseServer.aggregate without modifying
the historical protocol files. No Base, Local, sampling or alternate jobs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
DOMAINS = ('race', 'logiqa', 'openbookqa', 'sciq')
CONFIG = ROOT / 'configs/mcq_dataset_skewed_seed42.yaml'
RUN = ROOT / 'exp/mcq_dataset_skewed_seed42'
REPORT = ROOT / 'reports/mcq_dataset_skewed_seed42.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save_state(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def settings():
    import yaml
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(domains=list(DOMAINS), seed=42, rounds=10, client_num=4,
                    participation=1.0, aggregation_weights=[0.25] * 4,
                    samples_per_client=4000, batch_size=1, gradient_accumulation=8,
                    local_epochs=1, step=0, learning_rate=1e-4,
                    lora_rank=8, lora_alpha=32, lora_dropout=0.05, eval_batch_size=8,
                    model_path='models/models--llama3.2-1B',
                    initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt',
                    dataset_dir='dataset/mcq_balanced4000',
                    output_dir='exp/mcq_dataset_skewed_seed42')
    if value != expected:
        raise ValueError('Configuration differs from the authorized fixed protocol')
    return value


def train_rows():
    from utils.mcq_utils import read_jsonl
    pool = ROOT / settings()['dataset_dir']
    rows = {}
    for domain in DOMAINS:
        manifest = read(pool / domain / 'manifest.json')
        path = pool / domain / 'train.jsonl'
        if sha(path) != manifest['output_sha256']['train.jsonl']:
            raise ValueError(f'{domain}: frozen train hash mismatch')
        current = read_jsonl(path)
        ids = [str(row['id']) for row in current]
        assert len(ids) == len(set(ids)) == 4000, domain
        assert ids == manifest['selected_ids'], domain
        assert all(row['dataset'] == domain and row['split'] == 'train' for row in current)
        rows[domain] = current
    return rows


def verify(include_test=True):
    """Preflight hashes; run() skips test bytes until all training is finished."""
    baseline = read(ROOT / 'exp/mcq_base_local_seed42/manifest.json')
    receipt = read(ROOT / 'docs/migration_files_sha256.json')['files']
    digests = dict(baseline['preserved_files_sha256'])
    digests.update(receipt)
    missing, mismatched, checked = [], [], 0
    for relative, expected in digests.items():
        normalized = relative.replace('\\', '/')
        if not include_test and (normalized.startswith('data/') or '/test' in normalized):
            continue
        path = ROOT / normalized
        if not path.is_file():
            missing.append(normalized)
        elif sha(path) != expected:
            mismatched.append(normalized)
        else:
            checked += 1
    train_rows()
    result = dict(passed=not missing and not mismatched, checked_hashes=checked,
                  missing=missing, mismatched=mismatched, test_bytes_checked=include_test)
    if include_test:
        dump(RUN / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['passed']:
        raise RuntimeError('Preflight failed; restore the listed migration artifacts')
    return baseline


def args_for_run():
    config = settings()
    return SimpleNamespace(model_path=str(ROOT / config['model_path']),
        model='llama32_1b_base', device=0, mcq=True, task_type='CAUSAL_LM',
        seed=42, deterministic=True, bs=1, grad_accum=8, lr=1e-4,
        epoch=1, step=0, cn=4, sr=1.0, rnd=10, lora_rank=8,
        lora_alpha=32, lora_dropout=0.05, suffix=str(RUN),
        eval_batch_size=8, final_only_test=True, global_test=False,
        cross_domain_qa=False, max_length=read(
            ROOT / 'dataset/mcq_balanced4000/race/manifest.json')['max_length'])


def make_datasets(tokenizer, args):
    from utils.mcq_utils import MCQTrainingDataset
    from utils.race_mcq import RaceTrainingDataset
    rows = train_rows()
    return [RaceTrainingDataset(rows[domain], tokenizer, args.max_length)
            if domain == 'race' else MCQTrainingDataset(rows[domain], tokenizer)
            for domain in DOMAINS]


class ProgressLoader:
    """Observe the unchanged loader without changing sample order or batches."""
    def __init__(self, loader, client):
        self.loader, self.client = loader, client
        self.drop_last = loader.drop_last

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        started = time.monotonic()
        for index, batch in enumerate(self.loader, 1):
            yield batch
            if index % 200 == 0:
                status = dict(round=self.client.server.round + 1,
                              client=self.client.id, domain=DOMAINS[self.client.id],
                              samples_finished=index, samples_total=len(self),
                              elapsed_seconds=time.monotonic() - started)
                dump(RUN / 'live_progress.json', status)
                print(f"Round {status['round']} {status['domain']}: "
                      f"{index}/{len(self)} samples, {status['elapsed_seconds']:.1f}s", flush=True)


def lora_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if 'lora_' in name}


def check():
    """No training or test access: prompt, supervision, aggregation and reset checks."""
    import torch
    from alg.ftbase import FTBaseServer
    from utils.model_utils import load_tokenizer
    from utils.mcq_utils import encode_prompt, continuation_ids, LETTERS
    from utils.race_mcq import encode_race, race_prompt
    args = args_for_run()
    tokenizer = load_tokenizer(args)
    datasets = make_datasets(tokenizer, args)
    for domain, dataset in zip(DOMAINS, datasets):
        assert len(dataset) == 4000
        for row, encoded in zip(dataset.rows, dataset.encoded):
            if domain == 'race':
                prefix, candidates, _ = encode_race(tokenizer, row, args.max_length)
                answer = candidates[LETTERS.index(row['correct_answer'])]
                assert race_prompt(row).startswith('Passage: ' + row['context'])
            else:
                prefix = encode_prompt(tokenizer, row)
                answer = continuation_ids(tokenizer, row, row['correct_answer'])
            target = answer + [tokenizer.eos_token_id]
            assert encoded['input_ids'] == prefix + target
            assert encoded['labels'] == [-100] * len(prefix) + target
            assert encoded['attention_mask'] == [1] * len(encoded['input_ids'])
        print(f'{domain}: 4000 unchanged prompts, answer + EOS supervision PASS', flush=True)
    # Independent numeric reference for direct A/B factor averaging.
    model = torch.nn.Linear(2, 1, bias=False)
    initial = {'weight': torch.tensor([[11., -5.]])}
    model.load_state_dict(initial)
    updates = [torch.tensor([[1., 9.]]), torch.tensor([[3., 5.]]),
               torch.tensor([[7., 1.]]), torch.tensor([[13., -3.]])]
    clients = [SimpleNamespace(id=i, dataset={'train': range(4000)},
                               lora={'weight': update}) for i, update in enumerate(updates)]
    server = SimpleNamespace(model=model, sampled_clients=clients)
    FTBaseServer.aggregate(server)
    assert torch.equal(model.weight, torch.tensor([[6., 3.]]))
    # Exercise the existing local_run reset, so one client cannot train from
    # its predecessor's update instead of the round's common global state.
    for client in clients:
        def run(current, update=client.lora['weight']):
            assert torch.equal(current.weight, initial['weight'])
            current.load_state_dict({'weight': update})
        client.run, client.training_time = run, 1.
    model.load_state_dict(initial)
    server.global_lora, server.wall_clock_time = initial, 0.
    FTBaseServer.local_run(server)
    assert torch.equal(model.weight, initial['weight'])
    result = dict(passed=True, training_samples_checked=16000,
                  test_read=False, training_performed=False,
                  equal_weight_aggregation=True, common_round_start=True,
                  runner_sha256=sha(Path(__file__)))
    dump(RUN / 'implementation_checks.json', result)
    print('Existing FedAvg aggregation and client reset: PASS', flush=True)


def run():
    import torch
    from peft import get_peft_model
    from alg.base import BaseClient, BaseServer
    from alg.ftbase import FTBaseClient, FTBaseServer
    from utils.train_utils import Trainer
    from utils.model_utils import load_model, load_tokenizer, load_lora_config
    from utils.seed_utils import set_global_seed
    baseline = verify(include_test=False)
    args = args_for_run()
    if (RUN / 'completed.txt').exists():
        summarize()
        return
    initial_path = ROOT / settings()['initial_lora']
    assert sha(initial_path) == baseline['canonical_initial_sha256']
    protocol = dict(config=settings(), canonical_initial_sha256=sha(initial_path),
                    runner_sha256=sha(Path(__file__)), base_frozen=True,
                    optimizer='AdamW reset for each client epoch',
                    aggregation='FTBaseServer.aggregate: direct LoRA A/B average',
                    test_rounds=[10], client_ids=dict(enumerate(DOMAINS)),
                    train_sha256={d: sha(ROOT / settings()['dataset_dir'] / d / 'train.jsonl')
                                  for d in DOMAINS})
    # JSON object keys are strings on disk.
    protocol = json.loads(json.dumps(protocol))
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol, 'Resume protocol changed'
    else:
        dump(RUN / 'protocol.json', protocol)
    set_global_seed(42, device=0, deterministic=True)
    tokenizer = load_tokenizer(args)
    datasets = make_datasets(tokenizer, args)

    class Client(FTBaseClient):
        def __init__(self, client_id, dataset):
            BaseClient.__init__(self, client_id, args)
            self.tokenizer, self.dataset = tokenizer, {'train': dataset}
            self.lora, self.last_train_loss, self.evaluator, self.delay = {}, None, None, 1.
            self.trainer = Trainer(args, self.dataset, self)
            assert len(self.trainer.train_loader) == 4000
            assert not self.trainer.train_loader.drop_last
            self.trainer.train_loader = ProgressLoader(self.trainer.train_loader, self)

    class Server(FTBaseServer):
        def __init__(self, clients):
            BaseServer.__init__(self, args, clients)
            set_global_seed(42, device=0, deterministic=True)
            lora_config = load_lora_config(args)
            assert lora_config.bias == 'none'
            assert set(lora_config.target_modules) == {'q_proj', 'v_proj'}
            self.model = get_peft_model(load_model(args), lora_config)
            self.global_lora = torch.load(initial_path, map_location='cpu', weights_only=True)
            current = lora_state(self.model)
            assert current.keys() == self.global_lora.keys()
            assert all(current[k].shape == v.shape and current[k].dtype == v.dtype
                       for k, v in self.global_lora.items())
            self.model.load_state_dict(self.global_lora, strict=False)
            received = lora_state(self.model)
            assert all(torch.equal(received[k], v) for k, v in self.global_lora.items())
            assert all('lora_' in name for name, p in self.model.named_parameters() if p.requires_grad)
            assert next(p for n, p in self.model.named_parameters() if 'lora_' not in n).dtype == torch.bfloat16
            dump(RUN / 'parameters.json', dict(
                total_parameters=sum(p.numel() for p in self.model.parameters()),
                trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                trainable=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
                           for n, p in self.model.named_parameters() if p.requires_grad]))
            save_state(RUN / 'initial_lora.pt', self.global_lora)
            self.sample_rate, self.wall_clock_time, self.round = 1.0, 0., 0
            self.global_evaluator = None

    clients = [Client(i, dataset) for i, dataset in enumerate(datasets)]
    server = Server(clients)
    checkpoint = RUN / 'resume.pt'
    state = (torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists()
             else dict(completed_round=0, global_lora=server.global_lora, pending={}, curves=[]))
    server.global_lora = state['global_lora']
    server.model.load_state_dict(server.global_lora, strict=False)
    pending, curves = state['pending'], state['curves']
    # The checkpoint is authoritative if a crash occurred before the JSON update.
    dump(RUN / 'training_rounds.json', curves)
    # Roll back accounting from a crash between exposure append and checkpoint commit.
    from utils.mcq_utils import read_jsonl
    exposure = RUN / 'exposure.jsonl'
    if exposure.exists():
        records = read_jsonl(exposure)
        committed = [row for row in records if row['round'] <= state['completed_round'] or
                     (row['round'] == state['completed_round'] + 1 and row['client'] in pending)]
        temporary = exposure.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r) + '\n' for r in committed), encoding='utf-8')
        os.replace(temporary, exposure)
    for round_number in range(state['completed_round'] + 1, 11):
        server.round = round_number - 1
        server.sampled_clients = clients  # Full participation, stable C0..C3 order.
        for client in clients:
            if client.id in pending:
                client.lora = pending[client.id]['lora']
                continue
            server.model.load_state_dict(server.global_lora, strict=False)
            received = lora_state(server.model)
            assert all(torch.equal(received[k], v) for k, v in server.global_lora.items())
            print(f'Start Round {round_number} Client {client.id} {DOMAINS[client.id]}', flush=True)
            started = time.monotonic()
            client.run(server.model)
            assert client.last_seen_count == 4000 and client.last_optimizer_updates == 500
            client.lora = lora_state(server.model)
            pending[client.id] = dict(lora=client.lora, loss=client.last_train_loss,
                                     seconds=time.monotonic() - started)
            save_state(checkpoint, dict(completed_round=round_number - 1,
                       global_lora=server.global_lora, pending=pending, curves=curves))
        assert len(pending) == 4
        # Use the repository's existing sample-weighted A/B aggregation unchanged.
        assert [len(c.dataset['train']) / 16000 for c in clients] == [0.25] * 4
        server.aggregate()
        server.global_lora = lora_state(server.model)
        assert all(torch.isfinite(v).all() for v in server.global_lora.values())
        curves.append(dict(round=round_number, samples_visited=16000,
                           optimizer_updates=2000, weights=[0.25] * 4,
                           clients=[dict(client=i, domain=DOMAINS[i],
                                         loss=pending[i]['loss'], seconds=pending[i]['seconds'])
                                    for i in range(4)]))
        save_state(RUN / f'round_{round_number:02d}_lora.pt', server.global_lora)
        pending = {}
        save_state(checkpoint, dict(completed_round=round_number,
                   global_lora=server.global_lora, pending=pending, curves=curves))
        dump(RUN / 'training_rounds.json', curves)
        print(f'Round {round_number} committed: 16000 visits, 2000 updates.', flush=True)
    audit_exposure()
    # First test-file access in this process occurs after Round 10 is committed.
    metrics_path = RUN / 'evaluation/metrics.jsonl'
    if metrics_path.exists():
        metrics = read_jsonl(metrics_path)
        assert len(metrics) == 1 and metrics[0]['round'] == 10
        result = metrics[0]
    else:
        from utils.mcq_eval import MCQEvaluator
        from utils.mcq_utils import encode_prompt, continuation_ids, LETTERS
        from utils.race_mcq import encode_race

        class Evaluator(MCQEvaluator):
            def __init__(self):
                self.args, self.tokenizer, self.domains = args, tokenizer, DOMAINS
                self.output = RUN / 'evaluation'
                self.output.mkdir(parents=True, exist_ok=True)
                self.rows, self.encoded = {}, {}
                for domain in DOMAINS:
                    folder = ROOT / settings()['dataset_dir'] / domain
                    assert sha(folder / 'test.jsonl') == read(folder / 'manifest.json')['output_sha256']['test.jsonl']
                    rows = read_jsonl(folder / 'test.jsonl')
                    assert len(rows) == baseline['datasets'][domain]['official_test_count']
                    self.rows[domain] = rows
                    self.encoded[domain] = [encode_race(tokenizer, row, args.max_length)[:2]
                        if domain == 'race' else (encode_prompt(tokenizer, row),
                            [continuation_ids(tokenizer, row, letter) for letter in LETTERS])
                        for row in rows]
        result = Evaluator().evaluate(server.model, 10)
    dump(RUN / 'result.json', result)
    summarize()
    (RUN / 'completed.txt').write_text('10 rounds and one final four-domain evaluation completed.\n', encoding='utf-8')


def audit_exposure():
    from utils.mcq_utils import read_jsonl
    expected = {d: {d + ':' + str(r['id']) for r in rows} for d, rows in train_rows().items()}
    records = read_jsonl(RUN / 'exposure.jsonl')
    assert len(records) == 40
    assert {(r['round'], r['client']) for r in records} == {(r, c) for r in range(1, 11) for c in range(4)}
    for record in records:
        assert record['samples'] == 4000 and record['optimizer_updates'] == 500
        assert len(record['ids']) == len(set(record['ids'])) == 4000
        assert set(record['ids']) == expected[DOMAINS[record['client']]]
    dump(RUN / 'exposure_audit.json', dict(passed=True, client_epochs=40,
         sample_visits=160000, optimizer_updates=20000, fixed_train_ids=True))


def summarize():
    import csv
    from utils.mcq_utils import read_jsonl
    audit_exposure()
    metrics = read_jsonl(RUN / 'evaluation/metrics.jsonl')
    assert len(metrics) == 1 and metrics[0]['round'] == 10
    fed = metrics[0]
    baseline = read(ROOT / 'exp/mcq_base_local_seed42/manifest.json')['datasets']
    with (RUN / 'evaluation/round_10_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    assert {row['domain'] for row in predictions} == set(DOMAINS)
    for domain in DOMAINS:
        current = [row for row in predictions if row['domain'] == domain]
        assert len(current) == len({row['id'] for row in current}) == baseline[domain]['official_test_count']
        assert all(int(row['correct']) == int(row['prediction'] == row['gold']) for row in current)
        assert abs(sum(int(row['correct']) for row in current) / len(current) - fed[domain + '_accuracy']) < 1e-12
    assert abs(sum(fed[d + '_accuracy'] for d in DOMAINS) / 4 - fed['macro_accuracy']) < 1e-12
    curves = read(RUN / 'training_rounds.json')
    assert len(curves) == 10
    assert all(row['round'] == i and row['samples_visited'] == 16000
               and row['optimizer_updates'] == 2000 and row['weights'] == [0.25] * 4
               for i, row in enumerate(curves, 1))
    comparisons = {}
    lines = ['# Dataset-Skewed FedAvg, seed42', '',
             '| Dataset | Base | Local own-domain | FedAvg Round10 | FedAvg − Base | Local − FedAvg |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for domain in (*DOMAINS, 'macro'):
        base = (sum(baseline[d]['base_accuracy'] for d in DOMAINS) / 4 if domain == 'macro'
                else baseline[domain]['base_accuracy'])
        local = (sum(baseline[d]['local_epoch10_accuracy'] for d in DOMAINS) / 4 if domain == 'macro'
                 else baseline[domain]['local_epoch10_accuracy'])
        accuracy = fed[domain + '_accuracy']
        comparisons[domain] = dict(base_accuracy=base, local_accuracy=local,
            fedavg_accuracy=accuracy, fedavg_minus_base_pp=100 * (accuracy - base),
            local_minus_fedavg_pp=100 * (local - accuracy))
        lines.append(f'| {domain} | {100*base:.2f}% | {100*local:.2f}% | {100*accuracy:.2f}% | '
                     f'{100*(accuracy-base):+.2f} pp | {100*(local-accuracy):+.2f} pp |')
    lines += ['', 'Base and Local are reused from preserved results. Local represents four independent own-domain specialists; FedAvg is one shared adapter.', '',
              'Four clients each reuse their exact frozen 4000 train rows, with ten complete local epochs in total: 40000 visits and 5000 optimizer updates per client. All clients participate every round. Equal 0.25 weights directly average LoRA A/B tensors using the existing implementation.', '',
              'Canonical common initial LoRA; Llama-3.2-1B BF16 frozen base; q_proj/v_proj r8, alpha32, dropout0.05, bias=none; bs1, accumulation8, constant lr1e-4, step0, drop_last=False. Existing MCQ Trainer rebuilds AdamW each local epoch. RACE preserves passage; other prompts are unchanged. Only answer continuation + EOS is supervised.', '',
              'Only Round10 is tested, using unchanged A/B/C/D conditional likelihood. No validation selection. Positive Local − FedAvg measures the specialist-to-shared accuracy gap; this experiment alone does not identify its mechanism.', '',
              'Audit artifacts: `exp/mcq_dataset_skewed_seed42/` (protocol, exposure audit, all round adapters, final predictions and comparisons).']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(RUN / 'comparisons.json', comparisons)
    print('\n'.join(lines[:10]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('verify', 'check', 'run', 'report'), default='run')
    selected = parser.parse_args().job
    os.chdir(ROOT)
    settings()
    if selected == 'run':
        log_path = ROOT / 'logs/mcq_dataset_skewed_seed42/run.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)

        class Tee:
            def __init__(self, stream, log):
                self.stream, self.log = stream, log

            def write(self, value):
                self.stream.write(value)
                self.log.write(value)
                self.log.flush()
                return len(value)

            def flush(self):
                self.stream.flush()
                self.log.flush()

        with log_path.open('a', encoding='utf-8') as log:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
            try:
                run()
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
    else:
        {'verify': verify, 'check': check, 'report': summarize}[selected]()
