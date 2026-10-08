"""CosmosQA Base -> Independent Local10 -> report; no federated experiment."""
import argparse
import copy
import ctypes
import json
import math
import msvcrt
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from scripts import cosmosqa_data as data
from scripts import run_three_dataset_lora as shared
from utils import cosmosqa_mcq as encoding
from utils import mcq_utils

MASTER = data.MASTER
REPORT = ROOT / 'reports/cosmosqa_sanity_seed42.md'


def configure():
    shared.data, shared.MASTER = data, MASTER
    shared.DOMAINS, shared.NAMES = data.DOMAINS, data.NAMES
    shared.OUTPUTS = {'base': MASTER / 'base', 'clientlocal': MASTER / 'clientlocal'}


class Experiment(shared.Experiment):
    def __init__(self):
        configure()
        # Original MCQTrainingDataset resolves these globals when encoding rows.
        # The evaluator scores already-encoded inputs with its unchanged implementation.
        with patch.object(mcq_utils, 'encode_prompt', encoding.encode_prompt), \
             patch.object(mcq_utils, 'continuation_ids', encoding.continuation_ids):
            super().__init__(smoke=False)
        self.args.cn = 1
        self.rounds = 0
        self.outputs = dict(shared.OUTPUTS)
        self.protocol = {k: v for k, v in self.protocol.items()
                         if k not in ('rounds', 'aggregation', 'aggregation_sha256', 'weights')}
        self.protocol.update(scope='CosmosQA Base and Independent Local only', federated_training=False,
            cosmos_runner_sha256=data.sha(Path(__file__)),
            cosmos_encoder_sha256=data.sha(Path(encoding.__file__)),
            mcq_evaluator_sha256=data.sha(ROOT / 'utils/mcq_eval.py'),
            model_utils_sha256=data.sha(ROOT / 'utils/model_utils.py'),
            passage_encoder_sha256=data.sha(ROOT / 'utils/race_mcq.py'),
            config_sha256=data.sha(data.CONFIG), canonical_initial_tensor_sha256=shared.tensor_hash(self.initial),
            client_id=0, epoch_seed_rule='client_round_seed(42, 0, epoch-1)',
            test_epochs=[0, 10], checkpoint_selection='fixed epoch10; no selection',
            optimizer=dict(name='AdamW', lr=1e-4, betas=[0.9, 0.999], eps=1e-8,
                           weight_decay=0.01, reset_each_epoch=True))
        assert self.counts == {'cosmosqa': 4000} and len(self.test_rows['cosmosqa']) == 1500
        stats = data.read(ROOT / data.config()['prepared_dir'] / 'sequence_statistics.json')
        encoded = self.datasets['cosmosqa'].encoded
        assert sum(len(item['input_ids']) for item in encoded) == stats['train']['token_total']
        assert sum(sum(x != -100 for x in item['labels']) for item in encoded) == stats['train']['supervised_token_total']
        for row, item in zip(self.train_rows['cosmosqa'], encoded):
            prefix = encoding.encode_prompt(self.tokenizer, row)
            target = encoding.continuation_ids(self.tokenizer, row, row['correct_answer']) + [self.tokenizer.eos_token_id]
            assert item['input_ids'] == prefix + target
            assert item['labels'] == [-100] * len(prefix) + target
            assert item['native_key'] == 'cosmosqa:' + row['id']
        encoding.encoded.cache_clear()

    def local(self):
        import torch
        output = self.init_output('clientlocal')
        current = output / 'client_cosmosqa'
        current.mkdir(parents=True, exist_ok=True)
        resume = current / 'resume.pt'
        state = (torch.load(resume, map_location='cpu', weights_only=True) if resume.exists()
                 else dict(epoch=0, lora=self.initial, curves=[],
                           initial_tensor_sha256=shared.tensor_hash(self.initial)))
        assert state['initial_tensor_sha256'] == shared.tensor_hash(self.initial)
        assert 0 <= state['epoch'] <= 10 and len(state['curves']) == state['epoch']
        self.load(state['lora'])
        self.trim_exposure(current / 'exposure.jsonl', state['epoch'])
        server = SimpleNamespace(round=0)
        client = self.client('cosmosqa', current, server, 'clientlocal')
        for epoch in range(state['epoch'] + 1, 11):
            server.round = epoch - 1
            before = shared.tensor_hash(shared.adapter(self.model))
            if epoch == 1:
                assert before == state['initial_tensor_sha256']
            shared.status('clientlocal', dataset='cosmosqa', epoch=epoch)
            value, record = self.train(client, 'cosmosqa')
            state['curves'].append(dict(epoch=epoch, **record, before_lora_tensor_sha256=before,
                                       after_lora_tensor_sha256=shared.tensor_hash(value)))
            state.update(epoch=epoch, lora=value)
            shared.save_state(resume, state)
            data.dump(current / 'training_epochs.json', state['curves'])
        final = current / 'final_lora.pt'
        shared.save_state(final, state['lora'])
        self.load(state['lora'])
        shared.status('local_final_evaluation', epoch=10)
        result = self.score(current, ('cosmosqa',), 10, final)
        data.dump(output / 'result.json', result)
        (current / 'completed.txt').write_text('Independent Local epoch10 completed.\n', encoding='utf-8')
        (output / 'completed.txt').write_text('One CosmosQA specialist; fixed epoch10.\n', encoding='utf-8')
        return result

    def fedavg(self):
        raise RuntimeError('Federated training is outside the authorized CosmosQA sanity-check scope')


def audit(experiment):
    import torch
    inputs = data.verify(experiment.manifest)
    current = experiment.outputs['clientlocal'] / 'client_cosmosqa'
    exposure = data.rows(current / 'exposure.jsonl')
    curves = data.read(current / 'training_epochs.json')
    assert len(exposure) == len(curves) == 10
    expected_ids = {'cosmosqa:' + row['id'] for row in experiment.train_rows['cosmosqa']}
    stats = data.read(ROOT / data.config()['prepared_dir'] / 'sequence_statistics.json')
    for epoch, (record, curve) in enumerate(zip(exposure, curves), 1):
        assert record['round'] == curve['epoch'] == epoch and record['client'] == 0
        assert record['samples'] == curve['samples'] == len(record['ids']) == 4000
        assert len(set(record['ids'])) == 4000 and set(record['ids']) == expected_ids
        assert record['optimizer_updates'] == curve['optimizer_updates'] == 500
        assert record['supervised_tokens'] == stats['train']['supervised_token_total']
        assert math.isfinite(curve['train_loss'])
        expected_before = (experiment.protocol['canonical_initial_tensor_sha256'] if epoch == 1
                           else curves[epoch - 2]['after_lora_tensor_sha256'])
        assert curve['before_lora_tensor_sha256'] == expected_before
    final = torch.load(current / 'final_lora.pt', map_location='cpu', weights_only=True)
    assert shared.tensor_hash(final) == curves[-1]['after_lora_tensor_sha256']
    base_receipt = data.read(experiment.outputs['base'] / 'evaluation_provenance.json')
    local_receipt = data.read(current / 'evaluation_provenance.json')
    assert base_receipt['lora_tensor_sha256'] == experiment.protocol['canonical_initial_tensor_sha256']
    assert local_receipt['lora_tensor_sha256'] == shared.tensor_hash(final)
    assert base_receipt['domains'] == local_receipt['domains'] == ['cosmosqa']
    assert [r['round'] for r in data.rows(experiment.outputs['base'] / 'evaluation/metrics.jsonl')] == [0]
    assert [r['round'] for r in data.rows(current / 'evaluation/metrics.jsonl')] == [10]
    assert not (MASTER / 'fedavg').exists()
    result = dict(passed=True, input_integrity=inputs, independent_from_canonical_initial=True,
        base_frozen=True, trainable_only_q_proj_v_proj_lora=True, original_trainer_and_evaluator=True,
        original_passage_prompt=True, epochs=10, sample_visits=40000, optimizer_updates=5000,
        samples_per_epoch=4000, updates_per_epoch=500, optimizer_reset_each_epoch=True,
        final_checkpoint_epoch=10, accuracy_evaluation_epochs=[0, 10],
        no_checkpoint_selection=True, no_resampling=True, no_hyperparameter_tuning=True,
        no_federated_or_cross_domain_experiment=True,
        canonical_initial_lora_sha256=experiment.protocol['initial_lora_sha256'],
        canonical_initial_tensor_sha256=base_receipt['lora_tensor_sha256'],
        final_local_tensor_sha256=local_receipt['lora_tensor_sha256'])
    data.dump(MASTER / 'protocol_audit.json', result)
    return result


def report(experiment):
    audit_result = audit(experiment)
    base = data.read(experiment.outputs['base'] / 'result.json')['cosmosqa_accuracy']
    local_dir = experiment.outputs['clientlocal'] / 'client_cosmosqa'
    local = data.read(local_dir / 'result.json')['cosmosqa_accuracy']
    stats = data.read(ROOT / data.config()['prepared_dir'] / 'sequence_statistics.json')
    source = experiment.manifest['audit']
    curves = data.read(local_dir / 'training_epochs.json')
    lines = ['# CosmosQA Independent Local sanity check (seed42)', '',
        f'Base accuracy = {base * 100:.2f}%', f'Local epoch-10 accuracy = {local * 100:.2f}%',
        f'Local − Base = {(local - base) * 100:+.2f} pp', '',
        '## Dataset preparation', '', '| Actual source file | Format | Split | Rows | Valid labels |',
        '| --- | --- | --- | ---: | ---: |']
    for item in source['inspected_files']:
        lines.append(f"| {item['file']} | {item['format']} | {item['split']} | {item['rows']} | {item['valid_accuracy_labels']} |")
    lines += ['', f"Train: 4000 official train rows. Test: 1500 official {source['test_official_split']} rows.",
        f"Official test fully labeled: {source['official_test_usable']}. Unused validation rows are not evaluated or trained.",
        'One seed42 uniform sample after quality-only cleaning; source order restored. Original IDs, raw file, '
        'split, file row and source index are saved. No performance-dependent selection or resampling.', '',
        '| Split | N | A/B/C/D | Total tokens | Mean | Median | P90 | Max |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for split in ('train', 'test'):
        value = stats[split]
        distribution = '/'.join(str(value['label_distribution'][l]) for l in 'ABCD')
        lines.append(f"| {split} | {value['samples']} | {distribution} | {value['token_total']} | "
            f"{value['mean_input_length']:.2f} | {value['median_input_length']:.2f} | "
            f"{value['p90_input_length']:.2f} | {value['max_input_length']} |")
    lines += ['', 'Token lengths include BOS + the existing passage prompt + gold continuation + EOS; no padding.',
        f"Quality filtering: {json.dumps(source['quality'], ensure_ascii=False)}.",
        f"Train rows removed for selected-test QA/native-ID overlap: {source['train_overlap_removed']}.",
        f"Full cleaned source QA overlap: {source['full_clean_qa_overlap']}; selected full-QA overlap: 0; selected native-ID overlap: 0.",
        f"Selected shared context strings: {source['selected_shared_contexts']} (reported separately from full-QA identity).",
        f"Passage-truncated train/test samples: {stats['train']['passage_truncated_samples']}/{stats['test']['passage_truncated_samples']}.",
        '', '## Protocol and loss', '',
        'The existing RACE passage template encodes context + question + original four options. '
        'The original MCQEvaluator scores A/B/C/D conditional likelihood and saves all four log probabilities. '
        'No new prompt text or scoring rule was introduced.',
        'Canonical frozen BF16 Llama-3.2-1B and initial LoRA; q_proj/v_proj r8, alpha32, dropout0.05, bias=none; '
        'constant lr1e-4, batch1, accumulation8, original AdamW reset each full epoch. '
        'Client ID0 with original client_round_seed(42,0,epoch-1). Ten full epochs, 40000 visits, 5000 updates. '
        'Only epoch10 is the formal Local endpoint. No intermediate accuracy or checkpoint selection.', '',
        '| Epoch | Mean training loss | Samples | Updates |', '| ---: | ---: | ---: | ---: |']
    for curve in curves:
        lines.append(f"| {curve['epoch']} | {curve['train_loss']:.6f} | {curve['samples']} | {curve['optimizer_updates']} |")
    lines += ['', f"Loss change: {curves[0]['train_loss']:.6f} → {curves[-1]['train_loss']:.6f}.",
        f"Protocol audit passed: {audit_result['passed']}.", '', '## Sanity-check observation', '',
        f"On this fixed 1500-row test, epoch10 Local changes accuracy by {(local-base)*100:+.2f} pp from Base. "
        + ('This is an observed learnable gain under the unchanged protocol.' if local > base else
           'No positive gain was observed; preprocessing, scoring, sample exposure and training audits passed.'),
        'This is one fixed-seed descriptive comparison. No causal explanation, further training, tuning, '
        'FedAvg, replacement experiment or cross-domain analysis was performed.', '', '## Saved artifacts', '',
        f"- Prepared rows/manifest/statistics: `{data.config()['prepared_dir']}/`.",
        f"- Base metrics/receipt: `{experiment.outputs['base'].relative_to(ROOT)}/`.",
        f"- Base predictions and choice log probabilities: `{(experiment.outputs['base'] / 'evaluation/round_00_predictions.csv').relative_to(ROOT)}`.",
        f"- Local epoch10 checkpoint: `{(local_dir / 'final_lora.pt').relative_to(ROOT)}`.",
        f"- Local metrics/receipt/loss: `{local_dir.relative_to(ROOT)}/`.",
        f"- Local predictions and choice log probabilities: `{(local_dir / 'evaluation/round_10_predictions.csv').relative_to(ROOT)}`.",
        f"- Data/quality/input/protocol audits: `{MASTER.relative_to(ROOT)}/`."]
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    data.dump(MASTER / 'final_results.json', dict(base_accuracy=base, local_epoch10_accuracy=local,
        local_minus_base_pp=(local - base) * 100, protocol_audit_passed=True, report=str(REPORT)))
    return REPORT


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('prepare', 'check', 'all', 'report'), default='all')
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    data.SOURCE_OVERRIDE = args.source_dir
    configure()
    MASTER.mkdir(parents=True, exist_ok=True)
    with (MASTER / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('CosmosQA sanity check is already running; duplicate launch refused.')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            if args.job == 'check':
                print(json.dumps(data.verify()), flush=True)
                return
            data.prepare()
            if args.job == 'prepare':
                shared.status('data_ready')
                return
            experiment = Experiment()
            if args.job == 'all':
                experiment.base()
                experiment.local()
            shared.status('reporting')
            path = report(experiment)
            shared.status('complete', report=str(path))
            (MASTER / 'completed.txt').write_text('CosmosQA Base and Independent Local10 complete; STOP.\n', encoding='utf-8')
            print(json.dumps(data.read(MASTER / 'final_results.json')), flush=True)
        except BaseException as error:
            shared.status('failed', error=str(error), traceback=traceback.format_exc())
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
