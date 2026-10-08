"""Retrospective own-domain scoring of all saved baseline client models; no training."""
import csv
import ctypes
import msvcrt
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_fedlora_baseline as baseline

data, shared = baseline.data, baseline.shared
DOMAINS = data.DOMAINS
OUTPUT = data.MASTER / 'local_before_upload'
REPORT = ROOT / 'reports/fedlora_baselines_local_before_upload_seed42.md'


def local_state(folder, round_number, domain, record):
    """A frozen local backbone is the PREVIOUS global backbone, never this round's aggregate."""
    import torch
    start_path = (folder / 'initial_complete_state.pt' if round_number == 1 else
                  folder / f'round_{round_number-1:02d}/global/complete_state.pt')
    raw_path = folder / f'round_{round_number:02d}/client_{domain}/raw_lora.pt'
    upload_path = raw_path.with_name('upload_lora.pt')
    start = torch.load(start_path, map_location='cpu', weights_only=True)
    client = record['clients'][domain]
    assert baseline.function_hash(start) == client['start_function_hash']
    raw = torch.load(raw_path, map_location='cpu', weights_only=True)
    assert shared.tensor_hash(raw) == client['raw_hash']
    assert shared.tensor_hash(torch.load(upload_path, map_location='cpu', weights_only=True)) == client['upload_hash']
    assert client['budget']['samples'] == 4000 and client['budget']['optimizer_updates'] == 500
    value = dict(start, lora=raw)
    identity = dict(round=round_number, domain=domain, model_kind='trained_local_before_alignment',
                    local_function_state_sha256=baseline.function_hash(value),
                    raw_checkpoint=str(raw_path.relative_to(ROOT)), raw_checkpoint_sha256=data.sha(raw_path),
                    start_complete_checkpoint=str(start_path.relative_to(ROOT)),
                    start_complete_checkpoint_sha256=data.sha(start_path),
                    start_function_hash=client['start_function_hash'], start_base_hash=client['start_base_hash'],
                    raw_hash=client['raw_hash'], upload_hash=client['upload_hash'])
    return value, raw_path, identity


def status(stage, **details):
    data.dump(OUTPUT / 'status.json', dict(stage=stage, pid=os.getpid(), updated_at=shared.now(), **details))


def verify_source():
    snapshot = data.read(data.MASTER / 'code_state.json')
    for path, digest in snapshot['file_sha256'].items():
        assert data.sha(ROOT / path) == digest, 'Original input or code changed: ' + path
    hashes = dict(snapshot['file_sha256'])
    hashes[str(Path(__file__).relative_to(ROOT))] = data.sha(Path(__file__))
    for method in data.METHODS:
        folder = data.output(method)
        assert (folder / 'completed.txt').exists() and data.read(folder / 'protocol_audit.json')['passed']
        paths = [folder / 'initial_complete_state.pt', folder / 'round_trajectory.json', folder / 'protocol.json']
        paths += list(folder.glob('round_*/global/complete_state.pt'))
        paths += list(folder.glob('round_*/client_*/raw_lora.pt'))
        paths += list(folder.glob('round_*/client_*/upload_lora.pt'))
        assert len(paths) == 113
        hashes.update({str(p.relative_to(ROOT)): data.sha(p) for p in paths})
    path = OUTPUT / 'input_snapshot.json'
    if path.exists():
        assert data.read(path)['file_sha256'] == hashes, 'Incompatible evaluation resume'
    else:
        data.dump(path, dict(file_sha256=hashes, retraining=False, own_domain_only=True))


def summarize():
    records = []
    for method in data.METHODS:
        for t in range(1, 11):
            global_record = data.read(data.output(method) / 'round_trajectory.json')[t-1]
            for domain in DOMAINS:
                directory = OUTPUT / method / f'round_{t:02d}' / domain
                if not (directory / 'result.json').exists():
                    continue
                metric = data.read(directory / 'result.json')
                identity = data.read(directory / 'full_function_identity.json')
                records.append(dict(method=method, round=t, domain=domain,
                                    accuracy=metric[domain+'_accuracy'],
                                    global_accuracy=global_record['metrics'][domain+'_accuracy'],
                                    local_minus_global_pp=100*(metric[domain+'_accuracy']-global_record['metrics'][domain+'_accuracy']),
                                    train_loss=global_record['clients'][domain]['budget']['train_loss'],
                                    test_samples=dict(cosmosqa=1500, openbookqa=1000, sciq=1998, hellaswag=1500, race=1500)[domain],
                                    prediction_file=str((directory / 'evaluation' / f'round_{t:02d}_predictions.csv').relative_to(ROOT)),
                                    **{k:identity[k] for k in ('raw_checkpoint', 'start_complete_checkpoint', 'local_function_state_sha256')}))
    data.dump(OUTPUT / 'client_metrics.json', records)
    if records:
        with (OUTPUT / 'client_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    lines = ['# Federated LoRA 每轮上传前本地测试准确率', '',
             '仅补评已保存的模型，无训练、无抽样、无调参。每个客户端在自己的完整原测试集上评测，复用原 prompt 和 choice-likelihood evaluator。', '',
             '本地模型使用本轮开始时的完整 global backbone + 本轮训练结束的 raw_lora；FedRot 使用旋转对齐前的实际本地模型。', '',
             'Local own-domain mean 是五个不同本地模型的各自域准确率等权均值；Global Macro 是同一个聚合模型在五域的均值。前者不是单一共享模型的 Macro，也不是 Independent Local。', '',
             f'进度：{len(records)}/200 个客户端评测。准确率单位为 %，差距单位为 pp。']
    rounds = []
    for method in data.METHODS:
        lines += ['', '## '+data.NAMES_METHOD[method], '',
                  '| Round | '+' | '.join(data.NAMES[d] for d in DOMAINS)+' | Local own-domain mean | Global Macro | Local − Global |',
                  '|---:|'+'---:|'*8]
        for t in range(1, 11):
            rows = [r for r in records if r['method']==method and r['round']==t]
            by_domain = {r['domain']:r for r in rows}
            if len(rows) != 5:
                values = ['%.2f'%(by_domain[d]['accuracy']*100) if d in by_domain else '待评' for d in DOMAINS]
                lines.append('| '+str(t)+' | '+' | '.join(values)+' | — | — | — |')
                continue
            mean = sum(r['accuracy'] for r in rows)/5
            global_macro = sum(r['global_accuracy'] for r in rows)/5
            value = dict(method=method, round=t, local_own_domain_mean=mean, global_macro=global_macro,
                         local_minus_global_pp=100*(mean-global_macro),
                         **{d+'_accuracy':by_domain[d]['accuracy'] for d in DOMAINS})
            rounds.append(value)
            lines.append('| '+str(t)+' | '+' | '.join('%.2f'%(by_domain[d]['accuracy']*100) for d in DOMAINS)+
                         ' | %.2f | %.2f | %+.2f |'%(mean*100, global_macro*100, value['local_minus_global_pp']))
    lines += ['', '逐客户端 loss、准确率、同域聚合差距、预测路径及完整模型来源见 `outputs/fedlora_baselines_5source_seed42/local_before_upload/client_metrics.csv`。',
              '', '本地测试准确率不是训练集 accuracy。FedEx/FedMomentum 原实现的残差缩放差异沿用已完成的训练，补评不修改它们。']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    data.dump(OUTPUT / 'round_metrics.json', rounds)
    if rounds:
        with (OUTPUT / 'round_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rounds[0])); writer.writeheader(); writer.writerows(rounds)
    return len(records)


def run():
    verify_source()
    summarize()
    for method in data.METHODS:
        folder = data.output(method)
        ex = baseline.Experiment(method)
        trajectory = data.read(folder / 'round_trajectory.json')
        assert len(trajectory)==10
        for record in trajectory:
            t = record['round']
            for domain in DOMAINS:
                status('evaluating', method=method, round=t, domain=domain, completed_evaluations=summarize())
                value, checkpoint, identity = local_state(folder, t, domain, record)
                ex.restore(value)
                assert ex.base_hash()==identity['start_base_hash']
                directory = OUTPUT / method / f'round_{t:02d}' / domain
                directory.mkdir(parents=True, exist_ok=True)
                identity.update(method=method, canonical_model=ex.protocol['config']['model_path'],
                                prepared_manifest_sha256=ex.protocol['prepared_manifest_sha256'],
                                evaluator_sha256=ex.protocol['choice_evaluator_sha256'],
                                mutated_backbone_included=method in baseline.MUTATORS)
                path = directory / 'full_function_identity.json'
                if path.exists():
                    assert data.read(path)==identity, 'Cached local evaluation has different complete model'
                else:
                    data.dump(path, identity)
                scores = baseline.original.Experiment.score(ex, directory, [domain], t, checkpoint)
                receipt = data.read(directory / 'evaluation_provenance.json')
                receipt.update(identity)
                data.dump(directory / 'evaluation_provenance.json', receipt)
                assert baseline.function_hash(ex.capture())==identity['local_function_state_sha256']
                print('LOCAL',method,t,domain,scores[domain+'_accuracy'],flush=True)
                summarize()
        del ex
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
    assert summarize()==200
    verify_source()
    data.dump(OUTPUT / 'protocol_audit.json', dict(passed=True, evaluations=200, full_test_sample_visits=299920,
              own_domain_only=True, original_evaluator=True, no_training=True,
              previous_global_backbone_restored=True, raw_before_alignment=True,
              source_checkpoint_hashes_verified=True, predictions_validated_by_original_score=True))
    (OUTPUT / 'completed.txt').write_text('All 200 saved local own-domain evaluations complete.\n',encoding='utf-8')
    status('complete', completed_evaluations=200, report=str(REPORT))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'run.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            status('checking_inputs')
            run()
        except BaseException as error:
            status('failed', error=str(error), traceback=traceback.format_exc())
            traceback.print_exc(); raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__=='__main__':
    main()
