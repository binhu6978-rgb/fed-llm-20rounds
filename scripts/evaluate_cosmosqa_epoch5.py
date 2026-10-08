"""Capture committed epoch5 and score it with the existing evaluator, read-only to training."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cosmosqa_sanity as runner
from scripts.cosmosqa_data import MASTER, read, dump, sha
from scripts import run_three_dataset_lora as shared


def main():
    import torch
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    output = MASTER / 'diagnostics/epoch_05'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'lora_weights.pt'
    current = MASTER / 'clientlocal/client_cosmosqa'
    if checkpoint.exists():
        capture = read(output / 'capture_receipt.json')
        assert capture['epoch'] == 5 and sha(checkpoint) == capture['checkpoint_sha256']
    else:
        started = time.monotonic()
        while True:
            resume = current / 'resume.pt'
            if resume.exists():
                # Never keep a source file handle open while the training writer commits its next epoch.
                import io
                state = torch.load(io.BytesIO(resume.read_bytes()), map_location='cpu', weights_only=True)
                epoch = state['epoch']
                if epoch > 5:
                    raise RuntimeError('Epoch5 has already been overwritten; refusing to substitute another epoch or retrain')
                if epoch == 5:
                    assert len(state['curves']) == 5 and state['curves'][-1]['epoch'] == 5
                    fingerprint = shared.tensor_hash(state['lora'])
                    assert fingerprint == state['curves'][-1]['after_lora_tensor_sha256']
                    shared.save_state(checkpoint, state['lora'])
                    capture = dict(epoch=5, captured_at=shared.now(), source=str(resume.relative_to(ROOT)),
                        source_epoch_record=state['curves'][-1], lora_tensor_sha256=fingerprint,
                        checkpoint_sha256=sha(checkpoint), diagnostic_only=True,
                        formal_endpoint_epoch=10, selection_or_hyperparameter_changes=False)
                    dump(output / 'capture_receipt.json', capture)
                    print('Committed epoch5 captured:', json.dumps(capture), flush=True)
                    break
            status = read(MASTER / 'status.json')
            if status['stage'] in ('failed', 'complete'):
                raise RuntimeError(f'Training ended before epoch5 capture: {status}')
            if time.monotonic() - started > 3600:
                raise RuntimeError('Epoch5 capture timed out; no training changes made')
            time.sleep(1)
    experiment = runner.Experiment()
    value = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert shared.tensor_hash(value) == capture['lora_tensor_sha256']
    experiment.load(value)
    result = experiment.score(output, ('cosmosqa',), 5, checkpoint)
    base = read(MASTER / 'base/result.json')['cosmosqa_accuracy']
    dump(output / 'summary.json', dict(epoch=5, accuracy=result['cosmosqa_accuracy'], base_accuracy=base,
        local_minus_base_pp=100 * (result['cosmosqa_accuracy'] - base), diagnostic_only=True,
        formal_endpoint_epoch=10, train_loss=capture['source_epoch_record']['train_loss'],
        checkpoint=str(checkpoint), evaluation_receipt=str(output / 'evaluation_provenance.json')))
    print(json.dumps(read(output / 'summary.json')), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        dump(MASTER / 'diagnostics/epoch_05/failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
