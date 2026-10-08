"""Score an immutable snapshot of the latest committed Independent Local epoch."""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cosmosqa_sanity as runner
from scripts import run_three_dataset_lora as shared
from scripts.cosmosqa_data import MASTER, read, dump, sha


def main():
    import torch
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    resume = MASTER / 'clientlocal/client_cosmosqa/resume.pt'
    state = torch.load(io.BytesIO(resume.read_bytes()), map_location='cpu', weights_only=True)
    epoch = state['epoch']
    assert 1 <= epoch <= 10 and len(state['curves']) == epoch
    fingerprint = shared.tensor_hash(state['lora'])
    assert fingerprint == state['curves'][-1]['after_lora_tensor_sha256']
    output = MASTER / f'diagnostics/epoch_{epoch:02d}'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'lora_weights.pt'
    if checkpoint.exists():
        assert shared.tensor_hash(torch.load(checkpoint, map_location='cpu', weights_only=True)) == fingerprint
    else:
        shared.save_state(checkpoint, state['lora'])
    capture = dict(epoch=epoch, captured_at=shared.now(), source=str(resume.relative_to(ROOT)),
        source_epoch_record=state['curves'][-1], lora_tensor_sha256=fingerprint,
        checkpoint_sha256=sha(checkpoint), diagnostic_only=True, formal_endpoint_epoch=10,
        selection_or_hyperparameter_changes=False)
    dump(output / 'capture_receipt.json', capture)
    print('Evaluating committed epoch', epoch, flush=True)
    experiment = runner.Experiment()
    experiment.load(state['lora'])
    result = experiment.score(output, ('cosmosqa',), epoch, checkpoint)
    base = read(MASTER / 'base/result.json')['cosmosqa_accuracy']
    summary = dict(epoch=epoch, accuracy=result['cosmosqa_accuracy'], base_accuracy=base,
        local_minus_base_pp=100 * (result['cosmosqa_accuracy'] - base), diagnostic_only=True,
        formal_endpoint_epoch=10, train_loss=state['curves'][-1]['train_loss'], checkpoint=str(checkpoint),
        evaluation_receipt=str(output / 'evaluation_provenance.json'))
    dump(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
