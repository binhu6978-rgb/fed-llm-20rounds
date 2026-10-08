"""Report the single highest-Macro centralized epoch, using completed saved evaluations."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cosmosqa_five_data as data
from scripts.evaluate_cosmosqa_best_saved import central_summary


def main():
    output = data.MASTER / 'best_saved_analysis'
    audit = data.read(output / 'audit.json')
    assert audit['passed'] and audit['all_ten_centralized_epochs_scored']
    records = data.read(output / 'centralized_epoch_metrics.json')
    result = central_summary(records)
    endpoint = records[-1]['metrics']
    report = ROOT / 'reports/cosmosqa_centralized_best_epoch_seed42.md'
    result['gain_over_epoch10_pp'] = 100 * (result['selected_metrics']['macro_accuracy'] - endpoint['macro_accuracy'])
    lines = ['# Centralized-All: highest saved epoch (CosmosQA five domains, seed42)', '',
        f"Selected epoch: **{result['selected_epoch']}**.",
        f"Highest Macro: **{100 * result['selected_metrics']['macro_accuracy']:.2f}%**.",
        f"Epoch10 Macro: **{100 * endpoint['macro_accuracy']:.2f}%**; change: **{result['gain_over_epoch10_pp']:+.2f} pp**.", '',
        'All ten saved checkpoints were evaluated on the same fixed 7498-row test with the unchanged '
        'choice-likelihood evaluator. The selected epoch maximizes the unweighted mean of five domain accuracies; '
        'ties select the earliest epoch. All accuracies below come from that single shared checkpoint.', '',
        '| Dataset | Selected-epoch accuracy |', '| --- | ---: |']
    for d in data.DOMAINS:
        lines.append(f"| {data.NAMES[d]} | {100 * result['selected_metrics'][d + '_accuracy']:.2f}% |")
    lines += [f"| Macro | {100 * result['selected_metrics']['macro_accuracy']:.2f}% |", '',
        '## Every saved epoch', '', '| Epoch | ' + ' | '.join(data.NAMES[d] for d in data.DOMAINS) + ' | Macro |',
        '| ---: | ' + ' | '.join(['---:'] * 6) + ' |']
    for record in records:
        values = record['metrics']
        lines.append(f"| {record['epoch']} | " + ' | '.join(f"{100 * values[d + '_accuracy']:.2f}%" for d in data.DOMAINS)
                     + f" | {100 * values['macro_accuracy']:.2f}% |")
    lines += ['', f"Selected checkpoint: `{result['selected_checkpoint']}`.",
        f"Per-epoch predictions, A/B/C/D log probabilities and receipts: `{output.relative_to(ROOT)}/centralized/epoch_XX/`.",
        'Epoch10 predictions/receipt are reused from the original Centralized directory after matching the tensor hash.',
        'Training, hyperparameters, prepared rows, original results and checkpoints were unchanged; the integrity audit passed.',
        'This is a retrospective maximum selected on the test, reported separately from the fixed epoch10 result. '
        'Independent Local is not retrained or substituted.']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result['report'] = str(report)
    data.dump(output / 'centralized_best_result.json', result)
    print(json.dumps({key: result[key] for key in ('selected_epoch', 'selected_metrics', 'gain_over_epoch10_pp', 'report')}, ensure_ascii=False))


if __name__ == '__main__': main()
