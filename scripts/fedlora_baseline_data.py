"""Read-only original-five protocol checks and immutable baseline provenance."""
import json
import shutil
from pathlib import Path
from scripts import cosmosqa_five_data as original

ROOT, DOMAINS, NAMES = original.ROOT, original.DOMAINS, original.NAMES
MASTER = ROOT/'outputs/fedlora_baselines_5source_seed42'
CONFIG = ROOT/'configs/fedlora_baselines_5source_seed42.yaml'
read, rows, sha, dump, write_rows = original.read, original.rows, original.sha, original.dump, original.write_rows
METHODS = ('fedexlora','fedrotlora','flexlora','fedmomentum')
NAMES_METHOD = dict(zip(METHODS,('FedEx-LoRA','FedRot-LoRA','FlexLoRA','FedMomentum')))


def config():
    import yaml
    result=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert result==dict(common_config='configs/cosmosqa_five_client4000_seed42.yaml',
        original_command='python -X utf8 -u scripts/run_cosmosqa_five_experiments.py --job fedavg',
        algorithms=dict(fedexlora={},fedrotlora=dict(lam=.5),flexlora=dict(s=2),fedmomentum=dict(residual_threshold=.9999)),
        rounds=10,smoke_rounds=2,smoke_train_samples_per_client=8,smoke_full_test_sets=True)
    return result


def suffix(method): return 'baseline_'+method+'_5source_seed42'
def output(method): return MASTER/suffix(method)


def command(method,smoke=False):
    params=config()['algorithms'][method]
    text='python -X utf8 -u scripts/run_fedlora_baseline.py --alg '+method+' --suffix '+suffix(method)
    for k,v in params.items(): text+=' --'+k+' '+str(v)
    if smoke: text+=' --smoke'
    return text


def prepare():
    cfg=config(); oldcfg=original.config(); folder=ROOT/oldcfg['prepared_dir']
    manifest=read(folder/'manifest.json'); protocol=read(original.MASTER/'fedavg/protocol.json')
    assert protocol['config']==oldcfg and protocol['canonical_initial_tensor_sha256']=='2499167c8ba147f32045ce230fd2c26d88ddca0eddf6447061848502d7b87f23'
    for path,digest in manifest['files'].items(): assert sha(ROOT/path)==digest,path
    code=dict(runner_sha256='scripts/run_three_dataset_lora.py',suite_runner_sha256='scripts/run_cosmosqa_five_experiments.py',
        mcq_trainer_sha256='utils/train_utils.py',aggregation_sha256='alg/ftbase.py',
        encoder_sha256='utils/cosmosqa_five_mcq.py',choice_evaluator_sha256='utils/mcq_eval.py',
        passage_encoder_sha256='utils/race_mcq.py',cosmos_encoder_sha256='utils/cosmosqa_mcq.py',
        previous_encoder_sha256='utils/five_client_mcq.py',seed_utils_sha256='utils/seed_utils.py',model_utils_sha256='utils/model_utils.py')
    for key,path in code.items(): assert sha(ROOT/path)==protocol[key],'Original FedAvg code differs: '+path
    assert sha(ROOT/oldcfg['initial_lora'])==protocol['initial_lora_sha256']
    expected=dict(cosmosqa=1500,openbookqa=1000,sciq=1998,hellaswag=1500,race=1500)
    source_counts={}
    for d in DOMAINS:
        train=rows(folder/'train'/(d+'.jsonl')); test=rows(folder/'test'/(d+'.jsonl'))
        assert len(train)==len({original.native_key(r) for r in train})==4000
        assert len(test)==len({original.native_key(r) for r in test})==expected[d]
        assert [original.native_key(r) for r in train]==manifest['selected_ids'][d]['train']
        assert [original.native_key(r) for r in test]==manifest['selected_ids'][d]['test']
        assert not {original.qa_fingerprint(r) for r in train}&{original.qa_fingerprint(r) for r in test}
        source_counts[d]=dict(train=4000,test=expected[d],ids_unchanged=True,weight=.2)
    trajectory=read(original.MASTER/'fedavg/round_diagnostics.json')['rounds']
    best=max(trajectory,key=lambda r:(r['global_after']['macro_accuracy'],-r['round']))
    assert len(trajectory)==10 and best['round']==10
    assert abs(best['global_after']['macro_accuracy']-.6936664664664665)<1e-12
    MASTER.mkdir(parents=True,exist_ok=True)
    statepath=MASTER/'code_state.json'
    paths=[ROOT/p for p in code.values()]+[CONFIG,ROOT/oldcfg['initial_lora'],folder/'manifest.json',
        original.MASTER/'final_results.json',original.MASTER/'fedavg/protocol.json',
        original.MASTER/'fedavg/round_diagnostics.json',ROOT/'reports/cosmosqa_five_comparison_seed42.md']
    paths += list((ROOT/'models/models--llama3.2-1B').glob('*'))
    paths += [ROOT/'alg'/(m+'.py') for m in METHODS]
    paths += [ROOT/p for p in ('scripts/run_fedlora_baseline.py','scripts/run_fedlora_baseline_sequence.py','scripts/fedlora_baseline_data.py')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
    if statepath.exists():
        saved=read(statepath)
        assert saved['file_sha256']==hashes,'Baseline code/input snapshot changed; refusing incompatible resume'
    else:
        saved=dict(git_commit=None,git_status='This workspace has no .git repository; SHA256 code/input snapshot is the code state.',
            file_sha256=hashes,baseline_implementation_modified=False,
            original_command=cfg['original_command'],commands={m:command(m) for m in METHODS})
        dump(statepath,saved)
        for m in METHODS:
            target=MASTER/'implementation_snapshot'/('alg_'+m+'.py')
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/'alg'/(m+'.py'),target)
    audit=dict(passed=True,sources=source_counts,train_union=20000,test_total=7498,
        unchanged_ids_prompts_evaluator_model_initialization=True,original_reference=dict(round=10,metrics=best['global_after']),
        budget=dict(clients=5,rounds=10,samples_per_client=4000,updates_per_client_round=500,
            total_updates=25000,total_sample_visits=200000,weights={d:.2 for d in DOMAINS}),
        code_state=str(statepath.relative_to(ROOT)))
    dump(MASTER/'preflight_audit.json',audit)
    return audit
