"""Build a minimal, hash-audited transferable bundle for four Round10-to20 continuations."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_four_baselines20_server as runner


def build(archive=False):
    files=set()
    for directory in ('alg','utils','scripts','configs','tests'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml','.yml') and '__pycache__' not in p.parts:
                files.add(p)
    for p in (ROOT/'models/models--llama3.2-1B').iterdir():
        if p.is_file(): files.add(p)
    files.add(ROOT/'models/initial_lora/seed42_r8_alpha32_qv.pt')
    files.update(p for p in (ROOT/'dataset/cosmosqa_five_client4000_seed42').rglob('*') if p.is_file())
    files.add(ROOT/'dataset/mcq_balanced4000/race/manifest.json')
    files.add(ROOT/'requirements-fedlora20-server.txt')
    files.add(ROOT/'docs/FEDLORA20_SERVER_TRANSFER.md')
    baseline=ROOT/'outputs/fedlora_baselines_5source_seed42'
    previous=runner.read(baseline/'code_state.json')['file_sha256']
    # Verify unchanged protocol-defining code against the original run, not just a new checksum.
    for path,digest in previous.items():
        p=runner.portable_path(path)
        if p in files: assert runner.sha(p)==digest,'Original code/model changed: '+path
    for name in ('code_state.json','environment_receipt.json'):
        files.add(baseline/name)
    files.add(baseline/'local_before_upload/round_metrics.json')
    sources={}
    for method in runner.METHODS:
        source,state,protocol=runner.validate_source(method)
        for name in ('resume.pt','protocol.json','protocol_audit.json','round_trajectory.json','exposure.jsonl','result.json','completed.txt'):
            files.add(source/name)
        best=runner.read(source/'result.json')['best_round']
        for t in {10,best}:
            for name in ('complete_state.pt','evaluation_provenance.json','full_function_identity.json','result.json'):
                files.add(source/f'round_{t:02d}/global'/name)
        sources[method]=dict(resume=runner.relative(source/'resume.pt'),round=10,
             complete_function_sha256=runner.function_hash(state['global_state']),
             complete_backbone=method in runner.MUTATORS,original_best_round=best,
             algorithm_parameters=protocol['algorithm_parameters'],warnings=protocol['warnings'])
    for name in ('best_result.json','round_diagnostics.json','protocol_audit.json'):
        files.add(ROOT/'outputs/fedavg20_five_client4000_seed42'/name)
    assert all(p.is_file() for p in files),'A required migration input is missing'
    hashes={runner.relative(p):runner.sha(p) for p in sorted(files)}
    total=sum(p.stat().st_size for p in files)
    manifest=dict(format_version=1,methods=list(runner.METHODS),target_rounds=20,sources=sources,
             source_environment=runner.read(baseline/'environment_receipt.json'),files=hashes,
             total_bytes=total,required_raw_datasets=False,old_client_checkpoints_required=False,
             original_rounds_not_retrained=True,expected_new_updates_per_method=25000,
             expected_new_sample_visits_per_method=200000,created_at=runner.now())
    runner.dump(runner.MANIFEST,manifest)
    dest=ROOT/'server_transfer';dest.mkdir(exist_ok=True)
    listing=dest/'fedlora20_files.txt'
    listing.write_text('\n'.join(hashes)+'\nserver_fedlora20_manifest.json\n',encoding='utf-8')
    print('Transfer files',len(files)+1,'size GiB',round(total/2**30,3),flush=True)
    print('Inventory',listing,flush=True)
    if archive:
        target=dest/'fedlora20_server_seed42.tar'
        tmp=target.with_suffix('.tmp')
        with tarfile.open(tmp,'w',dereference=True) as tar:
            for p in sorted(files): tar.add(p,arcname=runner.relative(p),recursive=False)
            tar.add(runner.MANIFEST,arcname='server_fedlora20_manifest.json',recursive=False)
        tmp.replace(target)
        runner.dump(dest/'archive_receipt.json',dict(archive=runner.relative(target),bytes=target.stat().st_size,
                    sha256=runner.sha(target),manifest_sha256=runner.sha(runner.MANIFEST)))
        print('Ready-to-upload archive',target,flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',action='store_true',help='Also create one uncompressed .tar upload file')
    build(parser.parse_args().archive)
