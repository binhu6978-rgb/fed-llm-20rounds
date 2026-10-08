"""Local integration probe only: 8 train rows for one client per method, then full CosmosQA test."""
import importlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_four_baselines20_server as r


def main():
    r.os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    r.os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    r.os.environ.setdefault('HF_HUB_OFFLINE','1');r.os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    r.MASTER=ROOT/'outputs/fedlora20_server_validation'
    r.preflight()
    ex=r.Experiment('fedexlora',smoke=True)
    checks=[]
    for method in r.METHODS:
        ex.method=method;ex.module=importlib.import_module('alg.'+method)
        ex.source,ex.source_state,ex.source_protocol=r.validate_source(method)
        ex.output=r.MASTER/method;ex.output.mkdir(parents=True,exist_ok=True)
        ex.args.alg=method;ex.args.suffix=str(ex.output)
        ex.current_round=11;ex.restore(ex.source_state['global_state'])
        start=r.function_hash(ex.capture());base=ex.base_hash()
        server,clients=ex.bind(ex.source_state['global_state']);server.round=10
        server.global_lora={k:v.to(ex.model.device).clone() for k,v in ex.source_state['global_state']['lora'].items()}
        client=clients[0];client.run(ex.model)
        assert client.last_seen_count==8 and client.last_optimizer_updates==1 and ex.base_hash()==base
        raw=r.adapter(ex.model);ex.shape_check(raw);ex.shape_check(r.cpu(client.lora))
        checkpoint=ex.output/'raw_lora.pt';r.save(checkpoint,raw);state=ex.capture();r.save(ex.output/'complete_local_state.pt',state)
        metrics=ex.score(ex.output,['cosmosqa'],11,state,checkpoint)
        checks.append(dict(method=method,source_round=10,absolute_round_index=10,train_samples=8,updates=1,
                      full_cosmosqa_test=1500,start_function_hash=start,local_function_hash=r.function_hash(state),
                      frozen_backbone=True,raw_accuracy=metrics['cosmosqa_accuracy'],formal_result=False))
        r.dump(r.MASTER/'gpu_probe_receipt.json',dict(passed=len(checks)==4,checks=checks,formal_training_not_started=True))
        print('GPU PROBE PASSED',method,flush=True)
    print('All four original clients passed tiny training + full own-domain evaluator integration.',flush=True)


if __name__=='__main__': main()
