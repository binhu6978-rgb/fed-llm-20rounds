"""Check that seeding a continuation never resets resumed progress or rewrites source."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts import continue_five_source_fedavg20 as r


class ContinuationTests(unittest.TestCase):
    def test_bootstrap_reuses_exact_round10_and_preserves_later_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';output=root/'continuation'
            source.mkdir();output.mkdir();(source/'round_10/global').mkdir(parents=True)
            lora={'test.lora_A.default.weight':torch.ones(2,2)}
            records=[dict(round=t,global_after_tensor_sha256=r.shared.tensor_hash(lora)) for t in range(1,11)]
            state=dict(completed_round=10,pending={},records=records,global_lora=lora)
            torch.save(state,source/'resume.pt');torch.save(lora,source/'round_10/global/lora_weights.pt')
            r.data.dump(source/'round_diagnostics.json',dict(rounds=records))
            (source/'exposure.jsonl').write_text('{"round":10}\n',encoding='utf-8')
            hashes={p:r.data.sha(p) for p in source.rglob('*') if p.is_file()}
            with patch.object(r,'SOURCE',source),patch.object(r,'OUTPUT',output):
                imported=r.bootstrap()
                self.assertEqual(imported['completed_round'],10)
                copied=torch.load(output/'resume.pt',map_location='cpu',weights_only=True)
                self.assertEqual(copied['records'],records)
                self.assertEqual((output/'exposure.jsonl').read_bytes(),(source/'exposure.jsonl').read_bytes())
                copied['completed_round']=11;copied['records'].append(dict(round=11));copied['pending']={0:dict(start_hash='round11')}
                torch.save(copied,output/'resume.pt')
                r.bootstrap()
                resumed=torch.load(output/'resume.pt',map_location='cpu',weights_only=True)
                self.assertEqual(resumed['completed_round'],11)
                self.assertEqual(resumed['pending'],copied['pending'])
                resumed['records'][0]['round']=999
                torch.save(resumed,output/'resume.pt')
                with self.assertRaises(AssertionError):
                    r.bootstrap()
            self.assertTrue(all(r.data.sha(p)==sha for p,sha in hashes.items()))

    def test_round_seed_continues_absolute_index(self):
        from utils.seed_utils import client_round_seed
        self.assertEqual(client_round_seed(42,0,10),10000072)
        self.assertEqual(client_round_seed(42,4,19),19040127)


if __name__=='__main__':
    unittest.main()
