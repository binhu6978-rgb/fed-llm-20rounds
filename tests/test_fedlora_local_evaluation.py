"""Check the model reconstruction most likely to invalidate retrospective local scores."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from scripts import evaluate_fedlora_local_before_upload as evaluation


class LocalStateTests(unittest.TestCase):
    def test_previous_global_backbone_and_raw_not_rotated_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'method'
            for t in (1,2):
                (folder/f'round_{t:02d}/client_cosmosqa').mkdir(parents=True)
                (folder/f'round_{t:02d}/global').mkdir(parents=True)
            initial=dict(lora={'A':torch.ones(2)},base_weights={'base':torch.ones(2)})
            previous=dict(lora={'A':torch.ones(2)*2},base_weights={'base':torch.ones(2)*3})
            current=dict(lora={'A':torch.ones(2)*5},base_weights={'base':torch.ones(2)*7})
            torch.save(initial,folder/'initial_complete_state.pt')
            torch.save(previous,folder/'round_01/global/complete_state.pt')
            torch.save(current,folder/'round_02/global/complete_state.pt')
            raw={'A':torch.ones(2)*11}; upload={'A':torch.ones(2)*13}
            for t,start in ((1,initial),(2,previous)):
                client=folder/f'round_{t:02d}/client_cosmosqa'
                torch.save(raw,client/'raw_lora.pt'); torch.save(upload,client/'upload_lora.pt')
                record={'clients':{'cosmosqa':dict(start_function_hash=evaluation.baseline.function_hash(start),
                    start_base_hash=evaluation.shared.tensor_hash(start['base_weights']),
                    raw_hash=evaluation.shared.tensor_hash(raw),upload_hash=evaluation.shared.tensor_hash(upload),
                    budget=dict(samples=4000,optimizer_updates=500))}}
                with patch.object(evaluation,'ROOT',root):
                    state,path,identity=evaluation.local_state(folder,t,'cosmosqa',record)
                torch.testing.assert_close(state['base_weights']['base'],start['base_weights']['base'])
                torch.testing.assert_close(state['lora']['A'],raw['A'])
                self.assertNotEqual(identity['raw_hash'],identity['upload_hash'])
                self.assertEqual(identity['local_function_state_sha256'],evaluation.baseline.function_hash(state))
                record['clients']['cosmosqa']['start_function_hash']='incorrect'
                with patch.object(evaluation,'ROOT',root),self.assertRaises(AssertionError):
                    evaluation.local_state(folder,t,'cosmosqa',record)


if __name__=='__main__':
    unittest.main()
