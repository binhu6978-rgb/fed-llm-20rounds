"""Centralized pooled-data LoRA baseline.

This intentionally reuses the ordinary training and aggregation base classes.
With ``cn: 1`` the aggregation is the identity, making each configured round
one deterministic pass over the pooled training data while preserving the
same checkpoint/evaluation lifecycle as the federated runs.
"""

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)


class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()
