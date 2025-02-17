from abc import abstractmethod
import pyro
import pyro.nn as pnn
from torch import nn

class PyroModel(pnn.PyroModule):
    def forward(self, *args, **kwargs):
        tq = pyro.poutine.trace(self.guide).get_trace(*args, **kwargs)
        with pyro.poutine.replay(trace=tq):
            tp = pyro.poutine.trace(self.model).get_trace(*args, **kwargs)
        tq.compute_log_prob()
        log_q = sum(site['log_prob'] for site in tq.nodes.values()
                    if site['type'] == 'sample')
        tp.compute_log_prob()
        log_p = sum(site['log_prob'] for site in tp.nodes.values()
                    if site['type'] == 'sample')
        return tp.nodes["_RETURN"]["value"], tp, log_p - log_q

    @abstractmethod
    def guide(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def model(self, *args, **kwargs):
        raise NotImplementedError
