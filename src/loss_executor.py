import torch
import numpy as np
from src.config import SAFETY_CONFIG

class DynamicLossExecutor:
    """
    Computes adaptive weighted loss with safety constraints.
    """

    def __init__(self, base_loss_fn, aux_loss_fn=None):
        self.base_fn = base_loss_fn
        self.aux_fn = aux_loss_fn
        self.loss_history = []
        self.grad_stats = {}

    def compute(self, preds, targets, params):
        # Base loss
        base = self.base_fn(preds, targets)
        mw = float(np.clip(params['main_weight'],
                           SAFETY_CONFIG['min_loss_multiplier'],
                           SAFETY_CONFIG['max_loss_multiplier']))
        loss = base * mw

        # Auxiliary loss
        if self.aux_fn and params.get('aux_weight', 0.0) > 0:
            aw = float(np.clip(params['aux_weight'], 0.0, 1.0))
            loss = loss + aw * self.aux_fn(preds, targets)

        self.loss_history.append(loss.item())
        return loss

    def compute_grad_stats(self, model):
        norms = []
        for p in model.parameters():
            if p.grad is not None:
                norms.append(p.grad.data.norm(2).item())
        total = float(np.sqrt(sum([n**2 for n in norms]))) if norms else 0.0
        var = float(np.var(norms)) if norms else 0.0
        self.grad_stats = {'norm': total, 'variance': var}
        return self.grad_stats

    def is_stable(self):
        if len(self.loss_history) < 10:
            return True
        recent = self.loss_history[-10:]
        m, v = np.mean(recent), np.var(recent)
        return (v <= m * SAFETY_CONFIG['stability_threshold']
                and not any(np.isnan(recent)))
