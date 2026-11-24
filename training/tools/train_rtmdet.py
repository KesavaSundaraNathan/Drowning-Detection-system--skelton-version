"""Phase 6: fine-tune RTMDet from its COCO weights through the queue (resumable).

Registers FreezeBackboneHook (backbone frozen for the first N epochs, BN in eval mode; afterwards trained at the LR
multiplier set in the config), then runs MMEngine's Runner. `--resume` continues from the latest checkpoint in the
work dir (one is written every epoch).

    training\\.venv\\Scripts\\python.exe training\\tools\\train_rtmdet.py training\\configs\\rtmdet_l_v0.py [--resume]
        [--cfg-options train_cfg.max_epochs=1 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmdet.utils import register_all_modules
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner


def allow_own_checkpoints() -> None:
    """Narrow weights-only allowlist for the types MMEngine writes into its own checkpoints (resume after a crash)."""
    import numpy as np
    import torch
    from mmengine.logging import HistoryBuffer

    torch.serialization.add_safe_globals([
        HistoryBuffer, np._core.multiarray._reconstruct, np._core.multiarray.scalar, np.ndarray, np.dtype,
        np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType,
    ])


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Freeze the backbone for the first `epochs` epochs (MASTER_PROMPT Phase 6: first 10 %)."""

    priority = "NORMAL"

    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def before_train_epoch(self, runner) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        frozen = runner.epoch < self.epochs
        for p in model.backbone.parameters():
            p.requires_grad = not frozen
        if frozen:
            model.backbone.eval()  # keep the COCO BN statistics while frozen
        if runner.epoch in (0, self.epochs):
            runner.logger.info(f"backbone {'frozen' if frozen else 'unfrozen'} at epoch {runner.epoch}")

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        if runner.epoch < self.epochs:  # Runner.train() switches every module back to train mode each epoch
            model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            model.backbone.eval()


@HOOKS.register_module()
class PlainCheckpointHook(Hook):
    """Keep checkpoints loadable with weights_only (no unsafe pickle globals) so a resume needs no trust override."""

    priority = "VERY_LOW"

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        import numpy as np

        def plain(x):
            if isinstance(x, np.generic):
                return x.item()
            if isinstance(x, dict):
                return {k: plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(plain(v) for v in x)
            return x

        for k in ("meta", "message_hub"):
            if k in checkpoint:
                checkpoint[k] = plain(checkpoint[k])
        if isinstance(checkpoint.get("message_hub"), dict):
            # the loss-smoothing history stores bound methods (pickle global `getattr`); a resume does not need it
            checkpoint["message_hub"]["log_scalars"] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction)
    a = ap.parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(a.config)
    if a.cfg_options:
        cfg.merge_from_dict(a.cfg_options)
    cfg.custom_hooks = [*cfg.get("custom_hooks", []), dict(type="PlainCheckpointHook")]
    # MMEngine resumes from load_from when it is set, so clear it once the work dir has a checkpoint of its own
    if a.resume and (Path(cfg.work_dir) / "last_checkpoint").exists():
        cfg.resume, cfg.load_from = True, None
        allow_own_checkpoints()
        print("resuming from", (Path(cfg.work_dir) / "last_checkpoint").read_text().strip())
    Runner.from_cfg(cfg).train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
