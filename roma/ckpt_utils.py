import os
import pytorch_lightning as L


def _stage_str(train_stage):
    """Liste [1,2,4] -> '1_2_4', String '1' -> '1'."""
    if isinstance(train_stage, (list, tuple)):
        return "_".join(str(s) for s in train_stage)
    return str(train_stage)


def build_run_slug(cfg):
    """Nur für MLflow run_name (UI-Anzeige), nie als Pfad."""
    m, d, t = cfg.model, cfg.data, cfg.training
    lora = f"lora{t.lora_cfg.r}" if t.use_lora else f"stage{_stage_str(t.train_stage)}"
    tile = f"t{d.tile_size}" if d.tile_size else "full"
    return (f"lr{t.lr:.0e}_sym{int(m.symmetric)}_{lora}_{tile}"
            f"_cl{int(m.clahe)}_up{m.upsample_size}")


def build_ckpt_dir(cfg, run_id):
    """<ckpt_root>/<experiment>/<run_id>/"""
    return os.path.join(cfg.paths.out_dir, cfg.training.mlflow_experiment, run_id)


def build_filename(cfg):
    """epoch_042_val0.1234.ckpt"""
    return "epoch_{epoch:03d}_val{val_loss:.4f}"


class LogCkptPathToMLflow(L.Callback):
    """Schreibt ckpt_dir, best/last Pfad und run_slug als MLflow-Tags."""

    def __init__(self, ckpt_dir, run_slug):
        super().__init__()
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.run_slug = run_slug

    def _set(self, trainer, key, value):
        client = trainer.logger.experiment
        run_id = trainer.logger.run_id
        client.set_tag(run_id, key, value)

    def on_fit_start(self, trainer, pl_module):
        self._set(trainer, "ckpt_dir", self.ckpt_dir)
        self._set(trainer, "run_slug", self.run_slug)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        ck = trainer.checkpoint_callback
        if ck is not None:
            if ck.best_model_path:
                self._set(trainer, "best_ckpt", os.path.abspath(ck.best_model_path))
            self._set(trainer, "last_ckpt",
                      os.path.abspath(os.path.join(self.ckpt_dir, "last.ckpt")))
