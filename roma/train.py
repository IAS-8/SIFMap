# SIFMap/roma/roma_train.py
import os
import torch
import pytorch_lightning as L
import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from roma.data_module import SIFMatchDataModule
from roma.model import RoMaFinetuneModule

from roma.ckpt_utils import (build_run_slug, build_ckpt_dir, build_filename,
                             LogCkptPathToMLflow)


def _resolve_parallelism(cfg):
    """Return (accelerator, devices) as PL expects, consistently."""
    acc = cfg.training.accelerator
    if acc is None:
        acc = "gpu" if torch.cuda.is_available() else "cpu"

    dev = cfg.training.get("devices", None) if hasattr(cfg.training, "get") else getattr(cfg.training, "devices", None)

    if acc == "cpu":
        devices = 1 if dev in (None, "auto") else int(dev)
    else:
        if dev in (None, "auto"):
            devices = "auto"
        elif isinstance(dev, (list, tuple)):
            devices = list(dev)
        else:
            devices = int(dev)
    return acc, devices


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    torch.set_float32_matmul_precision('high')

    run_slug = build_run_slug(cfg)

    # 1. MLflow Logger ZUERST -> liefert run_id
    mlflow = MLFlowLogger(
        experiment_name=cfg.training.mlflow_experiment,
        tracking_uri=cfg.paths.mlflow_uri,
        run_name=run_slug,
        log_model=False,
    )
    mlflow.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # 2. run_id verfügbar -> Pfad ist Hex-String, YAML-sicher
    run_id = mlflow.run_id
    ckpt_dir = build_ckpt_dir(cfg, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 3. Loss-Block auflösen. Fallbacks = Original-Verhalten, damit ein
    #    Config ohne 'loss:'-Sektion exakt den alten Loss reproduziert.
    loss_cfg = cfg.training.get("loss", {}) if hasattr(cfg.training, "get") else getattr(cfg.training, "loss", {})

    # 4. Channel-Generator-Optionen (Fallback = replicate = Original)
    channel_mode = cfg.model.get("channel_mode", "replicate")

    module = RoMaFinetuneModule(
        lr=cfg.training.lr, train_stage=cfg.training.train_stage,
        cycle_weight=cfg.training.cycle_weight,
        cert_floor=cfg.training.cert_floor,
        cert_reg_weight=cfg.training.cert_reg_weight,
        clahe=cfg.model.clahe, symmetric=cfg.model.symmetric,
        warp_mode=cfg.model.warp_mode,
        coarse_res=cfg.model.coarse_res,
        upsample_res=cfg.model.upsample_size,
        amp_dtype=cfg.training.amp_dtype,
        use_lora=cfg.training.use_lora,
        lora_cfg=OmegaConf.to_container(cfg.training.lora_cfg,
                                        resolve=True) if cfg.training.lora_cfg is not None else dict(),
        on_matched=cfg.model.on_matched,
        # ── Channel-Generator ──
        channel_mode=channel_mode,
        gen_base_ch=cfg.model.get("gen_base_ch", 32),
        gen_depth=cfg.model.get("gen_depth", 4),
        # ── Loss-Optionen (Fallbacks = Original) ──
        loss_type=loss_cfg.get("loss_type", "pixel"),
        norm_mode=loss_cfg.get("norm_mode", "global"),
        local_norm_window=loss_cfg.get("local_norm_window", 32),
        blur_sigma=loss_cfg.get("blur_sigma", 3.0),
        gradient_kernel=loss_cfg.get("gradient_kernel", 3),
        ssim_window=loss_cfg.get("ssim_window", 11),
        w_pixel=loss_cfg.get("w_pixel", 1.0),
        w_gradient=loss_cfg.get("w_gradient", 0.0),
        w_ssim=loss_cfg.get("w_ssim", 0.0),
        w_blur=loss_cfg.get("w_blur", 0.0),
        cert_supervision=loss_cfg.get("cert_supervision", "floor"),
        multi_scale=loss_cfg.get("multi_scale", False),
        coarse_scale=loss_cfg.get("coarse_scale", 16),
        # ── Channel-Generator-Loss-Gewichte ──
        w_gen_cons=loss_cfg.get("w_gen_cons", 0.0),
        w_gen_var=loss_cfg.get("w_gen_var", 0.0),
        gen_var_floor=loss_cfg.get("gen_var_floor", 0.1),
    )

    # 5. Resume / Init-from-checkpoint
    resume_from = cfg.training.get("resume_from", None)
    init_from = cfg.training.get("init_from", None)

    if init_from:
        module = RoMaFinetuneModule.load_from_checkpoint(init_from, weights_only=False)
        module.lr = cfg.training.lr  # allow LR override without touching hparams

    roma = module.roma
    upsample_preds = bool(roma.upsample_preds)
    hs, ws = roma.h_resized, roma.w_resized
    upsample_res = tuple(roma.upsample_res) if upsample_preds else None

    dm = SIFMatchDataModule(
        base_dir=cfg.paths.data_base_dir, train_dir=cfg.data.train_dir,
        val_dir=cfg.data.val_dir, cam760_regex=cfg.data.cam760_regex,
        cam757_regex=cfg.data.cam757_regex, datatake_regex=cfg.data.datatake_regex,
        master_band=cfg.data.master_band,
        hs=hs, ws=ws, upsample_res=upsample_res, upsample_preds=upsample_preds,
        clahe=module.hparams.clahe, divide_by=cfg.data.divide_by,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        tile_size=cfg.data.tile_size, return_orig=False,
        on_matched=cfg.model.on_matched,
        sharpness_top_k=cfg.data.sharpness_top_k,
        sharpness_min_pct=cfg.data.sharpness_min_pct
    )

    # ── WICHTIG: Monitor-Key ist jetzt 'val/loss' (Slash), nicht 'val_loss' ──
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=build_filename(cfg),
        monitor='val/loss', mode='min',
        save_top_k=cfg.training.save_top_k,
        save_last=True,
        every_n_epochs=cfg.training.save_every_n_epochs,
        auto_insert_metric_name=False,
    )

    early_stop = EarlyStopping(
        monitor='val/loss', mode='min',
        patience=cfg.training.get('patience', 10),
        min_delta=cfg.training.get('min_delta', 1e-4),
        verbose=True,
    )

    accelerator, devices = _resolve_parallelism(cfg)
    strategy = cfg.training.get("strategy", "auto") if hasattr(cfg.training, "get") else getattr(cfg.training, "strategy", "auto")

    if accelerator == 'cpu':
        module.float()

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision='32-true',
        gradient_clip_val=cfg.training.grad_clip,
        callbacks=[ckpt, LearningRateMonitor(logging_interval='epoch'),
                   LogCkptPathToMLflow(ckpt_dir, run_slug), early_stop],
        logger=mlflow,
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.val_every,
    )
    trainer.fit(module, datamodule=dm, ckpt_path=resume_from)

    print('Run slug :', run_slug)
    print('Run ID   :', run_id)
    print('Ckpt dir :', os.path.abspath(ckpt_dir))
    print('Best ckpt:', ckpt.best_model_path)


if __name__ == "__main__":
    main()
