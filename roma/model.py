# roma/model.py

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as L
from PIL import Image

import romatch.models.matcher as matcher_mod
from romatch import roma_outdoor
from romatch.utils.utils import get_tuple_transform_ops


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch: ConvRefiner ohne use_custom_corr
# ─────────────────────────────────────────────────────────────────────────────

_orig_init = matcher_mod.ConvRefiner.__init__

def _patched_init(self, *args, **kwargs):
    kwargs["use_custom_corr"] = False
    _orig_init(self, *args, **kwargs)

matcher_mod.ConvRefiner.__init__ = _patched_init


# ─────────────────────────────────────────────────────────────────────────────
# Utility-Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def bn_to_gn(module, groups=8):
    """BatchNorm2d -> InstanceNorm2d (affine) rekursiv."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name,
                    torch.nn.InstanceNorm2d(child.num_features, affine=True, eps=child.eps))
        else:
            bn_to_gn(child, groups)


def freeze_bn(module):
    """BatchNorm-Layer auf eval + running stats fixieren."""
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.track_running_stats = False
            m.momentum = 0.0
    return module


# roma/channel_gen.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGenerator(nn.Module):
    """
    Maps 1-channel grayscale → 2 auxiliary channels.
    Lightweight U-Net; output is concatenated with the original → 3ch for DINOv2.
    """
    def __init__(self, base_ch=32, depth=4, out_ch=2):
        super().__init__()
        self.depth = depth

        # Encoder
        self.enc = nn.ModuleList()
        ch_in = 1
        for i in range(depth):
            ch_out = base_ch * (2 ** i)
            self.enc.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                nn.GroupNorm(8, ch_out),
                nn.GELU(),
                nn.Conv2d(ch_out, ch_out, 3, padding=1),
                nn.GroupNorm(8, ch_out),
                nn.GELU(),
            ))
            ch_in = ch_out

        # Bottleneck
        bottleneck_ch = ch_in * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch_in, bottleneck_ch, 3, padding=1),
            nn.GroupNorm(8, bottleneck_ch),
            nn.GELU(),
        )

        # Decoder — use bottleneck_ch as the starting ch_in
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch  # ← this was missing
        for i in range(depth - 1, -1, -1):
            ch_out = base_ch * (2 ** i)
            self.dec.append(nn.Sequential(
                nn.Conv2d(ch_in + ch_out, ch_out, 3, padding=1),
                nn.GroupNorm(8, ch_out),
                nn.GELU(),
            ))
            ch_in = ch_out

        # Head: 2 output channels, tanh → [-1, 1] to match DINOv2 normalization
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, out_ch, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """x: (B, 1, H, W) normalized grayscale → (B, 2, H, W)"""
        input_size = x.shape[-2:]

        skips = []
        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)

        x = self.bottleneck(x)

        for dec_block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = self.head(x)

        # Guarantee exact spatial dimensions
        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return x


def make_rgb(gray, generator):
    """
    gray: (B, 1, H, W) in [0, 1] or normalized
    Returns (B, 3, H, W) ready for DINOv2.
    """
    aux = generator(gray)  # (B, 2, H, W)
    return torch.cat([gray, aux], dim=1)

def make_rgb_residual(gray, generator, scale=1.0):
    """
    gray: (B,1,H,W). Basis = gray verdreifacht.
    Generator schätzt 2-Kanal-Residual, addiert nur auf Kanäle 1 und 2.
    Kanal 0 bleibt das reine Graubild. Bei res≈0 → [gray,gray,gray] (= replicate).
    Returns (B,3,H,W).
    """
    base = gray.repeat(1, 3, 1, 1)
    res = generator(gray)                       # (B,2,H,W)
    return base + scale * F.pad(res, (0, 0, 0, 0, 1, 0))   # 0 voran → auf Kanäle 1,2


# ─────────────────────────────────────────────────────────────────────────────
# Forward mit Gradient
# ─────────────────────────────────────────────────────────────────────────────

def match_with_grad(roma_model, im_A_t, im_B_t, device,
                    symmetric=False, im_A_up=None, im_B_up=None, warp_mode='bilinear'):
    """RoMa-Forward mit Gradient-Fluss. Gibt (warp, certainty) zurück."""
    roma_model.train(False)

    if im_A_t.ndim == 3:
        im_A_t = im_A_t[None]
        im_B_t = im_B_t[None]
        if im_A_up is not None:
            im_A_up, im_B_up = im_A_up[None], im_B_up[None]

    hs, ws = roma_model.h_resized, roma_model.w_resized
    scale_factor = math.sqrt(hs * ws / (560 ** 2))

    batch = {"im_A": im_A_t.to(device), "im_B": im_B_t.to(device)}
    finest_scale = 1

    if symmetric:
        corresps = roma_model.forward_symmetric(batch, scale_factor=scale_factor)
    else:
        corresps = roma_model(batch, batched=True, scale_factor=scale_factor)

    if roma_model.upsample_preds:
        hs, ws = roma_model.upsample_res

    low_res_certainty = 0
    if roma_model.attenuate_cert:
        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws),
            align_corners=False, mode=warp_mode,
        )
        low_res_certainty = 0.5 * low_res_certainty * (low_res_certainty < 0)

    finest_corresps = corresps[finest_scale]

    if roma_model.upsample_preds:
        assert im_A_up is not None and im_B_up is not None
        batch = {"im_A": im_A_up.to(device), "im_B": im_B_up.to(device),
                 "corresps": finest_corresps}
        scale_factor = math.sqrt(
            roma_model.upsample_res[0] * roma_model.upsample_res[1] / (560 ** 2))
        if symmetric:
            corresps = roma_model.forward_symmetric(
                batch, upsample=True, batched=True, scale_factor=scale_factor)
        else:
            corresps = roma_model(
                batch, batched=True, upsample=True, scale_factor=scale_factor)

    im_A_to_im_B = corresps[finest_scale]["flow"]
    certainty = corresps[finest_scale]["certainty"] - (
        low_res_certainty if roma_model.attenuate_cert else 0)

    if finest_scale != 1:
        im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode=warp_mode)
        certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode=warp_mode)

    certainty = certainty.sigmoid()

    b, h, w = im_A_to_im_B.shape[0], hs, ws
    im_A_coords = torch.meshgrid(
        torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
        torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
        indexing="ij",
    )
    im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]), dim=-1)[None].expand(b, h, w, 2)
    warp = torch.cat((im_A_coords, im_A_to_im_B.permute(0, 2, 3, 1)), dim=-1)

    if symmetric:
        A_to_B, B_to_A = warp.chunk(2)
        certainty_A, certainty_B = certainty.chunk(2)
        warp = torch.cat((A_to_B, B_to_A), dim=2)
        certainty = torch.cat((certainty_A, certainty_B), dim=3)

    return warp, certainty[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Loss-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _local_norm(x: torch.Tensor, window: int = 32) -> torch.Tensor:
    """Räumlich variierende Mittelwert/Std-Normalisierung (erwartet 1 Kanal).
    Asymmetrisches Padding, damit Ausgabegröße == Eingabegröße (odd UND even window)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    pad = window // 2
    pad_total = window - 1                 # H + pad_total - window + 1 = H
    pad_left, pad_right = pad, pad_total - pad   # asymmetrisch -> exakt H
    x_pad = F.pad(x, (pad_left, pad_right, pad_left, pad_right))
    k = torch.ones(1, 1, window, window, device=x.device, dtype=x.dtype) / (window ** 2)
    mu = F.conv2d(x_pad, k)                # kein padding mehr nötig
    var = F.conv2d(x_pad ** 2, k) - mu ** 2
    return (x - mu) / var.clamp(min=1e-4).sqrt()   # (B,1,H,W)


def _global_norm(x: torch.Tensor) -> torch.Tensor:
    """Globale Instance-Norm (Original-Verhalten)."""
    return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (
        x.std(dim=(1, 2, 3), keepdim=True) + 1e-6)


def _gradient_l1(pred: torch.Tensor, target: torch.Tensor,
                 cert: torch.Tensor | None = None,
                 kernel_size: int = 3, warp_mode='bilinear') -> torch.Tensor:
    """L1 auf Sobel-Gradientenbetrag.
    Invariant gegen monotone Wert-Remapping (CLAHE) und additive Offsets."""
    if kernel_size == 3:
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    else:  # 5x5 Scharr
        kx = torch.tensor([[-1, 0, 1], [-10, 0, 10], [-1, 0, 1]],
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()

    def grad_mag(x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-8).sqrt()

    diff = (grad_mag(pred) - grad_mag(target)).abs()

    if cert is not None:
        if cert.dim() == 3:
            cert = cert.unsqueeze(1)
        if cert.shape[-2:] != diff.shape[-2:]:
            cert = F.interpolate(cert, size=diff.shape[-2:],
                                 mode=warp_mode, align_corners=False)
        return (cert * diff).sum() / (cert.sum() + 1e-6)
    return diff.mean()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 11) -> torch.Tensor:
    """1 − mean SSIM. Invariant gegen lokale Helligkeits-/Kontrastverschiebungen."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size,
                        device=pred.device, dtype=pred.dtype) / (window_size ** 2)

    mu1 = F.conv2d(pred, kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = F.conv2d(pred ** 2, kernel, padding=pad) - mu1_sq
    s2 = F.conv2d(target ** 2, kernel, padding=pad) - mu2_sq
    s12 = F.conv2d(pred * target, kernel, padding=pad) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1 - ssim_map.mean()


def _blur_robust_l1(pred: torch.Tensor, target: torch.Tensor,
                    sigma: float = 3.0,
                    cert: torch.Tensor | None = None,
                    warp_mode: str = "bilinear") -> torch.Tensor:
    """Pixel-L1 nach Gaussian-Blur auf gemeinsame Frequenzbandbreite.
    Reduziert die Strafe für Motion-Blur-Unterschiede zwischen Kanälen."""
    k = int(2 * 3 * sigma + 1) | 1

    def gauss_blur(x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
        x = F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
        return x.squeeze(1)

    diff = (gauss_blur(pred) - gauss_blur(target)).abs()
    if cert is not None:
        if cert.dim() == 3:
            cert = cert.unsqueeze(1)
        if cert.shape[-2:] != diff.shape[-2:]:
            cert = F.interpolate(cert, size=diff.shape[-2:],
                                 mode=warp_mode, align_corners=False)
        return (cert * diff).sum() / (cert.sum() + 1e-6)
    return diff.mean()


def _pixel_l2(pred: torch.Tensor, target: torch.Tensor,
              cert: torch.Tensor | None = None,
              warp_mode: str = "bilinear") -> torch.Tensor:
    """Gewichtetes MSE (Original-Verhalten). warp_mode für Konsistenz im Interface."""
    if cert is not None and cert.dim() == 3:
        cert = cert.unsqueeze(1)
    err = (pred - target) ** 2
    if cert is not None:
        return (cert * err).mean()
    return err.mean()


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (B,1,H,W). Kanäle sind Kopien des Grauwerts, also Mittel=Kanal0."""
    if x.dim() == 3:
        x = x[None]
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    return x


def _gen_consistency_loss(gen_A: torch.Tensor, gen_B: torch.Tensor,
                          grid_AB: torch.Tensor, grid_BA: torch.Tensor,
                          cert_AB: torch.Tensor, cert_BA: torch.Tensor,
                          warp_mode: str = "bilinear") -> torch.Tensor:
    """
    Anti-shortcut: generated channels must transform consistently under the
    estimated flow. Position/fingerprint channels fail this constraint.
    gen_A, gen_B: (B,2,H,W). grid_AB/grid_BA: (B,H,W,2) in [-1,1].
    """
    cert_AB_w = cert_AB.detach()
    cert_BA_w = cert_BA.detach()

    gen_A_to_B = F.grid_sample(gen_A, grid_BA, mode=warp_mode, align_corners=False)
    gen_B_to_A = F.grid_sample(gen_B, grid_AB, mode=warp_mode, align_corners=False)

    def w_l1(pred, target, cert):
        err = (pred - target).abs()
        # per-channel spatial normalize → removes magnitude loophole
        err = err / (err.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return (cert * err).sum() / (cert.sum() * err.shape[1] + 1e-6)

    return 0.5 * (w_l1(gen_A_to_B, gen_B, cert_BA_w) +
                  w_l1(gen_B_to_A, gen_A, cert_AB_w))


def _gen_anticollapse_loss(gen_A: torch.Tensor, gen_B: torch.Tensor,
                           var_floor: float = 0.1) -> torch.Tensor:
    """
    Anti-collapse: per-channel spatial std floor + cross-channel decorrelation.
    Prevents the generator from drifting to a constant output.
    """
    std_A = gen_A.std(dim=(2, 3))  # (B,2)
    std_B = gen_B.std(dim=(2, 3))
    l_var = F.relu(var_floor - std_A).mean() + F.relu(var_floor - std_B).mean()

    a0 = gen_A[:, 0].flatten(1)
    a1 = gen_A[:, 1].flatten(1)
    corr = (a0 * a1).mean() / (a0.std() * a1.std() + 1e-6)
    l_decorr = corr ** 2

    return l_var + 0.5 * l_decorr


# ─────────────────────────────────────────────────────────────────────────────
# Lightning-Modul
# ─────────────────────────────────────────────────────────────────────────────

class RoMaFinetuneModule(L.LightningModule):
    def __init__(
        self,
        # --- bestehende Parameter ---
        lr=3e-4,
        train_stage="1",
        cycle_weight=1.0,
        cert_floor=0.35,
        cert_reg_weight=1.0,
        clahe=False,
        symmetric=None,
        warp_mode="bilinear",
        coarse_res=560,
        upsample_res=1024,
        amp_dtype="float32",
        use_lora=False,
        lora_cfg=None,
        on_matched=True,
        # --- NEU: Loss-Optionen ---
        loss_type: str = "gradient",
        norm_mode: str = "local",
        local_norm_window: int = 32,
        blur_sigma: float = 3.0,
        gradient_kernel: int = 3,
        ssim_window: int = 11,
        # --- NEU: Komponenten-Gewichte (loss_type="mixed") ---
        w_pixel: float = 0.0,
        w_gradient: float = 1.0,
        w_ssim: float = 0.3,
        w_blur: float = 0.0,
        # --- NEU: Certainty-Supervision ---
        cert_supervision: str = "cycle",
        # --- NEU: Multi-Scale ---
        multi_scale: bool = False,
        coarse_scale: int = 16,
        # --- Sonstiges ---
        channel_mode: str = "replicate",  # NEW: "replicate" | "learned"
        gen_base_ch: int = 32,
        gen_depth: int = 4,
        w_gen_cons: float = 0.0,
        w_gen_var: float = 0.0,
        gen_residual_scale=1,
        **kwargs,
        ):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.roma = roma_outdoor(
            device=self.device_,
            coarse_res=coarse_res,
            upsample_res=upsample_res,
            amp_dtype=getattr(torch, amp_dtype),
        )
        if symmetric is None:
            symmetric = bool(self.roma.symmetric)

        self.channel_mode = channel_mode
        self.gen_residual_scale = gen_residual_scale

        if channel_mode in ("learned", "additive"):
            self.channel_gen = ChannelGenerator(base_ch=gen_base_ch, depth=gen_depth, out_ch=2)
        else:
            self.channel_gen = None  # "replicate": kein Generator

        self.save_hyperparameters()
        self._set_freeze(train_stage, use_lora=use_lora, lora_cfg=lora_cfg)
        self.warp_mode = warp_mode

    # ── Freeze-Logik ──────────────────────────────────────────────────────

    def _set_freeze(self, train_stage, use_lora=False, lora_cfg=None):
        for p in self.roma.parameters():
            p.requires_grad = False

        if use_lora:
            from peft import inject_adapter_in_model, LoraConfig
            cfg = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg["target_modules"],
            )
            self.roma = inject_adapter_in_model(cfg, self.roma)
            return

        stages = [train_stage] if isinstance(train_stage, str) else list(train_stage)
        for stage in stages:
            for p in self.roma.decoder.conv_refiner[stage].parameters():
                p.requires_grad = True

    # ── Lightning-Hooks ───────────────────────────────────────────────────

    def on_train_epoch_start(self):
        freeze_bn(self.roma.decoder.conv_refiner)

    def on_train_batch_start(self, batch, batch_idx):
        freeze_bn(self.roma.decoder.conv_refiner)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, im_A, im_B, im_A_up=None, im_B_up=None,
            symmetric=None, return_gen=False):
        if symmetric is None:
            symmetric = self.hparams.symmetric

        gen_A = gen_B = None
        if self.channel_mode == "learned":
            gen_A = self.channel_gen(im_A[:, :1])
            gen_B = self.channel_gen(im_B[:, :1])
            im_A = torch.cat([im_A[:, :1], gen_A], dim=1)
            im_B = torch.cat([im_B[:, :1], gen_B], dim=1)
            if im_A_up is not None:
                im_A_up = make_rgb(im_A_up[:, :1], self.channel_gen)
                im_B_up = make_rgb(im_B_up[:, :1], self.channel_gen)

        elif self.channel_mode == "additive":
            s = self.gen_residual_scale
            gen_A = self.channel_gen(im_A[:, :1])  # (B,2,H,W)
            gen_B = self.channel_gen(im_B[:, :1])
            im_A = im_A[:, :1].repeat(1, 3, 1, 1) + s * F.pad(gen_A, (0, 0, 0, 0, 1, 0))
            im_B = im_B[:, :1].repeat(1, 3, 1, 1) + s * F.pad(gen_B, (0, 0, 0, 0, 1, 0))
            if im_A_up is not None:
                im_A_up = make_rgb_residual(im_A_up[:, :1], self.channel_gen, scale=s)
                im_B_up = make_rgb_residual(im_B_up[:, :1], self.channel_gen, scale=s)

        warp, cert = match_with_grad(self.roma, im_A, im_B, self.device_,
                                     symmetric=symmetric,
                                     im_A_up=im_A_up, im_B_up=im_B_up,
                                     warp_mode=self.warp_mode)

        if return_gen:
            return warp, cert, gen_A, gen_B
        return warp, cert

    # ── Normalisierung ────────────────────────────────────────────────────

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.hparams.norm_mode
        if mode == "local":
            return _local_norm(x, window=self.hparams.local_norm_window)
        elif mode == "global":
            return _global_norm(x)
        return x

    # ── Loss-Dispatch ─────────────────────────────────────────────────────

    def _compute_pair_loss(self, pred: torch.Tensor, target: torch.Tensor,
                           cert: torch.Tensor | None) -> torch.Tensor:
        """Wählt den konfigurierten Loss-Typ aus."""
        lt = self.hparams.loss_type
        wm = self.warp_mode

        if lt == "pixel":
            return _pixel_l2(pred, target, cert, warp_mode=wm)

        if lt == "gradient":
            return _gradient_l1(pred, target, cert,
                                kernel_size=self.hparams.gradient_kernel,
                                warp_mode=self.warp_mode)

        if lt == "blur_robust":
            return _blur_robust_l1(pred, target,
                                   sigma=self.hparams.blur_sigma, cert=cert, warp_mode=wm)

        if lt == "ssim":
            return _ssim_loss(pred, target,
                              window_size=self.hparams.ssim_window)

        if lt == "mixed":
            total = torch.tensor(0.0, device=pred.device)
            if self.hparams.w_pixel > 0:
                total = total + self.hparams.w_pixel * _pixel_l2(
                    pred, target, cert, warp_mode=wm)
            if self.hparams.w_gradient > 0:
                total = total + self.hparams.w_gradient * _gradient_l1(
                    pred, target, cert,
                    kernel_size=self.hparams.gradient_kernel,
                    warp_mode=self.warp_mode)
            if self.hparams.w_ssim > 0:
                total = total + self.hparams.w_ssim * _ssim_loss(
                    pred, target, window_size=self.hparams.ssim_window)
            if self.hparams.w_blur > 0:
                total = total + self.hparams.w_blur * _blur_robust_l1(
                    pred, target, sigma=self.hparams.blur_sigma, cert=cert, warp_mode=wm)
            return total

        raise ValueError(f"Unbekannter loss_type: {lt}")

    # ── Symmetrischer Transfer-Loss ───────────────────────────────────────

    def _symmetric_transfer_loss(self, batch):
        im_A = batch["im_A"].to(self.device_)
        im_B = batch["im_B"].to(self.device_)
        im_A_up = batch.get("im_A_up")
        im_B_up = batch.get("im_B_up")
        if im_A_up is not None:
            im_A_up, im_B_up = im_A_up.to(self.device_), im_B_up.to(self.device_)

        sym = self.hparams.symmetric
        wm = self.warp_mode

        # Forward-Pässe
        warp_AB, cert_AB, gen_A, gen_B = self.forward(
            im_A, im_B, im_A_up=im_A_up, im_B_up=im_B_up,
            symmetric=sym, return_gen=True)
        warp_BA, cert_BA = self.forward(
            im_B, im_A, im_A_up=im_B_up, im_B_up=im_A_up,
            symmetric=sym)

        # Grids + Certainty extrahieren
        if sym:
            Wm = warp_AB.shape[2] // 2
            grid_AB = warp_AB[:, :, :Wm, 2:]
            grid_BA = warp_BA[:, :, :Wm, 2:]
            Wc = cert_AB.shape[-1] // 2
            cert_AB = cert_AB[..., :Wc].unsqueeze(1)
            cert_BA = cert_BA[..., :Wc].unsqueeze(1)
        else:
            grid_AB = warp_AB[:, :, :, 2:]
            grid_BA = warp_BA[:, :, :, 2:]
            cert_AB = cert_AB.unsqueeze(1)
            cert_BA = cert_BA.unsqueeze(1)

        cert_AB_w = cert_AB.detach()
        cert_BA_w = cert_BA.detach()

        # Normalisierung
        if im_A_up is not None:
            im_A_n = self._normalize(_to_gray(im_A_up))
            im_B_n = self._normalize(_to_gray(im_B_up))
        else:
            im_A_n = self._normalize(_to_gray(im_A))
            im_B_n = self._normalize(_to_gray(im_B))

        # Warping — warp_mode wird hier durchgereicht
        A_prime = F.grid_sample(im_A_n, grid_BA, mode=wm, align_corners=False)
        B_prime = F.grid_sample(im_B_n, grid_AB, mode=wm, align_corners=False)
        B2 = F.grid_sample(B_prime, grid_BA, mode=wm, align_corners=False)

        # Primäre Losses
        loss_A = self._compute_pair_loss(A_prime, im_B_n, cert_BA_w)
        loss_B = self._compute_pair_loss(B_prime, im_A_n, cert_AB_w)
        loss_cycle = self._compute_pair_loss(B2, im_B_n, cert_BA_w)

        if self.channel_mode == "learned":
            loss_gen_cons = _gen_consistency_loss(
                gen_A, gen_B, grid_AB, grid_BA, cert_AB, cert_BA, warp_mode=wm)
            loss_gen_var = _gen_anticollapse_loss(gen_A, gen_B)
        else:
            loss_gen_cons = torch.tensor(0.0, device=self.device_)
            loss_gen_var = torch.tensor(0.0, device=self.device_)

        # Multi-Scale Coarse Loss (optional)
        loss_coarse = torch.tensor(0.0, device=self.device_)
        if self.hparams.multi_scale:
            scale = self.hparams.coarse_scale
            H_fine, W_fine = im_A_n.shape[-2:]
            H_c, W_c = max(1, H_fine // scale), max(1, W_fine // scale)

            im_A_c = F.interpolate(im_A_n, size=(H_c, W_c),
                                   mode=wm, align_corners=False)
            im_B_c = F.interpolate(im_B_n, size=(H_c, W_c),
                                   mode=wm, align_corners=False)
            grid_AB_c = F.interpolate(
                grid_AB.permute(0, 3, 1, 2), size=(H_c, W_c),
                mode=wm, align_corners=False).permute(0, 2, 3, 1)
            grid_BA_c = F.interpolate(
                grid_BA.permute(0, 3, 1, 2), size=(H_c, W_c),
                mode=wm, align_corners=False).permute(0, 2, 3, 1)
            cert_AB_c = F.interpolate(cert_AB, size=(H_c, W_c),
                                      mode=wm, align_corners=False)
            cert_BA_c = F.interpolate(cert_BA, size=(H_c, W_c),
                                      mode=wm, align_corners=False)

            A_c = F.grid_sample(im_A_c, grid_BA_c, mode=wm, align_corners=False)
            B_c = F.grid_sample(im_B_c, grid_AB_c, mode=wm, align_corners=False)

            loss_coarse = (
                _pixel_l2(A_c, im_B_c, cert_BA_c.detach(), warp_mode=wm) +
                _pixel_l2(B_c, im_A_c, cert_AB_c.detach(), warp_mode=wm)
            )

        # Certainty-Supervision
        cert_sup = torch.tensor(0.0, device=self.device_)
        cmode = self.hparams.cert_supervision

        if cmode in ("floor", "both"):
            floor = self.hparams.cert_floor
            cert_sup = cert_sup + self.hparams.cert_reg_weight * (
                F.relu(floor - cert_AB.mean()) ** 2 +
                F.relu(floor - cert_BA.mean()) ** 2
            )

        if cmode in ("cycle", "both"):
            cycle_err = (B2 - im_B_n).abs().detach()
            cycle_err_norm = cycle_err / (cycle_err.mean() + 1e-6)
            cert_sup = cert_sup + 0.5 * (
                (cycle_err_norm * (1.0 - cert_BA)).mean() +
                (cycle_err_norm * (1.0 - cert_AB)).mean()
            )

        # Gesamt
        total = (loss_A + loss_B +
                 self.hparams.cycle_weight * loss_cycle +
                 loss_coarse + cert_sup +
                 self.hparams.w_gen_cons * loss_gen_cons +
                 self.hparams.w_gen_var * loss_gen_var)

        return (total,
                cert_AB.mean(), cert_BA.mean(),
                loss_A.detach(), loss_B.detach(),
                loss_cycle.detach(), cert_sup.detach(),
                loss_coarse.detach())

    # ── Training / Validation Steps ───────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, cAB, cBA, lA, lB, lc, cr, lcoarse = self._symmetric_transfer_loss(batch)
        self.log("train/loss_A", lA, batch_size=1)
        self.log("train/loss_B", lB, batch_size=1)
        self.log("train/loss_cycle", lc, batch_size=1)
        self.log("train/loss_coarse", lcoarse, batch_size=1)
        self.log("train/cert_AB", cAB, batch_size=1)
        self.log("train/cert_BA", cBA, batch_size=1)
        self.log("train/cert_sup", cr, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, cAB, cBA, lA, lB, lc, cr, lcoarse = self._symmetric_transfer_loss(batch)
        self.log("val/loss_A", lA, batch_size=1)
        self.log("val/loss_B", lB, batch_size=1)
        self.log("val/loss_cycle", lc, batch_size=1)
        self.log("val/loss_coarse", lcoarse, batch_size=1)
        self.log("val/cert_AB", cAB, batch_size=1)
        self.log("val/cert_BA", cBA, batch_size=1)
        self.log("val/cert_sup", cr, batch_size=1)
        self.log("val/loss", loss, batch_size=1)
        return loss

    # ── Optimizer ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        trainable = [p for p in self.roma.parameters() if p.requires_grad]
        if self.channel_gen is not None:
            trainable += list(self.channel_gen.parameters())
        return torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=1e-4)

# ─────────────────────────────────────────────────────────────────────────────
# Inferenz-Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RomaMatcher:
    """
    Bekommt IMMER ein RoMaFinetuneModule (Vanilla oder trainiert).
    Der Forward läuft über module.forward → identisch zum Trainingspfad.
    """

    def __init__(self, module, device=None, symmetric=None,
                 warp_mode=None, clahe=None, on_matched=None):
        self.module = module
        self.module.eval()
        self.device = device or self.module.device_
        self.symmetric = (module.hparams.get("symmetric", False)
                          if symmetric is None else symmetric)
        self.warp_mode = (module.hparams.get("warp_mode", "bilinear")
                          if warp_mode is None else warp_mode)
        self.clahe = (module.hparams.get("clahe", False)
                      if clahe is None else clahe)
        self.on_matched = (module.hparams.get("on_matched", False)
                           if on_matched is None else on_matched)

    @classmethod
    def from_checkpoint(cls, ckpt_path, **kwargs):
        print(f"Loading finetuned RoMa from {ckpt_path}")
        module = RoMaFinetuneModule.load_from_checkpoint(
            ckpt_path,
            map_location=kwargs.get("device"),
            weights_only=False,
        )
        return cls(module, **kwargs)

    @classmethod
    def vanilla(cls, **kwargs):
        module = RoMaFinetuneModule(**kwargs)
        return cls(module, **kwargs)

    @torch.no_grad()
    def match_batch(self, batch, master_band, symmetric=None):
        sym = self.symmetric if symmetric is None else symmetric
        results = {}

        for item in batch:
            ref_orig, tgt_orig = item["orig_A"], item["orig_B"]
            ref_mask, tgt_mask = item["mask_A"], item["mask_B"]

            fwd_kwargs = {}
            if "im_A_up" in item:
                fwd_kwargs = {"im_A_up": item["im_A_up"][None],
                              "im_B_up": item["im_B_up"][None]}

            warp, cert = self.module.forward(
                item["im_A"][None], item["im_B"][None],
                symmetric=sym, **fwd_kwargs)
            flow_AB = self._extract_flow_sym(warp, sym)

            # Radiance UND Maske mit demselben Flow warpen
            tgt_warped = self._warp(ref_orig, tgt_orig, flow_AB)
            tgt_mask_warped = self._warp(
                ref_mask.astype(np.float32),
                tgt_mask.astype(np.float32),
                flow_AB) > 0.5
            ref_mask_out = ref_mask

            results.setdefault(item["pair_idx"], {})[item["tile_idx"]] = (
                ref_orig, tgt_warped, ref_mask_out, tgt_mask_warped,
                item["full_size"], item["tiles_per_side"])

        out = []
        for pi, tiles in results.items():
            tps = next(iter(tiles.values()))[5]
            full = next(iter(tiles.values()))[4]
            if tps == 1:
                ref, tgtw, rm, tm, _, _ = tiles[0]
            else:
                ts = tiles[0][0].shape[0]
                ref = merge_tiles_to_image(
                    [t[0] for t in tiles.values()], tps, ts, full)
                tgtw = merge_tiles_to_image(
                    [t[1] for t in tiles.values()], tps, ts, full)
                rm = merge_tiles_to_image(
                    [t[2].astype(np.float32) for t in tiles.values()],
                    tps, ts, full) > 0.5
                tm = merge_tiles_to_image(
                    [t[3].astype(np.float32) for t in tiles.values()],
                    tps, ts, full) > 0.5
            out.append({
                "idx": pi, "ref": ref, "tgt_warped": tgtw,
                "ref_mask": rm, "tgt_mask": tm,
                "is_757_master": (master_band == "Rad757"),
                "status": 0,
            })
        return out

    def _extract_flow_sym(self, warp, symmetric):
        if symmetric and warp.shape[2] % 2 == 0:
            warp = warp[:, :, :warp.shape[2] // 2, :]
        return warp[:, :, :, 2:]

    def _warp(self, ref_orig, tgt_orig, flow_AB):
        """Warpt tgt_orig mit flow_AB. warp_mode wird hier durchgereicht."""
        H, W = ref_orig.shape[:2]
        grid = F.interpolate(flow_AB.permute(0, 3, 1, 2),
                             size=(H, W), mode=self.warp_mode, align_corners=False)
        grid = grid.permute(0, 2, 3, 1)
        tgt = np.stack([tgt_orig] * 3, axis=-1).astype(np.float32)
        tgt = torch.from_numpy(np.ascontiguousarray(tgt)).permute(
            2, 0, 1).unsqueeze(0).to(self.device)
        warped = F.grid_sample(tgt, grid, mode=self.warp_mode, align_corners=False)
        return warped[0, 0].cpu().numpy()

