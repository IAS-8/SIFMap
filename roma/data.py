import numpy as np
import os
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from data.utils import load_image, save
from roma.model import RomaMatcher
from romatch.utils.utils import get_tuple_transform_ops


def _mask_from(arr):
    """NaN-Maske (True = maskiert) aus Roh-Radiance, VOR nan_to_num."""
    return np.isnan(arr)


def _fill_nan(arr):
    """NaN -> 0, damit to_uint8/grid_sample nicht irritiert werden."""
    return np.nan_to_num(arr, nan=0.0).astype(np.float32)


def tile_coords(tile_idx, tiles_per_side, tile_size, full_size):
    """
    tile_idx: linearer Index 0..tiles_per_side**2-1
    returns: (y0, x0) Offset der Kachel im Gesamtbild (Pixel).
    """
    ty = tile_idx // tiles_per_side
    tx = tile_idx % tiles_per_side
    return ty * tile_size, tx * tile_size


def split_image_to_tiles(arr, tile_size):
    """(H,W) -> Liste von (tile_size, tile_size) + Metadaten. Rand wird gepolstert."""
    H, W = arr.shape[:2]
    tiles_per_side = int(np.ceil(H / tile_size))  # quadratisch angenommen (H==W)
    pad_h = tiles_per_side * tile_size - H
    pad_w = tiles_per_side * tile_size - W
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    tiles = []
    for ty in range(tiles_per_side):
        for tx in range(tiles_per_side):
            y0, x0 = ty * tile_size, tx * tile_size
            tiles.append(arr[y0:y0+tile_size, x0:x0+tile_size])
    return tiles, tiles_per_side, (H, W)


def merge_tiles_to_image(tiles, tiles_per_side, tile_size, full_size):
    """Liste gekachelter (tile_size,tile_size) -> (H,W), Padding abgeschnitten."""
    H, W = full_size
    canvas = np.zeros((tiles_per_side * tile_size, tiles_per_side * tile_size),
                      dtype=np.float32)
    for i, tile in enumerate(tiles):
        y0, x0 = tile_coords(i, tiles_per_side, tile_size, full_size)
        canvas[y0:y0+tile_size, x0:x0+tile_size] = tile
    return canvas[:H, :W]


def tile_flow_to_global(flow_tile, tile_idx, tiles_per_side, tile_size, full_size,
                        upsample_res):
    """
    flow_tile: (1, up_h, up_w, 2) in normierten [-1,1]-Koordinaten DER KACHEL.
    Rückgabe: (1, up_h, up_w, 2) in normierten [-1,1]-Koordinaten des GESAMTBILDS.
    Weil normiert: Kachel-Koords [-1,1] -> Bild-Koords via Offset+Skalierung.
    """
    H, W = full_size
    ty = tile_idx // tiles_per_side
    tx = tile_idx % tiles_per_side
    # Kachel-Seitenlänge relativ zum Gesamtbild
    sx = tile_size / W
    sy = tile_size / H
    flow = flow_tile.clone()
    # x-Komponente (letzte Achse Index 0): [tile] -> [global]
    flow[..., 0] = flow[..., 0] * sx + (2 * tx * sx + sx - 1)
    flow[..., 1] = flow[..., 1] * sy + (2 * ty * sy + sy - 1)
    return flow


def to_uint8(raw_arr, clip_pct=(1, 99), clahe=False):
    """Radiance -> 0..255 uint8 (Perzentil-Stretch, optional CLAHE)."""
    arr = raw_arr.copy().astype(np.float32)
    lo, hi = np.nanpercentile(arr, clip_pct)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8) * 255
    arr = np.nan_to_num(arr, nan=0).astype(np.uint8)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = c.apply(arr)
    return arr


def _as_pil(x):
    """Pfad / PIL / numpy (H,W) -> RGB uint8 PIL."""
    if isinstance(x, (str, os.PathLike)):
        return Image.open(x).convert("RGB")
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr.astype(np.uint8))


def load_radiance(p, divide_by=1000):
    """Akzeptiert Pfad UND In-Memory-Array (keep_data_in_memory=True)."""
    if isinstance(p, np.ndarray):
        return p.astype(np.float32)
    return load_image(p, divide_by=divide_by).astype(np.float32)


import tifffile
from scipy import ndimage

def sharpness_from(src) -> float:
    """Variance of Laplacian on the 760nm band. Accepts Path OR ndarray."""
    try:
        img = src if isinstance(src, np.ndarray) else tifffile.imread(src)
        img = np.asarray(img, dtype=np.float64)
        if img.ndim == 3:
            img = img[0]
        if not np.isfinite(img).any():
            return 0.0
        # NaNs would poison the Laplacian variance → ignore masked pixels
        if np.isnan(img).any():
            img = np.nan_to_num(img, nan=0.0)
        return float(np.var(ndimage.laplace(img)))
    except Exception as e:
        print(f"  ⚠️ sharpness error: {e}")
        return 0.0


import numbers
def normalize_clahe(clahe):
    if isinstance(clahe, (bool, np.bool_)):
        return 2 if bool(clahe) else False
    if isinstance(clahe, numbers.Number):     # int, float, np scalars
        return float(clahe)
    return False


# --------------------------------------------------------------------------
#  Dataset — Training (return_orig=False) UND Inferenz (return_orig=True)
# --------------------------------------------------------------------------
class SIFMatchDataset(Dataset):
    """
    EINZIGES Dataset fuer Training UND Inferenz (identische Vorverarbeitung).

    Je Eintrag (einzelne Kachel eines Bildpaars):
      idx            : eindeutiger Eintragsindex (0..len-1)  -> fuer shuffle/collate
      pair_idx       : Index des Bildpaars in rad_paths      -> fuer Mapping/Merge
      tile_idx       : linearer Kachelindex innerhalb des Pairs
      tiles_per_side : Kacheln pro Seite (1 wenn kein Tiling)
      full_size      : (H,W) des ungeteilten Originals (None wenn kein Tiling)
      id             : Paar-Identifier (self.ids[pair_idx])
      im_A / im_B    : (3,hs,ws) normalisiert, model-ready  (A=Master, B=Target)
      im_A_up/...    : (3,up_h,up_w) nur wenn upsample_preds
      orig_A/orig_B  : (tile_size,tile_size) Roh-Radiance, nur wenn return_orig

    Orientation: A = Master(Ref), B = Target (haengt von master_band ab).
    """
    def __init__(self, rad_paths, ids, master_band, hs, ws,
                 upsample_res=None, upsample_preds=False,
                 clahe=False, load_image_fn=load_radiance,
                 return_orig=True, tile_size=None,
                 sharpness_min_pct=None, sharpness_top_k=None):
        self.ids = ids
        self.master_band = master_band
        self.hs, self.ws = hs, ws
        self.upsample_res = upsample_res
        self.upsample_preds = upsample_preds
        self.clahe = normalize_clahe(clahe)
        self.load_image = load_image_fn
        self.return_orig = return_orig
        self.tile_size = tile_size

        # ---- sharpness filter on the 760 nm image of each pair ----
        # score all 760 nm sources once (Path OR ndarray, handled in sharpness_from)
        scores = np.array([sharpness_from(p760) for (p760, _p757) in rad_paths],
                          dtype=np.float64)
        self.sharpness_scores = scores  # kept for inspection / logging

        keep = np.ones(len(rad_paths), dtype=bool)
        if sharpness_top_k is not None:
            # keep the N sharpest pairs
            top_idx = np.argsort(scores)[::-1][:int(sharpness_top_k)]
            keep[:] = False
            keep[top_idx] = True
        elif sharpness_min_pct is not None:
            # drop the bottom `sharpness_min_pct` of the non-zero scores
            nonzero = scores[scores > 0]
            if nonzero.size:
                cutoff = np.percentile(nonzero, float(sharpness_min_pct))
                keep = scores >= cutoff

        rad_paths = [p for p, k in zip(rad_paths, keep) if k]
        self.ids    = [i for i, k in zip(self.ids, keep) if k]
        print(f"[sharpness] kept {int(keep.sum())}/{len(keep)} pairs "
              f"(min kept score {scores[keep].min():.1f})"
              if keep.any() else "[sharpness] WARNING: all pairs filtered out")

        self.rad_paths = rad_paths

        # ---- EINDEUTIGER Eintrag je (pair, tile). idx = linearer Eintragsindex. ----
        self._items = []   # (pair_idx, tile_idx, tiles_per_side, full_size)
        for pi, (p760, p757) in enumerate(rad_paths):
            if tile_size:
                a = self.load_image(p760)
                _, tps, full = split_image_to_tiles(
                    np.nan_to_num(a.astype(np.float32)), tile_size)
                for ti in range(tps * tps):
                    self._items.append((pi, ti, tps, full))
            else:
                self._items.append((pi, 0, 1, None))

        # ---- DIE EINE Kette (identisch Training/Inferenz) ----
        self.transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True, clahe=False)
        if upsample_preds:
            self.up_transform = get_tuple_transform_ops(resize=upsample_res, normalize=True)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        pi, ti, tps, full = self._items[idx]
        p760, p757 = self.rad_paths[pi]
        a760 = self.load_image(p760).astype(np.float32)
        a757 = self.load_image(p757).astype(np.float32)

        ref_raw, tgt_raw = (a757, a760) if self.master_band == 'Rad757' else (a760, a757)

        if self.tile_size:
            ref_tiles, _, _ = split_image_to_tiles(ref_raw, self.tile_size)
            tgt_tiles, _, _ = split_image_to_tiles(tgt_raw, self.tile_size)
            ref_raw, tgt_raw = ref_tiles[ti], tgt_tiles[ti]

        # Maske VOR dem Füllen einfangen
        ref_mask = _mask_from(ref_raw)
        tgt_mask = _mask_from(tgt_raw)

        # RoMa sieht ein maskenfreies Bild (NaN->0), KEINE harte Insel
        ref_clean = _fill_nan(ref_raw)
        tgt_clean = _fill_nan(tgt_raw)

        ref_u8 = to_uint8(ref_clean, clahe=self.clahe)
        tgt_u8 = to_uint8(tgt_clean, clahe=self.clahe)
        im_A, im_B = self.transform((_as_pil(ref_u8), _as_pil(tgt_u8)))

        out = {'idx': idx, 'pair_idx': pi, 'tile_idx': ti,
               'tiles_per_side': tps, 'full_size': full, 'id': self.ids[pi],
               'im_A': im_A, 'im_B': im_B}
        if self.upsample_preds:
            uA, uB = self.up_transform((_as_pil(ref_u8), _as_pil(tgt_u8)))
            out['im_A_up'], out['im_B_up'] = uA, uB
        if self.return_orig:
            # orig = gefüllte Radiance + Maske separat (kein NaN in orig)
            out['orig_A'] = ref_clean
            out['orig_B'] = tgt_clean
            out['mask_A'] = ref_mask
            out['mask_B'] = tgt_mask
        return out

# --------------------------------------------------------------------------
#  Collate-Funktionen
# --------------------------------------------------------------------------
def collate_fn(batch):
    """Inferenz: Liste von Dicts (variable Groessen, ungestapelt)."""
    return batch


def train_collate(batch):
    keys = ['im_A', 'im_B']
    if 'im_A_up' in batch[0]:
        keys += ['im_A_up', 'im_B_up']
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    # Metadaten parallel stapeln (für Logging/Mapping, nicht fürs Modell)
    out['pair_idx'] = torch.tensor([b['pair_idx'] for b in batch])
    out['tile_idx'] = torch.tensor([b['tile_idx'] for b in batch])
    return out

