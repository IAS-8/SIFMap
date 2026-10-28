# SIFMap/roma/data_module.py
import os
import re
import numpy as np
import pytorch_lightning as L
from torch.utils.data import DataLoader
from roma.data import SIFMatchDataset, train_collate, load_radiance, collate_fn, train_collate


def collect_pairs_from_dir(raw_dir, cam760_regex, cam757_regex, datatake_regex):
    """Spiegelt SIFcam.read_paths: datatake_regex gruppiert, camera_regex trennt Bänder."""
    fils = os.listdir(raw_dir)
    datatakes = {}
    for f in fils:
        m = re.findall(datatake_regex, f)
        if len(m) == 1:
            datatakes.setdefault(m[0], []).append(f)
    pairs = []
    for group in datatakes.values():
        f760 = [f for f in group if re.findall(cam760_regex, f)]
        f757 = [f for f in group if re.findall(cam757_regex, f)]
        if len(f760) == 1 and len(f757) == 1:
            pairs.append((os.path.join(raw_dir, f760[0]),
                          os.path.join(raw_dir, f757[0])))
    return pairs


def collect_pairs(raw_dirs, cam760_regex, cam757_regex, datatake_regex):
    pairs = []
    for d in raw_dirs:
        p = collect_pairs_from_dir(d, cam760_regex, cam757_regex, datatake_regex)
        print(f'{d}: {len(p)} Paare')
        pairs.extend(p)
    return pairs


class SIFMatchDataModule(L.LightningDataModule):
    """
    Training-Seite. Nutzt SIFMatchDataset DIREKT (return_orig=False).
    hs/ws/upsample_res/upsample_preds kommen vom Modell (main) -> keine zweite Quelle.
    """
    def __init__(self, base_dir, train_dir, val_dir, cam760_regex, cam757_regex,
                 datatake_regex, master_band, hs, ws, upsample_res, upsample_preds,
                 clahe, divide_by, batch_size, num_workers, return_orig=False,
                 tile_size=None, on_matched=False, sharpness_min_pct=None, sharpness_top_k=None):
        super().__init__()

        radiance_subdir = 'RADIANCE_nonmatched' if not on_matched else 'RADIANCE'
        def _resolve(p):
            # base_dir + flight + radiance_subdir; absolute Eingaben bleiben unangetastet
            if os.path.isabs(p):
                return p
            return os.path.join(base_dir, p, radiance_subdir)
        self.train_dir = [_resolve(p) for p in train_dir]
        self.val_dir = [_resolve(p) for p in val_dir] if val_dir else []

        self.cam760_regex = cam760_regex
        self.cam757_regex = cam757_regex
        self.datatake_regex = datatake_regex
        self.master_band = master_band
        self.hs, self.ws = hs, ws
        self.upsample_res = upsample_res
        self.upsample_preds = upsample_preds
        self.clahe = clahe
        self.divide_by = divide_by
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_orig = return_orig
        self.tile_size = tile_size
        self.sharpness_top_k = sharpness_top_k
        self.sharpness_min_pct = sharpness_min_pct

        self.train_pairs = None
        self.val_pairs = None

    def _load_radiance(self, p):
        if isinstance(p, np.ndarray):
            return p.astype(np.float32)
        from data.utils import load_image
        return load_image(p, divide_by=self.divide_by).astype(np.float32)

    def setup(self, stage=None):
        common = dict(cam760_regex=self.cam760_regex,
                      cam757_regex=self.cam757_regex,
                      datatake_regex=self.datatake_regex)
        if stage in (None, 'fit', 'test'):
            self.train_pairs = collect_pairs(self.train_dir, **common)
        if stage in (None, 'fit', 'validate'):
            self.val_pairs = collect_pairs(self.val_dir, **common) if self.val_dir else []

    def _ds(self, pairs):
        return SIFMatchDataset(
            pairs, list(range(len(pairs))),
            master_band=self.master_band,
            hs=self.hs, ws=self.ws,
            upsample_res=self.upsample_res, upsample_preds=self.upsample_preds,
            clahe=self.clahe, load_image_fn=self._load_radiance,
            return_orig=self.return_orig, tile_size=self.tile_size,
            sharpness_top_k=self.sharpness_top_k, sharpness_min_pct=self.sharpness_min_pct    
        )

    def train_dataloader(self):
        return DataLoader(self._ds(self.train_pairs), batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=train_collate, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        if not self.val_pairs:
            return None
        return DataLoader(self._ds(self.val_pairs), batch_size=1, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=train_collate, pin_memory=True)

    def inference_dataloader(self, pairs=None):
        if pairs is None:
            if self.train_pairs is None:
                self.setup('test')

            pairs = self.train_pairs

        ds = self._ds(pairs)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)
