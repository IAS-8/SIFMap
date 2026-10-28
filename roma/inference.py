import numpy as np
import os
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from data.utils import load_image, save
from roma.model import RomaMatcher
from roma.data_module import SIFMatchDataModule
from romatch.utils.utils import get_tuple_transform_ops


def create_dataloader(rad_paths, ids, master_band, hs, ws,
                      upsample_res=None, upsample_preds=False,
                      clahe=False, batch_size=1, num_workers=2,
                      load_image_fn=load_image, tile_size=None):

    dm = SIFMatchDataModule(
            base_dir='', train_dir=[], val_dir=[], 
            cam760_regex='', cam757_regex='', datatake_regex='',
            master_band=master_band,
            hs=hs, ws=ws, upsample_res=upsample_res, upsample_preds=upsample_preds,
            clahe=clahe, divide_by=1000,
            batch_size=batch_size, num_workers=num_workers,
            tile_size=tile_size, return_orig=True,
        )
    loader = dm.inference_dataloader(pairs=rad_paths)
    return loader


def _to_device(obj, device):
    """Verschiebt rekursiv alle torch.Tensor in dict/list/tuple auf das Device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    return obj

def iterate_dataset(paths, ids, batch_size, out_paths, data,
                    pairing_master_band='Rad757', clahe=False,
                    module=None, ckpt_path=None, device=None,
                    num_workers=2, roma_config=None, **kwargs):

    if roma_config is None:
        roma_config = dict()

    weights = roma_config.get('ckpt', None)

    # ---- Device auflösen ----
    print('DEVCE', device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f'Running RoMa on {device}')

    # ---- Matcher bauen + auf Device verschieben ----
    if module is not None:
        module = module.to(device)
        module.device_ = device
        matcher = RomaMatcher(module, device=device, symmetric=False, **kwargs)
    elif weights:
        matcher = RomaMatcher.from_checkpoint(weights, device=device, **kwargs)
        matcher.module = matcher.module.to(device)
        matcher.module.device_ = device
        matcher.device = device
    else:
        matcher = RomaMatcher.vanilla(clahe=clahe, device=device, symmetric=False, **kwargs)
        matcher.module = matcher.module.to(device)
        matcher.module.device_ = device
        matcher.device = device

    clahe = matcher.clahe
    on_matched = matcher.on_matched

    print(f'RUNNING ROMA WITH clahe={clahe} and on_matched={on_matched}')

    roma = matcher.module.roma
    upsample_preds = bool(roma.upsample_preds)
    hs, ws = roma.h_resized, roma.w_resized
    upsample_res = tuple(roma.upsample_res) if upsample_preds else None

    def load_radiance(p, divide_by=1000):
        if isinstance(p, np.ndarray):
            return p.astype(np.float32)
        return load_image(p, divide_by=divide_by).astype(np.float32)

    loader = create_dataloader(
        paths, ids,
        master_band=pairing_master_band,
        hs=hs, ws=ws,
        upsample_res=upsample_res,
        upsample_preds=upsample_preds,
        clahe=clahe,
        batch_size=1,
        num_workers=num_workers,
        load_image_fn=load_radiance,
        tile_size=matcher.module.hparams.get('tile_size', None),
    )

    remove = []
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            for item in batch:
                item['out'] = out_paths[item['idx']]

            for res in matcher.match_batch(batch, master_band=pairing_master_band, symmetric=False):
                if res['status'] != 0 or res['ref'] is None:
                    print(item['out'], ' not successfully matched')
                    remove.append(res['idx'])
                    continue
                
                print(res['idx'], data[res['idx']].get('FAILURE', 'none'))
                out = out_paths[res['idx']]

                if res['is_757_master']:
                    rad760_w, rad757_w = res['tgt_warped'], res['ref']
                    mask760_w, mask757_w = res['tgt_mask'], res['ref_mask']
                else:
                    rad760_w, rad757_w = res['ref'], res['tgt_warped']
                    mask760_w, mask757_w = res['ref_mask'], res['tgt_mask']

                rad760_w[mask760_w] = np.nan
                rad757_w[mask757_w] = np.nan
                save([rad760_w, rad757_w], out)

                data[res['idx']]['Rad760'] = out[0]
                data[res['idx']]['Rad757'] = out[1]

    return data, remove

