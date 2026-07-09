import os
import random
import argparse
from datetime import datetime
from torch.utils.data import random_split
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from model import STAG3D
from datasets import STDataset,collate_fn
from utils import load_config ,load_loggers

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='stnet', help='config name under ./config (e.g. stnet, her2st, skin, pcw, mouse)')
    parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='gpu id') 
    parser.add_argument('--mode', type=str, default='cv', help='cv / test / external_test / inference')
    parser.add_argument('--fold', type=int, default=0, help='')
    parser.add_argument('--model_path', type=str, default='results/xxxxx.ckpt', help='')
    parser.add_argument('--select_fold', type=int, default=0, help='')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save Lightning checkpoints.')
    parser.add_argument('--save_tensorboard', action='store_true', help='Save TensorBoard event files.')
    args = parser.parse_args()
    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def main(cfg, fold=0):
    seed=cfg.GENERAL.seed
    name=cfg.MODEL.name
    data=cfg.DATASET.type
    batch_size=cfg.TRAINING.batch_size
    print("batch_size is {batch_size}\n")
    num_epochs=cfg.TRAINING.num_epochs
    mode = cfg.GENERAL.mode
    gpus = cfg.GENERAL.gpu
    exp_id = cfg.GENERAL.exp_id
    if mode == 'cv':
        trainset = STDataset(mode='train', fold=fold, **cfg.DATASET)
        train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1, pin_memory=True, shuffle=True)
        print(f"Length of train_loader: {len(train_loader)}")
    if mode in ['external_test', 'inference']:
        testset = STDataset(mode=mode, fold=fold, test_data=cfg.GENERAL.test_name, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)
        print("external_test or inference mode")
    else:
        testset = STDataset(mode='test', fold=fold, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, collate_fn=collate_fn, num_workers=1, pin_memory=True, shuffle=False)
        print("test mode")
        print(f"Length of test_loader: {len(test_loader)}")
    loggers = load_loggers(cfg)
    log_name=f'{fold}-{name}-{data}-{seed}-{exp_id}'
    cfg.GENERAL.log_name = log_name
    model_cfg = cfg.MODEL.copy()
    del model_cfg['name']
    model = STAG3D(fold,**model_cfg)
    if mode == 'cv':
        print("cv mode")
        is_tty = sys.stdout.isatty()
        trainer = pl.Trainer(
            accelerator="gpu", 
            strategy = DDPStrategy(find_unused_parameters=True),
            devices = gpus,
            max_epochs = num_epochs,
            logger = loggers,
            enable_checkpointing=bool(cfg.GENERAL.save_checkpoints),
            check_val_every_n_epoch = 1,
            log_every_n_steps=10,
            precision=16
        )
        trainer.fit(model, train_loader, test_loader)
    elif mode=='test':
        trainer = pl.Trainer(accelerator="gpu", devices=gpus)
        print(f"model_path is {cfg.GENERAL.model_path}")
        checkpoint = torch.load(cfg.GENERAL.model_path)
        print(checkpoint.keys() )
        model.load_state_dict(checkpoint)
        trainer.test(model, test_loader)
    
if __name__ == '__main__':
    args = get_parse()   
    cfg = load_config(args.config_name)
    seed = cfg.GENERAL.seed
    fix_seed(seed)
    cfg.GENERAL.gpu = args.gpu
    cfg.GENERAL.model_path = args.model_path
    cfg.GENERAL.mode = args.mode
    cfg.GENERAL.save_checkpoints = args.save_checkpoints
    cfg.GENERAL.save_tensorboard = args.save_tensorboard
    current_day = datetime.now().strftime('%Y-%m-%d')
    cfg.GENERAL.current_day = current_day
    if args.mode == 'cv':
        num_k = cfg.TRAINING.num_k
        for fold in range(num_k):
            if fold != args.select_fold:
                continue
            main(cfg, fold=fold)
    else:
        main(cfg, args.fold)
