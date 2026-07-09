import os

from datetime import datetime
import yaml
from addict import Dict
import numpy as np
import pandas as pd

import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

def load_config(config_name: str):
    config_path = os.path.join('./config', f'{config_name}.yaml')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    
    return Dict(config)

def load_loggers(cfg: Dict):
    log_path = os.path.join(cfg.GENERAL.log_path, cfg.GENERAL.current_day)

    csv_logger = CSVLogger(
        log_path,
        name = cfg.GENERAL.log_name)
    
    loggers = [csv_logger]
    if cfg.GENERAL.get('save_tensorboard', False):
        tb_logger = TensorBoardLogger(
            log_path,
            name = cfg.GENERAL.log_name
        )
        loggers.append(tb_logger)
    
    return loggers

def load_callbacks(cfg: Dict):
    log_path = os.path.join(cfg.GENERAL.log_path, cfg.GENERAL.current_day)
    
    Mycallbacks = []
    
    target = cfg.TRAINING.early_stopping.monitor
    patience = cfg.TRAINING.early_stopping.patience
    mode = cfg.TRAINING.early_stopping.mode
    early_stop_callback = EarlyStopping(
        monitor=target,
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode=mode
    )
    Mycallbacks.append(early_stop_callback)
    fname = cfg.GENERAL.log_name + '-{epoch:02d}-{valid_loss:.4f}-{R:.4f}'
    checkpoint_callback = ModelCheckpoint(monitor = target,
                                    dirpath = str(log_path) + '/' + cfg.GENERAL.log_name,
                                    filename=fname,
                                    verbose = True,
                                    save_last = False,
                                    save_top_k = 1,
                                    mode = mode,
                                    save_weights_only = True)
    Mycallbacks.append(checkpoint_callback)
        
    return Mycallbacks
