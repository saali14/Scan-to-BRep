import os
import torch
import torch.distributed
import logging
import dataclasses

from pytorch_lightning import Trainer, plugins
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from typing import Dict, Any, List, Optional, Union
from argparse import ArgumentParser
from util.callbacks_loggers import BndaryJncDet_ClbLogger, JncDet_ClbLogger
from util.lightning import add_parser_arguments, configure_logger
from model import load_model
from dataproc import load_data_module

def main(args):
    print(torch.cuda.is_available())
    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None}
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(args.log_dir, 'test'), 
                                             name=args.experiment if args.experiment \
                                                else args.model_name,)
    logger = configure_logger(tb_logger.log_dir)
    
    # NOTE:: Load model and Data set
    if not args.checkpoint_path:
        logger.warning('No checkpoint given!')
    model = load_model(args.model_name, args.checkpoint_path, dict_args)
    dm = load_data_module(args.dataset_type, dict_args)
    dm.setup(stage='test')
    
    # NOTE:: Configure lightning Callbacks
    test_callbacks = [BndaryJncDet_ClbLogger(tb_logger.log_dir, withNMS=False) if not args.jncOnly \
                       else JncDet_ClbLogger(tb_logger.log_dir, withNMS=False),]
    
    # NOTE:: Start Testing
    trainer = Trainer.from_argparse_args(args,
                                         accelerator='cuda', 
                                         devices=1,
                                         logger=tb_logger, 
                                         callbacks=test_callbacks, 
                                         limit_test_batches=1.)
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_parser_arguments(parser, require_checkpoint=True)
    args = parser.parse_args()
    print (args)
    # test
    main(args)
