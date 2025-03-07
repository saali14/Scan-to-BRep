import os
import torch
import torch.distributed
import logging
import dataclasses

from pytorch_lightning import Trainer, plugins
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins.environments import SLURMEnvironment

from argparse import ArgumentParser
from typing import Dict, Any, List, Optional, Union
from model import load_model
from dataproc import load_data_module
from util.lightning import add_parser_arguments, configure_logger

def main(args):
    print("train-validation")
    print(torch.cuda.device_count())
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(args.log_dir, 'train'), name=args.experiment if args.experiment else args.model_name,)
    
    print(tb_logger.log_dir)
    # NOTE configure lightning logging
    logger = configure_logger(tb_logger.log_dir)

    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None}  # remove None entries

    # NOTE load model and Data set Module
    network_model = load_model(args.model_name, args.checkpoint_path, dict_args)
    dm = load_data_module(args.dataset_type, dict_args)
    dm.setup(stage='fit')
    
    # NOTE define callbacks for Sanity Check // Debugging // Logging
    callbacks = []
    if not args.disable_save:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',
            save_top_k=1,
            mode='min',
            save_last=True,
            dirpath=os.path.join(tb_logger.log_dir, 'checkpoint'),
            save_weights_only=args.save_weights,
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    trainer_kwargs = {'lightning': dataclasses.field(default_factory=dict)}
    trainer_kwargs['gpus'] = args.num_gpus
    if args.num_gpus > 1:
        store_path = os.path.join(tb_logger.log_dir, 'torch_distributed_init.store')
        trainer_kwargs['accelerator'] = 'cuda'
        #trainer_kwargs['plugins'] = DDPPlugin(init_method='file://' + store_path, find_unused_parameters=False)
    else:
        trainer_kwargs['accelerator'] = 'cuda'

    if args.mixed_precision:
        if args.num_gpus == 0:
            logging.getLogger(__name__).warn('Requested 16-bit precision but no GPUs. 16-bit precision training is not available on CPU, it has been disabled for now.')
        else:
            trainer_kwargs['precision'] = 16
    
    if args.fast_debug:
        trainer = Trainer.from_argparse_args(args,
                                             accelerator="cuda", 
                                             devices=1,
                                             max_epochs=3,
                                             log_every_n_steps=5,
                                             callbacks=callbacks,
                                             logger=tb_logger,
                                             num_sanity_val_steps=1,
                                             checkpoint_callback=not args.disable_save,
                                             limit_train_batches=16,
                                             limit_val_batches=16,
                                             )
    else:
        trainer = Trainer.from_argparse_args(args,
                                             devices=1,
                                             #gpus=args.num_gpus,
                                             accelerator="cuda",
                                             log_every_n_steps=100,
                                             callbacks=callbacks,
                                             logger=tb_logger,
                                             #checkpoint_callback=not args.disable_save,
                                             )
    
    #torch.set_num_threads(10)
    trainer.fit(network_model, dm.train_dataloader(), dm.test_dataloader())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_parser_arguments(parser)
    args = parser.parse_args()
    print (args)
    main(args)
