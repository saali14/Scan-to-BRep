import logging
import os
import subprocess

from pytorch_lightning import Trainer
from dataproc import add_available_datasets, add_data_specific_args
from model import add_available_models, add_model_specific_args
from util.file_helper import create_dir


def add_parser_arguments(parser, require_checkpoint=False):
    # add trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    # figure out which model and dataset to use
    parser = add_available_models(parser)
    parser = add_available_datasets(parser)

    # parse model and data name
    temp_args, _ = parser.parse_known_args()

    # add model and data specific arguments
    parser = add_model_specific_args(temp_args.model_name, parser)
    parser = add_data_specific_args(temp_args.dataset_type, parser)

    # checkpoint to load weights and hyperparameter
    parser.add_argument('--checkpoint_path', type=str, metavar='PATH', required=require_checkpoint, help='Checkpoint to load from.')
    parser.add_argument('--experiment', type=str, help='Experiment name. Used for log path. Default to model_name.')
    parser.add_argument('--disable_save', action='store_true', help='Do not save model to file.)')
    parser.add_argument('--save_weights', action='store_true', help='Only save model weights. Otherwise, save full model.')
    parser.add_argument('--fast_debug', action='store_true', help='Run small fraction of steps and epochs.')
    parser.add_argument('--log_dir', type=str, metavar='PATH', default='../logs', help='Directory to save logs.')
    parser.add_argument('--num_gpus', type=int, default=1, help='> 1 for multi-gpu trainer.')
    parser.add_argument('--mixed_precision', default=False, action="store_true", help='enable 16 bit precision for model precision ') 
    parser.add_argument('--jncOnly',action='store_true', help='Only to detect Junction.') 
    return parser

def configure_logger(log_path, module='pytorch_lightning', log_level=logging.INFO):
    logger = logging.getLogger(module)
    logger.setLevel(log_level)
    create_dir(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, "core.log"))
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_git_commit(logger)

    # log general info to same file
    core_logger = logging.getLogger('core')
    core_logger.setLevel(log_level)
    core_logger.addHandler(file_handler)
    return core_logger

def log_git_commit(logger):
    try:
        commit = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=tformat:%H %ad', '--date=short'], universal_newlines=True).strip().split()
        logger.info('Run on Commit {} from {}.'.format(*commit))

    except subprocess.CalledProcessError:
        pass
