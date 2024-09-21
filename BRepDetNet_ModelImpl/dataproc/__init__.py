from dataproc.CC3DDataModule import CC3DDataModule
from dataproc.ABCDataModule import ABCDataModule


dataModules = {
    'CC3D': CC3DDataModule,
    'ABC' : ABCDataModule,
    }

def add_available_datasets(parser):
    parser.add_argument('--dataset_type', default='CC3D', choices=dataModules.keys(), metavar='DATASET', help='dataset type (default: CC3D)')
    return parser

def load_data_module(dataset, kwargs):
    d = dataModules[dataset]
    if d is None:
        raise NotImplementedError

    return d(**kwargs)

def add_data_specific_args(dataset, parser):
    d = dataModules[dataset]
    if d is None:
        raise NotImplementedError
    return d.add_data_specific_args(parser)



