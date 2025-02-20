import logging

from model.BRepEd import BRepEd
from model.BRepJd import BRepJd

models = {
    'BRepEd': BRepEd,
    'BRepJd': BRepJd,
    }

def load_model(model, checkpoint_path, kwargs):
    m = models[model]
    if m is None:
        raise NotImplementedError

    if checkpoint_path:
        kwargs.pop('checkpoint_path', None)
        try:
            return m.load_from_checkpoint(checkpoint_path, **kwargs)
        except RuntimeError as e:
            logger = logging.getLogger('core')
            logger.warning('Could not load model strict. Try without strict.')
            # logger.warning(e)
            return m.load_from_checkpoint(checkpoint_path, strict=False, **kwargs)

    return m(**kwargs)

def add_model_specific_args(model, parser):
    m = models[model]
    if m is None:
        raise NotImplementedError
    return m.add_model_specific_args(parser)

def add_available_models(parser):
    parser.add_argument('--model_name', type=str, default='BRepEd', choices=models.keys(), help='models: [BRepEd, BRepJd]')
    return parser
