from __future__ import absolute_import

import os
import shutil
import inspect

# video
from .vmgn_hgnn import *

__model_factory = {
    'vmgn_hgnn': vmgn_hgnn
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    if 'save_dir' in kwargs:
        # XXX: shutil.copy and shutil.copy2 raise PermissionError, so use copyfile here.
        model_file = inspect.getfile(__model_factory[name])
        shutil.copyfile(model_file, os.path.join(os.path.abspath(kwargs['save_dir']), os.path.basename(model_file)))
    return __model_factory[name](*args, **kwargs)
