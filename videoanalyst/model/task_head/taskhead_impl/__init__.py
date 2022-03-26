# -*- coding: utf-8 -*-

"""
NOTE: __all__ affects the from <module> import * behavior only. 
Members that are not mentioned in __all__ are still accessible from outside 
the module and can be imported with from <module> import <member>.
"""

import glob
from os.path import basename, dirname, isfile

modules = glob.glob(dirname(__file__) + "/*.py")
modules = [m for m in modules if not m.endswith(('_bak.py'))
           ]  # filter file with name ending with '_bak' (debugging)
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f)
    and not f.endswith("__init__.py") and not f.endswith("utils.py")
]