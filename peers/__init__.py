# set custom warnings.formatwarning

def _fmtwarn(*args):
    msg = args[0]
    return '* WARNING: %s\n' % msg.args[0]

import warnings as _w
_w.formatwarning = _fmtwarn

