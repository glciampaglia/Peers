# set custom warnings.formatwarning

def _fmtwarn(msg, cat, fn, no, line=None):
    import sys
    for p in sys.path:
        if len(p) and fn.startswith(p):
            n = len(p) + 1
            fn = fn[n:]
            break
    warning = cat.__name__.upper()
    return '* %s: %s (in %s:%d)\n' % (warning, msg, fn, no)

import warnings as _w
_w.formatwarning = _fmtwarn

